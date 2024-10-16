# Copyright 2022 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Any, Dict, List, Sequence

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.simclr import *
from solo.methods.base import BaseMethod
from solo.utils.misc import omegaconf_select

import numpy as np

def LN(x: torch.Tensor, eps: float = 1e-5):
    mu = x.mean(dim=-1, keepdim=True)
    x = x - mu
    std = x.std(dim=-1, keepdim=True)
    x = x / (std + eps)
    return x, mu, std

class SimCLR(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements SimCLR (https://arxiv.org/abs/2002.05709).

        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of the projected features.
                proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
                temperature (float): temperature for the softmax in the contrastive loss.
        """

        super().__init__(cfg)

        self.temperature: float = cfg.method_kwargs.temperature

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim

        self.non_neg = cfg.method_kwargs.non_neg
        self.tau = cfg.method_kwargs.tau
        self.loss_type = cfg.method_kwargs.loss_type
        # projector
        self.projector = nn.Sequential(
                nn.Linear(self.features_dim, proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, proj_output_dim),
            )
        if self.loss_type == 'SAE':

            self.pre_bias = torch.nn.Parameter(torch.zeros(2048))
            self.encoder = torch.nn.Linear(2048,2048, bias=False)
            self.latent_bias = torch.nn.Parameter(torch.zeros(2048))
            self.decoder = torch.nn.Linear(2048,2048, bias=False)
            self.slice = 256




        
        if 'tri' in self.loss_type:
            self.scale_param = nn.Parameter(torch.randn(proj_output_dim)*0.1 )
        else:
            self.scale_param = None

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(SimCLR, SimCLR).add_and_assert_specific_cfg(cfg)
        cfg.method_kwargs.non_neg = omegaconf_select(cfg, "method_kwargs.non_neg", None)
        cfg.method_kwargs.tau = omegaconf_select(cfg, "method_kwargs.tau", 0.0)
        cfg.method_kwargs.loss_type = omegaconf_select(cfg, "method_kwargs.loss_type", 'xent')
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.temperature")



        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"name": "projector", "params": self.projector.parameters()}]
        if self.loss_type == 'SAE':
            extra_learnable_params2 = [{"name": "decoder", "params": self.decoder.parameters(),"lr": 0.3}]
            extra_learnable_params3 = [{"name": "encoder", "params": self.encoder.parameters()}]
            extra_learnable_params4 = [{"name": "pre_bias", "params": self.pre_bias}]
            extra_learnable_params5 = [{"name": "latent_bias", "params": self.latent_bias}]
            return super().learnable_params + extra_learnable_params+extra_learnable_params2+extra_learnable_params3+extra_learnable_params4+extra_learnable_params5
        else:
            return super().learnable_params + extra_learnable_params
    def forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs the forward pass of the backbone and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent
                and the projected features.
        """
        if self.loss_type == 'SAE':
            grads_backbone = False
        else:
            grads_backbone = True


        with torch.set_grad_enabled(grads_backbone):
            out = super().forward(X)
        
        with torch.set_grad_enabled(grads_backbone):
            z = self.projector(out["feats"])

            z = F.normalize(z, dim=-1)

            original_input = z.clone()

        if self.loss_type == 'SAE':
        #z,mu,std = LN(z)

            z = z - self.pre_bias
            latents = torch.nn.functional.linear(z,self.encoder.weight,self.latent_bias)
            topk = torch.topk(latents, self.slice,dim=-1)
            values = torch.nn.functional.relu(topk.values)
            latents = torch.zeros_like(latents)
            latents.scatter_(-1,topk.indices,values)
            reconstruction = self.decoder(latents) + self.pre_bias
            #reconstruction = reconstruction*std + mu



        out.update({"z": z})
        if self.loss_type == 'SAE':
            out.update({"reconstruction":reconstruction})
            out.update({"original_input":original_input})

        return out

    def multicrop_forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs the forward pass for the multicrop views.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[]: a dict containing the outputs of the parent
                and the projected features.
        """

        out = super().multicrop_forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for SimCLR reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of SimCLR loss and classification loss.


        """




        indexes = batch[0]

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        z = torch.cat(out["z"])

        # ------- non-negative -------
        supported_non_neg_list = [None, 'relu', 'rep_relu', 'gelu', 'sigmoid', 'softplus', 'exp', 'leakyrelu']
        assert self.non_neg in supported_non_neg_list, f"non_neg {self.non_neg} should be one of {supported_non_neg_list}"

        if self.non_neg is None:
            # leave blank to restore original contrastive learning
            pass # z=z
        if self.non_neg == 'relu': 
            z = F.relu(z)
        if self.non_neg == 'rep_relu': 
            # reparameterized ReLU: forward is ReLU and backward is GELU
            # more friendly to backpropagation and avoids dead neurons 
            gelu_z = F.gelu(z)
            z = gelu_z - gelu_z.data + F.relu(z).data

        # some other choices of activation functions that are not non-negative
        if self.non_neg == 'gelu':
            z = F.gelu(z)
        if self.non_neg == 'sigmoid':
            z = F.sigmoid(z)
        if self.non_neg == 'softplus':
            z = F.softplus(z)
        if self.non_neg == 'exp':
            z = torch.exp(z)
        if self.non_neg == 'leakyrelu':
            z = F.leaky_relu(z)

        # ------- contrastive loss -------
        n_augs = self.num_large_crops + self.num_small_crops
        indexes = indexes.repeat(n_augs)
        z = F.normalize(z, dim=-1)



        if self.loss_type == 'xent':

            nce_loss = simclr_loss_func(
                z,
                indexes=indexes,
                temperature=self.temperature,
            )


            #z_sums = torch.mean(z,dim=0)
            z_sums = torch.mean(torch.abs(z),dim=0)
            var_loss = -1*torch.mean(z_sums)


        
            self.log("train_nce_loss", nce_loss, on_epoch=True, sync_dist=True)
            self.log("train_var_loss", var_loss, on_epoch=True, sync_dist=True)



            if batch_idx == 0:
                _, X, targets = batch
                targets = targets.repeat(n_augs)            
                log_stats(self.log, z, targets)
            
            if self.non_neg is None:
                return nce_loss +class_loss
            else:
                return nce_loss + class_loss + var_loss 


        elif self.loss_type == 'SAE':



            reconstruction = torch.cat(out["reconstruction"])
            original_input = torch.cat(out["original_input"])



            mse_loss = ((reconstruction - original_input) ** 2).mean(dim=1) / (original_input**2).mean(dim=1)
            mse_loss = mse_loss.mean()

            self.log("train_sparse_loss", mse_loss, on_epoch=True, sync_dist=True)



            if batch_idx == 0:
                _, X, targets = batch
                targets = targets.repeat(n_augs)
                log_stats(self.log, reconstruction, targets)
            return mse_loss



    def validation_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for SimCLR reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of SimCLR loss and classification loss.


        """




        indexes = batch[0]

        out = super().validation_step(batch, batch_idx)

        if self.loss_type == 'SAE':
            reconstruction = out["reconstruction"]
            original_input = out["original_input"]

            mse_loss = ((reconstruction - original_input) ** 2).mean(dim=1) / (original_input**2).mean(dim=1)
            mse_loss = mse_loss.mean().detach()


            self.log("val_sparse_loss", mse_loss, on_epoch=True, sync_dist=True)
            return mse_loss

def log_stats(logger, z, targets=None):

        # --------- track feature statistics ---------
                
        # ratio of non-negative values (fact check that outputs are all non-negative)
        def non_neg(z): 
            return (z>=0).float().mean()
        logger("non_neg_ratio", non_neg(z), on_epoch=True, sync_dist=True)

        # ratio of activated dimensions along minibatch samples
        def act_dim(z): 
            return (z.abs().mean(dim=0)>0).float().sum()
        logger("num_active_dim", act_dim(z), on_epoch=True, sync_dist=True)

        # avereage ratio of zero-values per sample
        def sparsity(z):
            return 1 - (z.abs()>1e-5).float().mean()
        logger("sparse_vals_ratio", sparsity(z), on_epoch=True, sync_dist=True)

        # effective rank of the feature matrix
        def erank(z):
            z = z.float()
            s = torch.linalg.svdvals(z)
            s = s / s.sum()
            return -torch.sum(s * torch.log(s + 1e-6))
        logger("effective_rank", erank(z), on_step=False, on_epoch=True, sync_dist=True)


        # semantic consistency
        def semantic_consistency(features, labels, eps=1e-5, take_abs=False, topk=False):
            # find activated dimensions
            active_dim_mask = features.abs().sum(0)>0
            features  = features[:, active_dim_mask]
            features = F.normalize(features, dim=1)

            # if topk:
            #     sorted, indices = torch.sort(features.sum(dim=0), descending=True)
            #     indices = indices[sorted>1]
            #     features = features[:, indices]

            acc_per_dim = []
            for i in range(features.shape[1]): # sweep each feature dimension
                # only account for activated samples
                active_sample_mask = features.abs()[:,i] > eps
                labels_selected = labels[active_sample_mask]
                try:
                    dist = labels_selected.bincount()
                    dist = dist / dist.sum() # normalize to 1
                    acc = dist.max().item() # ratio of the most frequent label among activatived samples
                    acc_per_dim.append(acc)
                except:
                    pass # sometimes it goes into err
            mean_acc =  torch.tensor(acc_per_dim).mean()
            return mean_acc

        if targets is not None:
            logger("semantic_consistency", semantic_consistency(z, targets), on_epoch=True, sync_dist=True)
        
        def orthogonality(features, eps=1e-5):
            features  = features[:,features.abs().sum(0)>0]
            n, d = features.shape
            features = F.normalize(features, dim=0)
            corr = features.T @ features
            err = (corr - torch.eye(d, device=features.device)).abs()
            err = err.mean()
            return err

        logger("orthogonality", orthogonality(z), on_epoch=True, sync_dist=True)
