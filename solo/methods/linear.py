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

import numpy as np
import torchvision.transforms as transforms
import logging
from typing import Any, Callable, Dict, List, Tuple, Union
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from solo.utils.lars import LARS
from solo.utils.metrics import accuracy_at_k, weighted_mean
from solo.methods.metric import disentangle,sparsity,cluster_acc
from solo.utils.misc import (
    omegaconf_select,
    param_groups_layer_decay,
    remove_bias_and_norm_from_weight_decay,
)
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, ReduceLROnPlateau
from solo.utils.robust_loss import SCELoss


def LN(x: torch.Tensor, eps: float = 1e-5):
    mu = x.mean(dim=-1, keepdim=True)
    x = x - mu
    std = x.std(dim=-1, keepdim=True)
    x = x / (std + eps)
    return x, mu, std


class LinearModel(pl.LightningModule):
    _OPTIMIZERS = {
        "sgd": torch.optim.SGD,
        "lars": LARS,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }
    _SCHEDULERS = [
        "reduce",
        "warmup_cosine",
        "step",
        "exponential",
        "none",
    ]

    def __init__(
        self,
        backbone: nn.Module,
        cfg: omegaconf.DictConfig,
        loss_func: Callable = None,
        mixup_func: Callable = None,
        scale_param=0,
    ):
        """Implements linear and finetune evaluation.

        .. note:: Cfg defaults are set in init by calling `cfg = add_and_assert_specific_cfg(cfg)`

        backbone (nn.Module): backbone architecture for feature extraction.
        Cfg basic structure:
            data:
                num_classes (int): number of classes in the dataset.
            max_epochs (int): total number of epochs.

            optimizer:
                name (str): name of the optimizer.
                batch_size (int): number of samples in the batch.
                lr (float): learning rate.
                weight_decay (float): weight decay for optimizer.
                kwargs (Dict): extra named arguments for the optimizer.
            scheduler:
                name (str): name of the scheduler.
                min_lr (float): minimum learning rate for warmup scheduler. Defaults to 0.0.
                warmup_start_lr (float): initial learning rate for warmup scheduler.
                    Defaults to 0.00003.
                warmup_epochs (float): number of warmup epochs. Defaults to 10.
                lr_decay_steps (Sequence, optional): steps to decay the learning rate if scheduler is
                    step. Defaults to None.
                interval (str): interval to update the lr scheduler. Defaults to 'step'.

            finetune (bool): whether or not to finetune the backbone. Defaults to False.

            performance:
                disable_channel_last (bool). Disables channel last conversion operation which
                speeds up training considerably. Defaults to False.
                https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html#converting-existing-models

        loss_func (Callable): loss function to use (for mixup, label smoothing or default). Defaults to None
        mixup_func (Callable, optional). function to convert data and targets with mixup/cutmix.
            Defaults to None.
        """

        super().__init__()

        # add default values and assert that config has the basic needed settings
        cfg = self.add_and_assert_specific_cfg(cfg)

        # backbone
        self.backbone = backbone
        if hasattr(self.backbone, "inplanes"):
            features_dim = self.backbone.inplanes
        else:
            features_dim = self.backbone.num_features

        self.feature_selection:int = cfg.feature_selection
        if self.feature_selection is not None:
             features_dim = self.feature_selection


        features_dim=2048
       
        # classifier
        self.classifier = nn.Linear(features_dim, cfg.data.num_classes)  # type: ignore

        # mixup/cutmix function
        self.mixup_func: Callable = mixup_func

        if loss_func is None:
            loss_func = nn.CrossEntropyLoss()
        self.loss_func = loss_func

        # training related
        self.max_epochs: int = cfg.max_epochs
        self.accumulate_grad_batches: Union[int, None] = cfg.accumulate_grad_batches

        # optimizer related
        self.optimizer: str = cfg.optimizer.name
        self.batch_size: int = cfg.optimizer.batch_size
        self.lr: float = cfg.optimizer.lr
        self.weight_decay: float = cfg.optimizer.weight_decay
        self.extra_optimizer_args: Dict[str, Any] = cfg.optimizer.kwargs
        self.exclude_bias_n_norm_wd: bool = cfg.optimizer.exclude_bias_n_norm_wd
        self.layer_decay: float = cfg.optimizer.layer_decay

        # scheduler related
        self.scheduler: str = cfg.scheduler.name
        self.lr_decay_steps: Union[List[int], None] = cfg.scheduler.lr_decay_steps
        self.min_lr: float = cfg.scheduler.min_lr
        self.warmup_start_lr: float = cfg.scheduler.warmup_start_lr
        self.warmup_epochs: int = cfg.scheduler.warmup_epochs
        self.scheduler_interval: str = cfg.scheduler.interval
        assert self.scheduler_interval in ["step", "epoch"]
        if self.scheduler_interval == "step":
            logging.warn(
                f"Using scheduler_interval={self.scheduler_interval} might generate "
                "issues when resuming a checkpoint."
            )



        #loss_func = 'SCE'

        if loss_func is None:
            self.loss_func = nn.CrossEntropyLoss()
        elif loss_func == 'SCE':
            self.loss_func = SCELoss(alpha=6.0, beta=0.1, num_classes=100)

        # if finetuning the backbone

        self.finetune: bool = cfg.finetune

        # for performance
        self.no_channel_last = cfg.performance.disable_channel_last

        if not self.finetune:
            for param in self.backbone.parameters():
                param.requires_grad = False



        self.i_index = None
        self.u = None
        #self.i_index = torch.load('importance.pt')
        
        self.projector = nn.Sequential(
                nn.Linear(512, 16384),
                nn.ReLU(),
                nn.Linear(16384,2048),
        )
        self.non_neg = cfg.non_neg
        self.SAE = cfg.SAE
        self.uniform_noise = cfg.uniform_noise
        self.gaussian_noise = cfg.gaussian_noise



    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        # default parameters for optimizer
        cfg.optimizer.exclude_bias_n_norm_wd = omegaconf_select(
            cfg, "optimizer.exclude_bias_n_norm_wd", False
        )
        # default for extra optimizer kwargs (use pytorch's default if not available)
        cfg.optimizer.kwargs = omegaconf_select(cfg, "optimizer.kwargs", {})
        cfg.optimizer.layer_decay = omegaconf_select(cfg, "optimizer.layer_decay", 0.0)

        # whether or not to finetune the backbone
        cfg.finetune = omegaconf_select(cfg, "finetune", False)

        
        cfg.non_neg = omegaconf_select(cfg, "non_neg", None)
        cfg.SAE = omegaconf_select(cfg, "SAE", None)
        cfg.uniform_noise = omegaconf_select(cfg, "uniform_noise",0.0)
        cfg.gaussian_noise = omegaconf_select(cfg, "gaussian_noise", 0.0)

        # default for acc grad batches
        cfg.accumulate_grad_batches = omegaconf_select(cfg, "accumulate_grad_batches", None)

        # default parameters for the scheduler
        cfg.scheduler.lr_decay_steps = omegaconf_select(cfg, "scheduler.lr_decay_steps", None)
        cfg.scheduler.min_lr = omegaconf_select(cfg, "scheduler.min_lr", 0.0)
        cfg.scheduler.warmup_start_lr = omegaconf_select(cfg, "scheduler.warmup_start_lr", 3e-5)
        cfg.scheduler.warmup_epochs = omegaconf_select(cfg, "scheduler.warmup_epochs", 10)
        cfg.scheduler.interval = omegaconf_select(cfg, "scheduler.interval", "step")

        # default parameters for performance optimization
        cfg.performance = omegaconf_select(cfg, "performance", {})
        cfg.performance.disable_channel_last = omegaconf_select(
            cfg, "performance.disable_channel_last", False
        )

        cfg.feature_selection = omegaconf_select(cfg, "selected_dims", None)

        return cfg

    def configure_optimizers(self) -> Tuple[List, List]:
        """Collects learnable parameters and configures the optimizer and learning rate scheduler.

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """

        if self.layer_decay > 0:
            assert self.finetune, "Only with use layer weight decay with finetune on."
            msg = "Method should implement no_weight_decay() that returns a set of parameter names to ignore from weight decay"
            assert hasattr(self.backbone, "no_weight_decay"), msg

            learnable_params = param_groups_layer_decay(
                self.backbone,
                self.weight_decay,
                no_weight_decay_list=self.backbone.no_weight_decay(),
                layer_decay=self.layer_decay,
            )
            learnable_params.append({"name": "classifier", "params": self.classifier.parameters()})
        else:
            learnable_params = (
                self.classifier.parameters()
                if not self.finetune
                else [
                    {"name": "backbone", "params": self.backbone.parameters()},
                    {"name": "classifier", "params": self.classifier.parameters()},
                ]
            )

        # exclude bias and norm from weight decay
        if self.exclude_bias_n_norm_wd:
            learnable_params = remove_bias_and_norm_from_weight_decay(learnable_params)

        assert self.optimizer in self._OPTIMIZERS
        optimizer = self._OPTIMIZERS[self.optimizer]

        optimizer = optimizer(
            learnable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.extra_optimizer_args,
        )

        # select scheduler
        if self.scheduler == "none":
            return optimizer

        if self.scheduler == "warmup_cosine":
            max_warmup_steps = (
                self.warmup_epochs * (self.trainer.estimated_stepping_batches / self.max_epochs)
                if self.scheduler_interval == "step"
                else self.warmup_epochs
            )
            max_scheduler_steps = (
                self.trainer.estimated_stepping_batches
                if self.scheduler_interval == "step"
                else self.max_epochs
            )
            scheduler = {
                "scheduler": LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=max_warmup_steps,
                    max_epochs=max_scheduler_steps,
                    warmup_start_lr=self.warmup_start_lr if self.warmup_epochs > 0 else self.lr,
                    eta_min=self.min_lr,
                ),
                "interval": self.scheduler_interval,
                "frequency": 1,
            }
        elif self.scheduler == "reduce":
            scheduler = ReduceLROnPlateau(optimizer)
        elif self.scheduler == "step":
            scheduler = MultiStepLR(optimizer, self.lr_decay_steps, gamma=0.1)
        elif self.scheduler == "exponential":
            scheduler = ExponentialLR(optimizer, self.weight_decay)
        else:
            raise ValueError(
                f"{self.scheduler} not in (warmup_cosine, cosine, reduce, step, exponential)"
            )

        return [optimizer], [scheduler]

    def forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs forward pass of the frozen backbone and the linear layer for evaluation.

        Args:
            X (torch.tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing features and logits.
        """
        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)


         

        with torch.set_grad_enabled(self.finetune):
            
            feats = self.backbone(X)
            feats = self.projector(feats)
            if self.non_neg is not None:
                feats = torch.nn.functional.relu(feats)

            if self.SAE is not None:
                self.slice=256
                z = feats
                #z,mu,std = LN(feats)
                z = z - self.pre_bias
                latents = torch.nn.functional.linear(z,self.encoder.weight,self.latent_bias)
                topk = torch.topk(latents, self.slice,dim=-1)
                values = torch.nn.functional.relu(topk.values)
                latents = torch.zeros_like(latents)
                latents.scatter_(-1,topk.indices,values)
                feats = latents





        feats_c = feats.clone()



        logits = self.classifier(feats)

        return {"logits": logits, "input":X,"feats": feats_c.detach().cpu()}

    def shared_step(
        self, batch: Tuple, batch_idx: int
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs operations that are shared between the training nd validation steps.

        Args:
            batch (Tuple): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
                batch size, loss, accuracy @1 and accuracy @5.
        """

        X, target = batch

        metrics = {"batch_size": X.size(0)}
        if self.training and self.mixup_func is not None:
            X, target = self.mixup_func(X, target)
            out = self(X)["logits"]
            loss = self.loss_func(out, target)
            metrics.update({"loss": loss})
        else:
            out = self(X)["logits"]
            #loss = F.cross_entropy(out, target)
            loss = self.loss_func(out, target)
            acc1, acc5 = accuracy_at_k(out, target, top_k=(1, 5))
            metrics.update({"loss": loss, "acc1": acc1, "acc5": acc5})
            metrics.update({"targets":target})
            metrics.update({"inputs":X})
            metrics.update({"feats":self(X)["feats"]})
            metrics.update({"logits":self(X)["logits"]})



       


        return metrics

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Performs the training step for the linear eval.

        Args:
            batch (torch.Tensor): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            torch.Tensor: cross-entropy loss between the predictions and the ground truth.
        """



        # set backbone to eval mode
        if not self.finetune:
            self.backbone.eval()

        out = self.shared_step(batch, batch_idx)

        log = {"train_loss": out["loss"]}
        if self.mixup_func is None:
            log.update({"train_acc1": out["acc1"], "train_acc5": out["acc5"]})

        self.log_dict(log, on_epoch=True, sync_dist=True)
        return {"loss":out["loss"],"targets":out["targets"],"feats":out["feats"],}

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, Any]:
        """Performs the validation step for the linear eval.

        Args:
            batch (torch.Tensor): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            Dict[str, Any]:
                dict with the batch_size (used for averaging),
                the classification loss and accuracies.
        """
        if self.uniform_noise > 0:
            sa,sb,sc,sd = batch[0].size()
            noise  = torch.rand((sa,sb,sc,sd)).to("cuda")
            batch = (batch[0]+ noise * self.uniform_noise,batch[1])
        if self.gaussian_noise > 0 :
            sa,sb,sc,sd = batch[0].size()
            noise  = torch.randn((sa,sb,sc,sd)).to("cuda")
            batch = (batch[0]+ noise * self.gaussian_noise,batch[1])



        out = self.shared_step(batch, batch_idx)


        results = {
            "batch_size": out["batch_size"],
            "val_loss": out["loss"],
            "val_acc1": out["acc1"],
            "val_acc5": out["acc5"],
            "targets":out["targets"],
            "feats": out["feats"]

        }
        return results
   
    def test_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, Any]:
        """Performs the validation step for the linear eval.

        Args:
            batch (torch.Tensor): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            Dict[str, Any]:
                dict with the batch_size (used for averaging),
                the classification loss and accuracies.
        """

        if self.uniform_noise > 0:
            print(self.uniform_noise)
            sa,sb,sc,sd = batch[0].size()
            noise  = torch.rand((sa,sb,sc,sd)).to("cuda")
            batch = (batch[0]+ noise * self.uniform_noise,batch[1])
        if self.gaussian_noise > 0 :
            print(self.gaussian_noise)
            sa,sb,sc,sd = batch[0].size()
            noise  = torch.randn((sa,sb,sc,sd)).to("cuda")
            batch = (batch[0]+ noise * self.gaussian_noise,batch[1])

        out = self.shared_step(batch, batch_idx)
        #print(out)
        results = {
            "batch_size": out["batch_size"],
            "val_loss": out["loss"],
            "val_acc1": out["acc1"],
            "val_acc5": out["acc5"],
            "feats":out["feats"],
            "targets":out["targets"],
            "logits": out["logits"],
            "inputs": out["inputs"],

        }
        return results


    def training_epoch_end(self,outs):
        feats = outs[0]["feats"]

        for i in range(len(outs)):
            feats = torch.cat((feats,outs[i]["feats"]),0)



        feats = feats.cuda()
        feats = torch.nn.functional.relu(feats)
        feats = F.normalize(feats)

        feats_sum = torch.sum(feats,dim=0)
        _,index = torch.sort(feats_sum,descending=True)

        #torch.save(index,'importance.pt')

        if self.i_index == None:
            #self.i_index = index
            self.i_index = [i for i in range(2048)]

    

    def validation_epoch_end(self, outs: List[Dict[str, Any]]):
        """Averages the losses and accuracies of all the validation batches.
        This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.

        Args:
            outs (List[Dict[str, Any]]): list of outputs of the validation step.
       

        """
        feats = outs[0]["feats"]
        targets = outs[0]["targets"]




        for i in range(1,len(outs)):
            feats = torch.cat((feats,outs[i]["feats"]),0)
            targets = torch.cat((targets,outs[i]["targets"]),0)

        feats = feats.cuda()
        feats = F.normalize(feats)

        

        val_loss = weighted_mean(outs, "val_loss", "batch_size")
        val_acc1 = weighted_mean(outs, "val_acc1", "batch_size")
        val_acc5 = weighted_mean(outs, "val_acc5", "batch_size")

        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5}
        self.log_dict(log, sync_dist=True)

    def test_epoch_end(self, outs: List[Dict[str, Any]]):

        feats = outs[0]["feats"]
        logits = outs[0]["logits"]
        targets = outs[0]["targets"]
        inputs = outs[0]["inputs"]




        for i in range(1,len(outs)):
            feats = torch.cat((feats,outs[i]["feats"]),0)
            logits = torch.cat((logits,outs[i]["logits"]),0)
            targets = torch.cat((targets,outs[i]["targets"]),0)
            inputs = torch.cat((inputs,outs[i]["inputs"]),0)





        val_loss = weighted_mean(outs, "val_loss", "batch_size")
        val_acc1 = weighted_mean(outs, "val_acc1", "batch_size")
        val_acc5 = weighted_mean(outs, "val_acc5", "batch_size")
        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5}
        self.log_dict(log, sync_dist=True)


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
            print(mean_acc)
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
