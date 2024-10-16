# Beyond Interpretability: The Gains of Feature Monosemanticity on Model Robustness

Official PyTorch implementation of the paper Beyond Interpretability: The Gains of Feature Monosemanticity on Model Robustness by Qi Zhang, [Yifei Wang*](https://yifeiwang77.com/), Jingyi Cui, Xiang Pan, Qi Lei, Stefanie Jegelka, [Yisen Wang](https://yisenwang.github.io/)



##TLDR

This work challenges the common “accuracy-interpretability” tradeoff by demonstrating the potential of feature monosemanticity to bring clear gains in model accuracy. These gains manifest
themselves in various aspects of “learning robustness” that we can think of: input noise, label noise, out-of-domain data, few-shot image data, and few-shot language data. The diverse set of evidence
strongly indicates that feature monosemanticity provides a general sense of robustness compared to polysemantic features,


![image](visualization.png) 






## Installation

The codebase is built upon a previous version of [```solo-learn```](https://github.com/vturrisi/solo-learn) (the version on Sep 27, 2022). To avoid unexpected errors, first create a ``Python3.8`` environment, and then install the reposoity as below.
```
# clone the repository
git clone https://github.com/PKU-ML/non_neg
# create environment
conda create -n non_neg python=3.8
conda activate non_neg
# install dependences
cd non_neg
pip3 install .[dali,umap,h5] --extra-index-url https://developer.download.nvidia.com/compute/redist --extra-index-url https://download.pytorch.org/whl/cu113
```

## Obtain Polysemantic/Monosemantic Representations

Pretrain with the default configuration files using the following command.

### CIFAR-100 
```bash
# SimCLR (Poly)
python3 main_pretrain.py \
    --config-path scripts/pretrain/cifar \
    --config-name simclr.yaml
# NCL (Mono)
python3 main_pretrain.py \
    --config-path scripts/pretrain/cifar \
    --config-name ncl.yaml
```

# SAE (Mono)
python3 main_sparse.py \
    --config-path scripts/pretrain/cifar \
    --config-name sae.yaml
```


### ImageNet-100
```bash
# SimCLR (Poly)
python3 main_pretrain.py \
    --config-path scripts/pretrain/imagenet-100 \
    --config-name simclr.yaml
# NCL (Mono)
python3 main_pretrain.py \
    --config-path scripts/pretrain/imagenet-100 \
    --config-name ncl.yaml
```

# SAE (Mono)
python3 main_sparse.py \
    --config-path scripts/pretrain/imagenet-100 \
    --config-name sae.yaml
```




## Linear Evaluation


After that, for linear evaluation, run the following command:

```bash
python3 main_linear.py \
    --config-path scripts/linear/{dataset} \
    --config-name simclr.yaml \
    pretrained_feature_extractor=path/to/pretrained/feature/extractor
```
Here ``dataset={cifar,imagenet100}``. We use the argument ``pretrained_feature_extractor`` to configure the path of the pretrained checkpoints.


## Full finetuning

And for fine-tuning evaluation, run the following command:


```bash
python3 main_linear.py \
    --config-path scripts/finetuning/{dataset} \
    --config-name simclr.yaml
```

## Feature Selection

And for offline linear probing with selected dimensions, run the following command:

```bash
python3 main_linear.py \
    --config-path scripts/selected \
    --config-name simclr.yaml \
    selected_dims=256
```
where the argument ``selected_dims`` configures the dimensions of selected features.



## Pretrained Checkpoints

The following table provides the pre-trained checkpoints for CL and NCL.

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">CIFAR-10</th>
<th valign="bottom">CIFAR-100</th>
<th valign="bottom">ImageNet-100</th>
<!-- TABLE BODY -->
<tr><td align="left">checkpoints</td>
<td align="center"><a href="https://drive.google.com/drive/folders/1z57D9WOZk5N5nsqVixkUza9ZX6NiH6Wx?usp=sharing">download</a></td>
<td align="center"><a href="https://drive.google.com/drive/folders/1EbF9oKFu9rjsfRj_hv-Q-GVYKUSVxIkP?usp=sharing">download</a></td>
<td align="center"><a href="https://drive.google.com/drive/folders/1iIqn2hklptrlG3bLmjULw_rfKKO-JC5s?usp=sharing">download</a></td>
</tr>
</tbody></table>



## Citing this work
If you find the work useful, please cite the accompanying paper:
```
@inproceedings{
wang2024nonnegative,
title={Non-negative Contrastive Learning},
author={Yifei Wang and Qi Zhang and Yaoyu Guo and Yisen Wang},
booktitle={ICLR},
year={2024},
}
```

## Acknowledgement

Our codes borrow the implementations of SimCLR in the solo-learn repository: https://github.com/vturrisi/solo-learn
