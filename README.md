# Welcome to MulDEns code base.

MulDEns is written on top of DomainBeD, a PyTorch suite containing benchmark datasets and algorithms for domain generalization, as introduced in (https://arxiv.org/abs/2007.01434).



## Quick start

Download the datasets:

```sh
python -m domainbed.scripts.download \
       --data_dir='DATA'
```

Train a MULDENS ensemble with M=2 :

```sh

python3 -m domainbed.scripts.train_aug --data_dir='DATA'\
--dataset OfficeHome --test_env 0 --trial_seed 0\
--output_dir='muldens_ouputs/OfficeHome_M2/env0/trial_seed0/'\
--hparams='{"batch_size":32,"data_augmentation":1,"MULDENS_num_models":2}' 
    
```

## Details
* In domainbed/algorithms.py MULDENS is implemented
* Through domainbed/scripts/train_aug.py and domainbed/lib/misc_aug.py we train MULDENS
* Once checkpoints are saved, if we want to load and re-evaluate, we use domainbed/scripts/eval_muldens_aug.py

## Model selection criteria

We use two different model selection criteria Overall Average and Overall Ensemble. More details in paper.


## Baseline Results.
We report baseline results In search of lost domain generalization paper (https://openreview.net/pdf?id=lQdXeXDoWtI) and their github repo.
Full results for [commit 7df6f06](https://github.com/facebookresearch/DomainBed/tree/7df6f06a6f9062284812a3f174c306218932c5e4) in LaTeX format available [here](domainbed/results/2020_10_06_7df6f06/results.tex).


