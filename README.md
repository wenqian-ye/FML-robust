# Improved Adversarial Robustness via Data Augmentation

**Group Members: Wenqian Ye, Yunsheng Ma, Xu Cao**

This is the repository of the final project of the Spring 2022 graduate course, Foundation of Machine Learning at Courant Institute, New York University, taught by Mehryar Mohri. 

## Requirements

The code has been implemented on the single RTX 8000 and tested with `Python 3.8.5` and `PyTorch 1.8.0`.  To install the required packages:
```
$ pip install -r requirements.txt
```

## Usage

### Training

DDPM training setting

```
pip install denoising_diffusion_pytorch
```

Train DDPM

```
python train.py
```

Generate CIFAR-10 data

```
python generate.py
```

Adversarial Robustness Training. Run the following command to reproduce our model:

```
$ python train-wa.py --data-dir <data_dir> \
    --log-dir <log_dir> \
    --desc <name_of_the_experiment> \
    --data cifar10s \
    --batch-size 256 \
    --model wrn-28-10-swish \
    --num-adv-epochs 100 \
    --lr 0.04 \
    --beta 6.0 \
    --unsup-fraction 0.7 \
    --aux-data-filename <path_to_additional_data>
```

### Adversarial Robustness Evaluation

The trained models can be evaluated by running [`eval-aa.py`](./eval-aa.py) which uses [AutoAttack](https://github.com/fra31/auto-attack) for evaluating the robust accuracy. 

```
$ python eval-aa.py --data-dir <data_dir> \
    --log-dir <log_dir> \
    --desc <name_of_the_experiment>
```

### Logs and trained model:

DDPM generated data: https://drive.google.com/file/d/1GoHKieyHLoH1OXjWWSFIEZ2xUpuxGJwj/view

Training logs and model: https://drive.google.com/file/d/1FVmkLpxdzTlxaMmzNsEIrJZ2sRR1xQtM/view?usp=sharing

## References

```
@article{rebuffi2021fixing,
  title={Fixing Data Augmentation to Improve Adversarial Robustness},
  author={Rebuffi, Sylvestre-Alvise and Gowal, Sven and Calian, Dan A. and Stimberg, Florian and Wiles, Olivia and Mann, Timothy},
  journal={arXiv preprint arXiv:2103.01946},
  year={2021},
  url={https://arxiv.org/pdf/2103.01946}
}
@misc{rade2021pytorch,
    title = {{PyTorch} Implementation of Uncovering the Limits of Adversarial Training against Norm-Bounded Adversarial Examples},
    author = {Rade, Rahul},
    year = {2021},
    url = {https://github.com/imrahulr/adversarial_robustness_pytorch}
}
@misc{ho2020denoising,
    title   = {Denoising Diffusion Probabilistic Models},
    author  = {Jonathan Ho and Ajay Jain and Pieter Abbeel},
    year    = {2020},
    eprint  = {2006.11239},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```

DDPM Code reference from https://github.com/lucidrains/denoising-diffusion-pytorch.
