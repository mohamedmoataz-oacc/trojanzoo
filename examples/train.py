#!/usr/bin/env python3

r"""
CUDA_VISIBLE_DEVICES=0 python examples/train.py --verbose 1 --color --epochs 200 --batch_size 96 --cutout --lr 0.025 --lr_scheduler

# adv train pgd
CUDA_VISIBLE_DEVICES=0 python examples/train.py --verbose 1 --color --adv_train --adv_train_random_init --validate_interval 1 --epochs 15 --lr 0.1 --lr_scheduler

# adv train fgsm
CUDA_VISIBLE_DEVICES=0 python examples/train.py --verbose 1 --color --adv_train --adv_train_random_init --adv_train_iter 1 --adv_train_alpha 0.0392156862745 --adv_train_eval_iter 7 --adv_train_eval_alpha 0.0078431372549 --validate_interval 1 --epochs 15 --lr 0.1 --lr_scheduler

# adv train fgsm mnist
CUDA_VISIBLE_DEVICES=0 python examples/train.py --verbose 1 --color --dataset mnist --adv_train --adv_train_random_init --adv_train_iter 1 --adv_train_alpha 0.375 --adv_train_eps 0.3 --adv_train_eval_iter 7 --adv_train_eval_alpha 0.1 --adv_train_eval_eps 0.3 --validate_interval 1 --epochs 15 --lr 0.1 --lr_scheduler
"""  # noqa: E501

import trojanvision

def train(dataset_name: str, model_name: str, epochs: int):
    kwargs = {
        'cmd_config_path': None, 'seed': None, 'data_seed': None, 'cache_threshold': None,
        'device': None, 'cudnn_benchmark': False, 'verbose': 1, 'color': True, 'tqdm': False,
        'dataset_name': dataset_name, 'batch_size': None, 'valid_batch_size': None, 'num_workers': None,
        'download': True, 'data_dir': None, 'normalize': False, 'transform': None, 'auto_augment': False,
        'mixup': False, 'mixup_alpha': None, 'cutmix': False, 'cutmix_alpha': None, 'cutout': True,
        'cutout_length': None, 'data_format': None, 'memory': False, 'model_name': model_name, 'suffix': None,
        'pretrained': False, 'official': False, 'randomized_smooth': None, 'rs_sigma': None, 'rs_n': None,
        'model_dir': None, 'adv_train': None, 'adv_train_random_init': False, 'adv_train_iter': None,
        'adv_train_alpha': None, 'adv_train_eps': None, 'adv_train_eval_iter': None,
        'adv_train_eval_alpha': None, 'adv_train_eval_eps': None, 'adv_train_trades_beta': None,
        'norm_layer': 'bn', 'sgm': False, 'sgm_gamma': None, 'epochs': epochs, 'resume': None, 'lr': None,
        'parameters': None, 'OptimType': None, 'momentum': None, 'weight_decay': None, 'nesterov': False,
        'lr_scheduler': True, 'lr_scheduler_type': None, 'lr_min': None, 'lr_warmup_epochs': None,
        'lr_warmup_method': None, 'lr_step_size': None, 'lr_gamma': None, 'model_ema': False,
        'model_ema_steps': None, 'model_ema_decay': None, 'pre_conditioner': None, 'amp': False,
        'grad_clip': 5.0, 'validate_interval': None, 'save': True, 'tensorboard': False, 'log_dir': None,
        'flush_secs': None
    }

    print("Still didn't download")
    env = trojanvision.environ.create(**kwargs)
    dataset = trojanvision.datasets.create(**kwargs)
    print("Downloaded")
    model = trojanvision.models.create(dataset=dataset, **kwargs)
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **kwargs)

    if env['verbose']:
        trojanvision.summary(env=env, dataset=dataset, model=model, trainer=trainer)
    model._train(**trainer)
