# Training script for ControlNet

from omegaconf import OmegaConf
import pytorch_lightning as pl
from cldm_seg.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import os
import re
from datetime import datetime
import argparse
from pytorch_lightning import seed_everything
from distutils.util import strtobool
import yaml
import pandas as pd
import copy

# Configs
def parse_args():
    parser = argparse.ArgumentParser(description='ControlNet training')
    parser.add_argument(
        '--config',
        default='./models/cldm_seg.yaml',
        help='config dir'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='cityscapes', # ade
        help='train on which dataset'
    )
    parser.add_argument(
        '--work_dir', '--work-dir',
        default='./train_log',
        help='the dir to save logs and models'
    )
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use ',
    )
    parser.add_argument('--batch_size', type=int, default=2, help='batch size per GPU')
    parser.add_argument('--val_batch_size', type=int, default=4, help='batch size per GPU')
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate for ControlNet') # 1e-5
    parser.add_argument('--weight_decay', type=float, default=0.005,
                        help='weight decay') # for Adam, default=0.0, AdamW default=0.01
    parser.add_argument(
        '--logger_freq',
        type=int,
        default=300,
        help='logging frequency',
    )
    parser.add_argument(
        '--logger_freq_epoch',
        type=int,
        default=20,
        help='logging frequency',
    )
    parser.add_argument(
        '--resume_path',
        type=str,
        default='./checkpoint/control_sd15_ini.ckpt',
        help='resume from directory'
    )
    parser.add_argument('--seed', type=int, default=777, help='random seed')
    parser.add_argument('--max_it', type=int, default=300, help='max iterations in K')
    parser.add_argument('--sd_locked', type=lambda x: bool(strtobool(x)), default='true', help='sd locked')
    parser.add_argument('--only_mid_control', type=lambda x: bool(strtobool(x)), default='false', help='only mid control')

    # Hyperparameters #
    parser.add_argument('--d_lr', type=float, default=0.00001,
                        help='learning rate for the discriminator')  # 1e-5
    parser.add_argument('--lambda_D_loss', type=float, default=0.1,  # TODO: need to try!
                        help='weight for discriminator loss')
    parser.add_argument('--start_dis_fake', type=int, default=5000, help='warm start starting iteration')  # 5000
    parser.add_argument('--end_dis_fake', type=int, default=10000, help='warm start ending iteration')  # 10_000
    parser.add_argument('--weight_fake_D', type=float, default=0.01, help='weight of updating D based on fake images')

    # Multi-step sampling related #
    parser.add_argument('--multi_step', type=int, default=9, help='number of forwarded steps during training')
    # Ablation: Random number of sampling steps
    parser.add_argument('--random_multi_step', type=lambda x: bool(strtobool(x)), default='false',
                        help='if randomly sampling number of multiple steps')
    parser.add_argument('--min_sample_step', type=int, default=3, help='max. random sampling step')
    parser.add_argument('--max_sample_step', type=int, default=15, help='max. random sampling step')

    # Stable parameters #
    parser.add_argument('--use_diffusion_loss', type=lambda x: bool(strtobool(x)), default='true',
                        help='use diffusion training loss when updating the generator')
    parser.add_argument('--lambda_diffusion_loss', type=float, default=1.0,
                        help='weight for original diffusion loss')
    parser.add_argument('--num_subset_timesteps', type=int, default=25, help='number of subset timesteps')
    parser.add_argument('--use_caption_training', type=lambda x: bool(strtobool(x)), default='true',
                        help='if using caption as text condition during training')
    parser.add_argument('--drop_caption_ratio', type=float, default=-1.0, help='drop caption ratio')
    parser.add_argument('--warm_up_dis', type=lambda x: bool(strtobool(x)), default='false',
                        help='if warm up segmenter or just the generator')
    parser.add_argument('--D_lr_scheduler', type=str, default='one_cycle',
                        help='learning rate scheduler for the discriminator, None-constant lr')
    parser.add_argument('--D_lr_warmup_it', type=int, default=200,
                        help='warm up steps of learning rate scheduler for the discriminator')
    parser.add_argument('--D_sampler_mode', type=str, default='V1',
                        help='loss balancing sampler')
    parser.add_argument('--train_dis_from', type=int, default=-1, help='starting training Discriminator')  # 5000
    parser.add_argument('--n_lazy_guidance', type=int, default=8, help='applying discriminator guidance every N steps')
    parser.add_argument('--train_D_before_lazy', type=lambda x: bool(strtobool(x)), default='false',
                        help='train D at every step before lazy guidance')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.dataset == 'cityscapes':
        num_classes = 19
        fake_class_id = 19
        dataset_short_name = 'CS'
    elif args.dataset == 'ade':
        num_classes = 150
        fake_class_id = 150
        dataset_short_name = 'ade'
    elif args.dataset == 'coco':
        num_classes = 171
        fake_class_id = 171
        dataset_short_name = 'COCO'
    else:
        raise ValueError('Given dataset is not supported yet!')

    segmenter_config = OmegaConf.create({
        'num_classes': num_classes,
        'loss_sampler_version': args.D_sampler_mode,
    })

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(args.config, extra_segmenter_config=segmenter_config).cpu()
    model.load_state_dict(load_state_dict(args.resume_path, location='cpu'),strict=False)
    model.segmenter.load_pretrained_segmenter()

    print('---> sd_locked:', args.sd_locked, 'only_mid_control:', args.only_mid_control)
    model.sd_locked = args.sd_locked
    model.only_mid_control = args.only_mid_control

    model.fake_class_id = fake_class_id # for all fake images!
    model.use_diffusion_loss = args.use_diffusion_loss
    model.lambda_diffusion_loss = args.lambda_diffusion_loss
    model.lambda_D_loss = args.lambda_D_loss
    model.start_dis_fake = args.start_dis_fake
    model.end_dis_fake = args.end_dis_fake
    model.num_subset_timesteps = args.num_subset_timesteps
    model.warm_up_dis = args.warm_up_dis
    model.use_caption_training = args.use_caption_training
    model.drop_caption_ratio = args.drop_caption_ratio
    model.weight_fake_D = args.weight_fake_D
    model.multi_step = args.multi_step
    model.train_dis_from = args.train_dis_from
    model.n_lazy_guidance = args.n_lazy_guidance
    if args.n_lazy_guidance > 1:
        use_lazy_guidance = True
    else:
        use_lazy_guidance = False
    model.use_lazy_guidance = use_lazy_guidance
    model.train_D_before_lazy = args.train_D_before_lazy

    # For random sampling step in ablation study
    model.random_multi_step = args.random_multi_step
    model.min_sample_step = args.min_sample_step
    model.max_sample_step = args.max_sample_step

    model.set_ddim_sampler(args.num_subset_timesteps)
    optimizer_config= {
        'type': 'AdamW', # 'Adam'
        'G_lr': args.lr,
        'D_lr': args.d_lr,
        'weight_decay': args.weight_decay,
        'D_lr_scheduler': args.D_lr_scheduler,
        'D_lr_all_it': int(args.max_it * 1000),
        'pct_start': args.D_lr_warmup_it / (args.max_it * 1000),
    }
    model.optimizer_config = optimizer_config
    model.batch_size_allGPU = args.batch_size * args.gpus
    model.dataset = args.dataset
    model.batch_size = args.batch_size
    model.val_batch_size = args.val_batch_size


    # Misc logging
    logger = ImageLogger(
        batch_frequency=args.logger_freq,
        max_images=args.val_batch_size,
    )

    # Choose logdir
    outdir = args.work_dir
    os.makedirs(outdir, exist_ok=True)
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    seed = args.seed
    desc = f'seed{seed}-cldm_seg_{dataset_short_name}-lambdaD{args.lambda_D_loss}-DFake{args.weight_fake_D}'
    logdir = f'{cur_run_id:04d}-{dt_string}-{desc}'
    logdir = os.path.join(outdir,logdir)

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=logdir,name=None,version='',
        default_hp_metric=False,
    )

    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(logdir, 'checkpoint'),
        save_top_k=3,
        every_n_train_steps=args.logger_freq,
        save_on_train_epoch_end=True,
        save_last=True,
        monitor="loss",
    )
    checkpoint.CHECKPOINT_NAME_LAST = "{epoch}-{step}-last"

    checkpoint_epochEnd = ModelCheckpoint(
        dirpath=os.path.join(logdir, 'checkpoint'),
        save_top_k=-1,
        every_n_epochs=args.logger_freq_epoch,
        save_on_train_epoch_end=True,
        save_last=False,
        filename='end-{epoch}-{step}',
        monitor=None,
    )
    checkpoint_epochEnd.CHECKPOINT_NAME_LAST = "{epoch}-{step}-last"

    seed_everything(seed)

    plugins = []
    if args.gpus == 1:
        accelerator = 'gpu'
    else:
        print('----> Num GPUs = ', args.gpus)
        accelerator = 'ddp'
        from pytorch_lightning.plugins import DDPPlugin
        # important for working with gradient checkpoint
        plugins.append(DDPPlugin(find_unused_parameters=False))

    model.D_lr_scheduler = args.D_lr_scheduler

    callbacks = [logger, checkpoint, checkpoint_epochEnd]

    trainer = pl.Trainer(
        precision=32,
        gpus=args.gpus,
        #devices=args.gpus,
        accelerator=accelerator,
        logger=tb_logger,
        callbacks=callbacks,
        plugins=plugins,
        max_steps=int(args.max_it * 1000),
        # accumulate_grad_batches=args.accum_batches,
        # fast_dev_run=True, # for debugging
    )

    # log hyperparameters
    converted_dict = copy.deepcopy(vars(args))  # otherwise will overwrite args
    converted_dict['work_dir'] = logdir
    print('global_rank = ', trainer.global_rank)
    if trainer.global_rank == 0:
        os.makedirs(logdir, exist_ok=True)
        with open(os.path.join(logdir, 'config.yaml'), 'w') as f:
            yaml.dump(converted_dict, f)
        csv_name = os.path.join(args.work_dir, 'experiments_cldm_segPixel.csv')
        df = pd.DataFrame.from_dict([converted_dict], orient='columns')
        hdr = False if os.path.isfile(csv_name) else True
        try:
            if os.path.isfile(csv_name):
                df_old = pd.read_csv(csv_name)
                df = pd.concat([df_old, df])
                df.to_csv(csv_name, mode='w', header=True, index=False)
            else:
                df.to_csv(csv_name, mode='a', header=True, index=False)
        except Exception as e:
            print(e)

    # Train!
    trainer.fit(model)


if __name__ == '__main__':
    main()
