import argparse

import torch

from fomo.dataset import MultiModalDatasets
from models.ccvae.models.ccvae_pl import CCVAE_L
from lightning.pytorch import Trainer
from lightning.pytorch import loggers as pl_loggers

import lightning as L


def main(args):
    """
    run inference for SS-VAE
    :param args: arguments for SS-VAE
    :return: None
    """

    train_data =  MultiModalDatasets('/data/compoundx/anand/benchmark-dataset/', split_type = "train", xs_list = ["age","laicum"], target="ALL", classify=True)
    val_data =  MultiModalDatasets('/data/compoundx/anand/benchmark-dataset/', split_type = "validation", xs_list = ["age","laicum"], target="ALL", classify=True)

    in_shape = (1, sum(train_data.get_seq_len()))
    num_classes = train_data.get_lbl_info()[1]
    
    cc_vae = CCVAE_L(z_dim=args.z_dim,
                   num_classes=num_classes,
                   in_shape=in_shape,
                   lr = args.learning_rate,
                   prior_fn=lambda:torch.ones(1, num_classes) / 2,
                   train_dataset=train_data,
                   valid_dataset=val_data)
    
    devices = 1
    if args.accelerator == "cuda":
        devices = -1
    
    model_dir = f"/data/compoundx/anand/fomo-vaem/ccvae"
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=model_dir)
    trainer = Trainer(
        max_steps = 100000,
        logger=tb_logger,
        accelerator=args.accelerator,
        devices=devices,
    )

    trainer.fit(cc_vae)



def parser_args(parser):
    parser.add_argument('-a', '--accelerator', default='cpu',
                        help="use GPU(s) to speed up training")
    parser.add_argument('-n', '--num-epochs', default=200, type=int,
                        help="number of epochs to run")
    parser.add_argument('-zd', '--z_dim', default=64, type=int,
                        help="size of the tensor representing the latent variable z "
                             "variable (handwriting style for our MNIST dataset)")
    parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float,
                        help="learning rate for Adam optimizer")
    parser.add_argument('-bs', '--batch-size', default=200, type=int,
                        help="number of images (and labels) to be considered in a batch")

    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parser_args(parser)
    args = parser.parse_args()

    main(args)

