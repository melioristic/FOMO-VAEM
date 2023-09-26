import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import (
    EarlyStopping,
)

from fomo.dataset import MultiModalDatasets
from torch.utils.data import DataLoader

from vae.vae_pl import ModalVAE


parser = argparse.ArgumentParser(description='ModalVAE Analysis')

parser.add_argument('--data_path', type=str, default='/data/compoundx/anand/benchmark-dataset/',
                    help='path for storing the dataset')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 1024)')
parser.add_argument('--modality', "-m", type=str, default="rad", 
                    help='modality (default: rad)')
parser.add_argument('--accelerator', "-a", type=str, default="cuda", 
                    help='modality (default: rad)')

args = parser.parse_args()

train_data = MultiModalDatasets(args.data_path, split_type = "train", xs_list = ["age", "laicum"])
val_data = MultiModalDatasets(args.data_path, split_type = "validation", xs_list = ["age","laicum"])

if args.modality == "rad":
    inp_shape = (1,36,1)
elif args.modality == "precip":
    inp_shape = (1,36,1)
elif args.modality == "temp":
    inp_shape = (1,36,1)
elif args.modality == "age":
    inp_shape = (1,100,1)
elif args.modality == "lai":
    inp_shape = (1,100,1)

if args.accelerator == "cuda":
    devices = -1
elif args.accelerator == "cpu":
    devices = 1


model = ModalVAE(
    inp_shape = inp_shape,
    modality= args.modality,
    train_dataset=train_data,
    valid_dataset= val_data,
    batch_size=args.batch_size
)
    
model_dir = f"/data/compoundx/anand/fomo-vaem/{args.modality}"

tb_logger = pl_loggers.TensorBoardLogger(save_dir=model_dir)

lr_monitor = LearningRateMonitor(logging_interval="step")

trainer = pl.Trainer(
        max_steps=100000,
        accelerator=args.accelerator,
        devices=devices,
        callbacks=[lr_monitor],
        logger=tb_logger,
    )

trainer.fit(model)
