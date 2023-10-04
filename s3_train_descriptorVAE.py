import h5py
import numpy as np

import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import (
    EarlyStopping,
)

from vae.vae_pl import DescriptorVAE

from fomo.dataset import DescriptorDatasets
 
train_data = DescriptorDatasets("data/train.h5")
val_data = DescriptorDatasets("data/val.h5")

model = DescriptorVAE(train_dataset=train_data, valid_dataset=val_data, latent_dim=8, batch_size=64)

model_dir = f"/data/compoundx/anand/fomo-vaem/descriptor/"

tb_logger = pl_loggers.TensorBoardLogger(save_dir=model_dir)

lr_monitor = LearningRateMonitor(logging_interval="step")

trainer = pl.Trainer(
        max_steps=1000000,
        accelerator="gpu",
        devices=-1,
        callbacks=[lr_monitor],
        logger=tb_logger,
    )

trainer.fit(model)