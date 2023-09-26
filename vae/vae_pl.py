import numpy as np

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torchvision
from torch.nn import functional as F


from .vae import ConvVAE

class ModalVAE(pl.LightningModule):
    def __init__(self,
        inp_shape,
        modality,
        train_dataset = None,
        valid_dataset = None,
        n_step = 2, 
        dim = None,
        batch_size = 32,
        lr = 1e-3,
        lr_scheduler_name="ReduceLROnPlateau",
        num_workers = 1,
        beta = 1e-3
        ):
        super().__init__()

        self.modal_dict = {
            "rad":(0,),
            "precip":(1,),
            "temp":(2,),
            "age":(3,),
            "lai":(4,),
            "grouped_weather": (0,1,2),
            "grouped_states":(3,4),
        }

        self.beta = beta
        self.num_workers = num_workers
        self.lr_scheduler_name = lr_scheduler_name
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        
        if train_dataset!=None:
            len_train = len(self.train_dataset)
            len_val = len(self.valid_dataset)

            self.train_kld_weight = batch_size/len_train
            self.val_kld_weight = batch_size/len_val



        else:
            len_train = None
            len_val = None
            self.train_kld_weight = None
            self.val_kld_weight = None

        self.lr = lr
        self.batch_size = batch_size
        self.modality = modality
        self.model = ConvVAE(
            inp_shape=inp_shape,
            dim=dim,
            n_step = n_step
        )

        

    @torch.no_grad()
    def forward(self, batch, *args):
        
        x = torch.permute(x, (0, 2, 1 ,3))
        if x.shape[-1]==101:
            x = F.pad(x, (1,0), "constant", 0)
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        
        x = self.get_input(batch[0])

        x = torch.permute(x, (0, 2, 1 ,3))

        if x.shape[-1]==101:
            x = F.pad(x, (2,2), "constant", 0)
        r, x, mu, log_var = self.model(x)
        
        loss = self.model.loss_function(r, x, mu, log_var, self.beta, self.train_kld_weight)

        self.log("train_loss", loss["loss"], prog_bar=True, on_epoch=True)
        self.log("train_recon_loss", loss["recon_loss"], prog_bar=True, on_epoch=True)
        self.log("train_KLD_loss", loss["KLD_loss"] , prog_bar=True, on_epoch=True)

        if self.trainer.global_step == 0:
            self.log("val_loss", np.inf)

        return loss["loss"] 

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch[0])
        x = torch.permute(x, (0, 2, 1 ,3))

        if x.shape[-1]==101:
            x = F.pad(x, (2,2), "constant", 0)
        
        r, x, mu, log_var = self.model(x)

        loss = self.model.loss_function(r, x, mu, log_var, self.beta, self.val_kld_weight)

        self.log("val_loss", loss["loss"], prog_bar=True, on_epoch=True)
        self.log("val_recon_loss", loss["recon_loss"], prog_bar=True, on_epoch=True)
        self.log("val_KLD_loss", loss["KLD_loss"] , prog_bar=True, on_epoch=True)

        if batch_idx == 0:
            n_images = 5
            img_stack = torch.concat([r[:n_images, :, :], x[:n_images, :, :]], dim=0)
                                      
            grid = torchvision.utils.make_grid(img_stack[:,:,:, None], nrow=n_images, padding=10) # plot the first n_images images.
            self.logger.experiment.add_image('generated_images', grid, self.current_epoch)

    def configure_optimizers(self):
        # Cosine Annealing LR Scheduler

        optimizer = torch.optim.AdamW(
            list(
                filter(
                    lambda p: p.requires_grad,
                    self.model.parameters(),
                )
            ),
            lr=self.lr,
        )
        scheduler = self._get_scheduler(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
            "interval":"epoch"
        }

    def _get_scheduler(self, optimizer):
        # for experimental purposes only. 
        # All epoch related things are in respect to the "1x longer" epoch length.
        return getattr(torch.optim.lr_scheduler, self.lr_scheduler_name)(optimizer=optimizer)
        
    def train_dataloader(self):
        if self.train_dataset is not None:
            return DataLoader(
                self.train_dataset,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                # shuffle=False,
            )
        else:
            return None
    
    def val_dataloader(self):
        if self.valid_dataset is not None:
            return DataLoader(
                self.valid_dataset,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                # shuffle=False,
            )
        else:
            return None
        
    def get_input(self, x):
        
        assert self.modality in self.modal_dict

        tuple_index = self.modal_dict[self.modality]
        return torch.stack([x[i] for i in tuple_index], axis=2)