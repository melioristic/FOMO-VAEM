import torch
from vae.vae_pl import ModalVAE
from fomo.dataset import MultiModalDatasets
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np

import h5py

def get_predictions(model, ds):
    trainer = pl.Trainer()
    out = trainer.predict(model, DataLoader(ds, batch_size=64))
    r_list = []
    x_list = []
    mu_list = []
    log_var_list = []
    for each in out:
        r, x, mu, log_var = each
        r_list.append(r)
        x_list.append(x)
        mu_list.append(mu)
        log_var_list.append(log_var)  

    r = np.vstack(r_list)
    x = np.vstack(x_list)
    mu = np.vstack(mu_list)
    log_var = np.vstack(log_var_list)

    return r, x, mu, log_var

    

train_data = MultiModalDatasets('/data/compoundx/anand/benchmark-dataset/', split_type = "train", xs_list = ["age","laicum"])
val_data = MultiModalDatasets('/data/compoundx/anand/benchmark-dataset/', split_type = "validation", xs_list = ["age","laicum"])
test_data = MultiModalDatasets('/data/compoundx/anand/benchmark-dataset/', split_type = "validation", xs_list = ["age","laicum"])

weather_model = ModalVAE.load_from_checkpoint("/data/compoundx/anand/fomo-vaem/grouped_weather/lightning_logs/version_5/checkpoints/epoch=192-step=96500.ckpt", map_location=torch.device("cpu"), inp_shape=(1,36,3), modality="grouped_weather", beta=0.0001, batch_size=16)
state_model = ModalVAE.load_from_checkpoint("/data/compoundx/anand/fomo-vaem/grouped_states/lightning_logs/version_7/checkpoints/epoch=401-step=201000.ckpt", map_location=torch.device("cpu"), inp_shape=(1,104,2), modality="grouped_states", beta=0.0001, batch_size=16)


def prep_discriptor_data(data, name):

    biomass_loss = []
    dl = DataLoader(data, batch_size=64)
    for each in dl:
        biomass_loss.extend(each[-1])
    
    bl = torch.stack(biomass_loss)

    r_w, x_w, mu_w, log_var_w = get_predictions(weather_model, data)
    r_s, x_s, mu_s, log_var_s = get_predictions(state_model, data)

    assert r_w.shape[0] == r_s.shape[0] ==bl.shape[0]

    with h5py.File(f"data/{name}.h5", "w") as f:
        f.create_dataset("r_w", data = r_w)
        f.create_dataset("r_s", data = r_s)
        f.create_dataset("x_w", data = x_w)
        f.create_dataset("x_s", data = x_s)
        f.create_dataset("mu_w", data = mu_w)
        f.create_dataset("mu_s", data = mu_s)
        f.create_dataset("log_var_w", data = log_var_w)
        f.create_dataset("log_var_s", data = log_var_s)
        f.create_dataset("bl", data = bl)

prep_discriptor_data(test_data, "test_data")