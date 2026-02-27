from dotenv import load_dotenv
load_dotenv()
import os
from pathlib import Path
import sys
import torch
from torch import Tensor
import argparse
import json
from typing import Optional, Sequence
import look2hear.datas
import look2hear.models
import look2hear.system
import look2hear.losses
import look2hear.metrics
import look2hear.utils
from look2hear.system import make_optimizer
from dataclasses import dataclass
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import *
from rich.console import Console
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from rich import print, reconfigure
from collections.abc import MutableMapping
from look2hear.utils import print_only, MyRichProgressBar, RichProgressBarTheme

import warnings

from pathlib import Path
from own_modules.callbacks_lightning import ExternalStopCallback
from torch.utils.data import DataLoader, Subset

warnings.filterwarnings("ignore")

import wandb
if os.environ.get("WANDB_API_KEY"):
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
else:
    raise "WANDB_API_KEY not found"
parser = argparse.ArgumentParser()
parser.add_argument(
    "--conf_dir",
    default="local/conf.yml",
    help="Full path to save best validation model",
)


def _build_loss_from_config(loss_cfg):
    """Supports TSE configs where loss_func can be null/None."""
    sdr_obj = getattr(look2hear.losses, loss_cfg["sdr_type"])
    loss_name = loss_cfg.get("loss_func", None)
    if loss_name is None or str(loss_name).lower() in {"none", "null", ""}:
        return sdr_obj
    return getattr(look2hear.losses, loss_name)(
        sdr_obj,
        **loss_cfg.get("config", {}),
    )


def _subset_from_config(ds, indices: Optional[Sequence[int]], n_samples: Optional[int]):
    if ds is None:
        return None

    if indices is not None:
        ds_len = len(ds)
        idx = []
        for raw_i in indices:
            i = int(raw_i)
            if i < 0:
                i += ds_len
            if i < 0 or i >= ds_len:
                raise IndexError(
                    f"Subset index out of range: {raw_i} for dataset length {ds_len}"
                )
            idx.append(i)
        return Subset(ds, idx)

    if n_samples is None:
        return ds

    n = min(int(n_samples), len(ds))
    return Subset(ds, list(range(n)))


def _make_loader(base_dm, dataset, shuffle: bool, drop_last: bool):
    if dataset is None:
        return None
    return DataLoader(
        dataset,
        batch_size=base_dm.batch_size,
        shuffle=shuffle,
        num_workers=base_dm.num_workers,
        pin_memory=base_dm.pin_memory,
        persistent_workers=base_dm.persistent_workers and base_dm.num_workers > 0,
        drop_last=drop_last,
    )

def main(config):
    print_only(
        "Instantiating base datamodule <{}>".format(config["datamodule"]["data_name"])
    )

    base_dm = getattr(look2hear.datas, config["datamodule"]["data_name"])(
        **config["datamodule"]["data_config"]
    )
    print_only(
        "Building subset loaders for <{}>".format(config["datamodule"]["data_name"])
    )

    # Keep old behavior by default: single deterministic sample at index 20.
    subset_cfg = config.get("subset", {})
    train_idx = subset_cfg.get("train_indices", [20])
    val_idx = subset_cfg.get("val_indices", [20])
    test_idx = subset_cfg.get("test_indices", None)
    n_train = subset_cfg.get("n_train", None)
    n_val = subset_cfg.get("n_val", None)
    n_test = subset_cfg.get("n_test", None)
    shuffle_train = bool(subset_cfg.get("shuffle_train", False))

    # TSE_ECHIDataModule should be setup with stage='fit' to avoid creating test set when has_test_targets=False.
    try:
        base_dm.setup(stage="fit")
    except TypeError:
        # Compatibility with datamodules exposing setup() without stage arg.
        base_dm.setup()

    train_subset = _subset_from_config(getattr(base_dm, "data_train", None), train_idx, n_train)
    val_subset = _subset_from_config(getattr(base_dm, "data_val", None), val_idx, n_val)
    test_subset = _subset_from_config(getattr(base_dm, "data_test", None), test_idx, n_test)

    train_loader = _make_loader(base_dm, train_subset, shuffle=shuffle_train, drop_last=True)
    val_loader = _make_loader(base_dm, val_subset, shuffle=False, drop_last=False)
    test_loader = _make_loader(base_dm, test_subset, shuffle=False, drop_last=False)
    
    # Define model and optimizer
    print_only(
        "Instantiating AudioNet <{}>".format(config["audionet"]["audionet_name"])
    )
    model = getattr(look2hear.models, config["audionet"]["audionet_name"])(
        sample_rate=config["datamodule"]["data_config"]["sample_rate"],
        **config["audionet"]["audionet_config"],
    )
    # import pdb; pdb.set_trace()
    print_only("Instantiating Optimizer <{}>".format(config["optimizer"]["optim_name"]))
    optimizer = make_optimizer(model.parameters(), **config["optimizer"])

    # Define scheduler
    scheduler = None
    if config["scheduler"]["sche_name"]:
        print_only(
            "Instantiating Scheduler <{}>".format(config["scheduler"]["sche_name"])
        )
        if config["scheduler"]["sche_name"] != "DPTNetScheduler":
            scheduler = getattr(torch.optim.lr_scheduler, config["scheduler"]["sche_name"])(
                optimizer=optimizer, **config["scheduler"]["sche_config"]
            )
        else:
            scheduler = {
                "scheduler": getattr(look2hear.system.schedulers, config["scheduler"]["sche_name"])(
                    optimizer, len(train_loader) // config["datamodule"]["data_config"]["batch_size"], 64
                ),
                "interval": "step",
            }

    # Just after instantiating, save the args. Easy loading in the future.
    config["main_args"]["exp_dir"] = os.path.join(
        os.getcwd(), "Experiments", "checkpoint", config["exp"]["exp_name"]
    )
    exp_dir = config["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(config, outfile)

    # Define Loss function.
    print_only(
        "Instantiating Loss, Train <{}>, Val <{}>".format(
            config["loss"]["train"]["sdr_type"], config["loss"]["val"]["sdr_type"]
        )
    )
    loss_func = {
        "train": _build_loss_from_config(config["loss"]["train"]),
        "val": _build_loss_from_config(config["loss"]["val"]),
    }

    print_only("Instantiating System <{}>".format(config["training"]["system"]))
    system = getattr(look2hear.system, config["training"]["system"])(
        audio_model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scheduler=scheduler,
        config=config,
    )

    # Define callbacks
    print_only("Instantiating ModelCheckpoint")
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir)
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        filename="{epoch}",
        monitor= config["training"]["checkpoint"]["monitor"], #val_loss, decides what validation will create a checkpoint
        mode=config["training"]["checkpoint"]["mode"],
        save_top_k=config["training"]["checkpoint"]["save_top_k"],
        verbose=config["training"]["checkpoint"]["verbose"],
        save_last=config["training"]["checkpoint"]["save_last"],
    )
    callbacks.append(checkpoint)

    if config["training"]["early_stop"]:
        print_only("Instantiating EarlyStopping")
        callbacks.append(EarlyStopping(**config["training"]["early_stop"]))
    #callbacks.append(MyRichProgressBar(theme=RichProgressBarTheme()))
    #eigener Callback, um Training per flag STOP manuell nach einem batch-Durchlauf beenden zu koennen
    stop_flag_path = Path(exp_dir) / "STOP" #hier muss ein file "STOP" existieren, um das Training zu beenden.
    callbacks.append(ExternalStopCallback(flag_path=stop_flag_path))


    # Don't ask GPU if they are not available.
    gpus = config["training"]["gpus"] if torch.cuda.is_available() else None
    distributed_backend = "cuda" if torch.cuda.is_available() else None

    # default logger used by trainer
    logger_dir = os.path.join(os.getcwd(), "Experiments", "tensorboard_logs")
    os.makedirs(os.path.join(logger_dir, config["exp"]["exp_name"]), exist_ok=True)
    # comet_logger = TensorBoardLogger(logger_dir, name=config["exp"]["exp_name"])
    comet_logger = WandbLogger(
            name=config["exp"]["exp_name"], 
            save_dir=os.path.join(logger_dir, config["exp"]["exp_name"]), 
            project="Enhancing-Conversations-for-Hearing-Impairements",
            # offline=True
    )

    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        devices=gpus,
        accelerator=distributed_backend,
        strategy=DDPStrategy(find_unused_parameters=True),
        limit_train_batches=1.0,  # Useful for fast experiment
        gradient_clip_val=5.0,
        logger=comet_logger,
        sync_batchnorm=True,
        # precision="bf16-mixed",
        # num_sanity_val_steps=0,
        # sync_batchnorm=True,
        # fast_dev_run=True,
    )
    ckpt_path = config["training"].get("ckpt_path", None) #restore training from checkpoint, if given in tiger.yml
    trainer.fit(system, ckpt_path=ckpt_path)
    print_only("Finished Training")
    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.audio_model.serialize()
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth")) #best model is saved in best_model.pth


if __name__ == "__main__":
    import yaml
    from pprint import pprint
    from look2hear.utils.parser_utils import (
        prepare_parser_from_dict,
        parse_args_as_dict,
    )

    args = parser.parse_args()
    with open(args.conf_dir) as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)

    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    # pprint(arg_dic)
    main(arg_dic)
