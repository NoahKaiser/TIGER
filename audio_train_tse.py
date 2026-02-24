from dotenv import load_dotenv
load_dotenv()
import os
from pathlib import Path
import sys
import torch
from torch import Tensor
import argparse
import json
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


def _target_manifest_paths(split_dir: Path, n_src: int):
    return [split_dir / f"target_pos{pos}.json" for pos in range(1, n_src + 1)]


def _collect_target_ids_from_split(split_name: str, split_dir: Path, n_src: int):
    if not split_dir.is_dir():
        raise NotADirectoryError(f"{split_name} split directory not found: {split_dir}")

    missing = [p for p in _target_manifest_paths(split_dir, n_src) if not p.is_file()]
    if missing:
        missing_str = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"{split_name} split is missing required target manifests for n_src={n_src}: {missing_str}"
        )

    out = set()
    for p in _target_manifest_paths(split_dir, n_src):
        rows = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"{p} must contain a JSON list.")
        for row in rows:
            if not (isinstance(row, list) and len(row) >= 2):
                raise ValueError(
                    f"Invalid row in {p}: expected at least [path, spk_id, ...], got {row}"
                )
            out.add(str(row[1]).strip())
    return out


def run_tse_preflight_speaker_check(data_cfg: dict) -> None:
    """
    Fast fail if speaker IDs in target manifests are missing from spk_emb_path.
    Controlled by datamodule.data_config.verify_spk_alignment (default: True).
    """
    if not bool(data_cfg.get("verify_spk_alignment", True)):
        print_only("Skipping TSE speaker-id preflight (verify_spk_alignment=False).")
        return

    n_src = int(data_cfg.get("n_src", 4))
    has_test_targets = bool(data_cfg.get("has_test_targets", True))
    spk_emb_path = Path(str(data_cfg["spk_emb_path"]))

    if not spk_emb_path.is_file():
        raise FileNotFoundError(f"Speaker embedding file not found: {spk_emb_path}")

    split_specs = [
        ("train", Path(str(data_cfg["train_dir"]))),
        ("valid", Path(str(data_cfg["valid_dir"]))),
    ]
    if has_test_targets:
        split_specs.append(("test", Path(str(data_cfg["test_dir"]))))

    target_ids = set()
    for split_name, split_dir in split_specs:
        target_ids |= _collect_target_ids_from_split(split_name, split_dir, n_src)

    obj = torch.load(str(spk_emb_path), map_location="cpu")
    if not isinstance(obj, dict) or not obj:
        raise ValueError(
            f"{spk_emb_path} must contain a non-empty dict mapping spk_id->embedding, got: {type(obj)}"
        )
    emb_ids = {str(k) for k in obj.keys()}
    missing = sorted(target_ids - emb_ids)
    if missing:
        preview = missing[:20]
        raise RuntimeError(
            "TSE speaker-id preflight failed: "
            f"{len(missing)} speaker IDs are in target manifests but missing in {spk_emb_path}. "
            f"Examples: {preview}. "
            "Use DataPreProcess/verify_tse_spk_id_alignment.py and "
            "DataPreProcess/patch_missing_spk_embeddings_from_targets.py."
        )

    print_only(
        "TSE speaker-id preflight passed: target_ids={}, emb_ids={}, checked_splits={}.".format(
            len(target_ids),
            len(emb_ids),
            [name for name, _ in split_specs],
        )
    )


def build_loss_from_config(loss_cfg):
    sdr_obj = getattr(look2hear.losses, loss_cfg["sdr_type"])
    loss_name = loss_cfg.get("loss_func", None)

    if loss_name is None or str(loss_name).lower() in {"none", "null", ""}:
        return sdr_obj

    return getattr(look2hear.losses, loss_name)(
        sdr_obj,
        **loss_cfg.get("config", {}),
    )


def main(config):
    run_tse_preflight_speaker_check(config["datamodule"]["data_config"])

    print_only(
        "Instantiating datamodule <{}>".format(config["datamodule"]["data_name"])
    )
    datamodule: object = getattr(look2hear.datas, config["datamodule"]["data_name"])(
        **config["datamodule"]["data_config"]
    )
    datamodule.setup(stage="fit")

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = None
    
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
        "train": build_loss_from_config(config["loss"]["train"]),
        "val": build_loss_from_config(config["loss"]["val"]),
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
