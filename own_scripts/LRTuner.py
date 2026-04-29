from __future__ import annotations

import argparse
import copy
import warnings
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv

try:
    from lightning.pytorch import Trainer
    from lightning.pytorch.tuner import Tuner
except Exception:  # fallback for pytorch_lightning installations
    from pytorch_lightning import Trainer
    from pytorch_lightning.tuner import Tuner

import look2hear.datas
import look2hear.losses
import look2hear.models
import look2hear.system
from look2hear.system import make_optimizer

warnings.filterwarnings("ignore")


def _build_loss_from_config(loss_cfg: dict):
    sdr_obj = getattr(look2hear.losses, loss_cfg["sdr_type"])
    loss_name = loss_cfg.get("loss_func", None)

    if loss_name is None or str(loss_name).lower() in {"none", "null", ""}:
        return sdr_obj

    return getattr(look2hear.losses, loss_name)(
        sdr_obj,
        **loss_cfg.get("config", {}),
    )


def _instantiate_datamodule(config: dict):
    dm_cls = getattr(look2hear.datas, config["datamodule"]["data_name"])
    datamodule = dm_cls(**config["datamodule"]["data_config"])

    try:
        datamodule.setup(stage="fit")
    except TypeError:
        datamodule.setup()

    if hasattr(datamodule, "train_dataloader"):
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader() if hasattr(datamodule, "val_dataloader") else None

        test_loader = None
        if hasattr(datamodule, "test_dataloader"):
            try:
                test_loader = datamodule.test_dataloader()
            except Exception:
                test_loader = None
        return datamodule, train_loader, val_loader, test_loader

    train_loader, val_loader, test_loader = datamodule.make_loader
    return datamodule, train_loader, val_loader, test_loader


def _build_scheduler(config: dict, optimizer, train_loader):
    scheduler_cfg = config.get("scheduler", {})
    sche_name = scheduler_cfg.get("sche_name")

    if sche_name is None or str(sche_name).lower() in {"none", "null", ""}:
        return None

    if sche_name != "DPTNetScheduler":
        return getattr(torch.optim.lr_scheduler, sche_name)(
            optimizer=optimizer,
            **scheduler_cfg.get("sche_config", {}),
        )

    batch_size = int(config["datamodule"]["data_config"].get("batch_size", 1))
    steps_per_epoch = max(1, len(train_loader) // max(1, batch_size))
    return {
        "scheduler": getattr(look2hear.system.schedulers, sche_name)(
            optimizer,
            steps_per_epoch,
            64,
        ),
        "interval": "step",
    }


def _build_system(config: dict, train_loader, val_loader, test_loader, use_scheduler: bool):
    model = getattr(look2hear.models, config["audionet"]["audionet_name"])(
        sample_rate=config["datamodule"]["data_config"]["sample_rate"],
        **config["audionet"]["audionet_config"],
    )

    optimizer = make_optimizer(model.parameters(), **config["optimizer"])
    scheduler = _build_scheduler(config, optimizer, train_loader) if use_scheduler else None

    loss_func = {
        "train": _build_loss_from_config(config["loss"]["train"]),
        "val": _build_loss_from_config(config["loss"]["val"]),
    }

    system_cls = getattr(look2hear.system, config["training"]["system"])
    system = system_cls(
        audio_model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scheduler=scheduler,
        config=config,
    )
    return system


def _parse_devices(devices: str):
    value = devices.strip().lower()
    if value == "auto":
        return "auto"
    if "," in value:
        return [int(v.strip()) for v in value.split(",") if v.strip()]
    return int(value)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Lightning LR finder for look2hear models.")
    parser.add_argument("--conf_dir", type=str, required=True, help="Path to a training YAML config.")
    parser.add_argument("--min_lr", type=float, default=1e-8, help="Minimum LR for lr_find.")
    parser.add_argument("--max_lr", type=float, default=1.0, help="Maximum LR for lr_find.")
    parser.add_argument("--num_training", type=int, default=100, help="Number of lr_find steps.")
    parser.add_argument(
        "--mode",
        type=str,
        default="exponential",
        choices=["exponential", "linear"],
        help="Sweep mode for lr_find.",
    )
    parser.add_argument(
        "--early_stop_threshold",
        type=float,
        default=4.0,
        help="Stop sweep if loss diverges by this factor.",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "cuda", "mps", "tpu"],
        help="Trainer accelerator.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="1",
        help="Trainer devices (e.g. '1', '0,1', 'auto').",
    )
    parser.add_argument("--strategy", type=str, default="auto", help="Trainer strategy.")
    parser.add_argument(
        "--limit_train_batches",
        type=float,
        default=1.0,
        help="Fraction or count of train batches to use per epoch.",
    )
    parser.add_argument(
        "--use_scheduler",
        action="store_true",
        help="Include config scheduler during lr_find (default: disabled).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save lr_find plot to --plot_path.",
    )
    parser.add_argument(
        "--plot_path",
        type=str,
        default="Experiments/lr_find.png",
        help="Output path for lr_find plot.",
    )
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    conf_path = Path(args.conf_dir)
    if not conf_path.is_file():
        raise FileNotFoundError(f"Config not found: {conf_path}")

    with open(conf_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config = copy.deepcopy(config)
    _, train_loader, val_loader, test_loader = _instantiate_datamodule(config)
    system = _build_system(config, train_loader, val_loader, test_loader, use_scheduler=args.use_scheduler)

    accelerator = args.accelerator
    devices = _parse_devices(args.devices)

    if accelerator in {"gpu", "cuda"} and not torch.cuda.is_available():
        print("CUDA not available. Falling back to CPU with devices=1")
        accelerator = "cpu"
        devices = 1

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=args.strategy,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        limit_val_batches=0.0,
        limit_train_batches=args.limit_train_batches,
    )

    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(
        system,
        train_dataloaders=train_loader,
        min_lr=args.min_lr,
        max_lr=args.max_lr,
        num_training=args.num_training,
        mode=args.mode,
        early_stop_threshold=args.early_stop_threshold,
        update_attr=False,
    )

    print(lr_finder.results)
    suggested = lr_finder.suggestion()
    print("suggested lr", suggested)

    if suggested is not None:
        print(f"Set optimizer.lr in your config to about {suggested:.6g} for full training.")
    else:
        print("No LR suggestion was found. Try increasing --num_training or adjusting --min_lr/--max_lr.")

    if args.plot:
        fig = lr_finder.plot(suggest=True)
        out_path = Path(args.plot_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved LR finder plot to: {out_path}")


if __name__ == "__main__":
    main()
