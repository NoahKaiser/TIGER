import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections.abc import MutableMapping
from look2hear.utils.speaker_embedding_utils import build_spk_table_from_pt

# Optional: keep import but do not use for now
# from speechbrain.augment.time_domain import SpeedPerturb


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class AudioLightningModuleTSE_ECHI(pl.LightningModule):
    """
    Minimal TSE LightningModule:
      - batch: (mixture[B,T], target[B,T], spk_idx[B], utt_id[list])
      - lookup speaker embedding table and pass spk_emb into audio_model
      - audio_model must accept: audio_model(wav, spk_emb=...)
      - for first checkpoint, audio_model may ignore spk_emb
    """

    def __init__(
        self,
        audio_model=None,
        video_model=None,
        optimizer=None,
        loss_func=None,
        train_loader=None,
        val_loader=None,
        test_loader=None,
        scheduler=None,
        config=None,
        spk_emb_path: str | None = None,  # if None, read from config
    ):
        super().__init__()
        self.audio_model = audio_model
        self.video_model = video_model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scheduler = scheduler
        self.config = {} if config is None else config

        # Resolve speaker embedding path (prefer explicit arg; fallback to config)
        path = spk_emb_path
        if path is None:
            # try a few common config locations
            path = (
                self.config.get("speaker_embedding", {}).get("path")
                or self.config.get("datamodule", {}).get("data_config", {}).get("spk_emb_path")
                or self.config.get("audionet", {}).get("audionet_config", {}).get("spk_emb_path")
            )
        if path is None:
            raise ValueError(
                "Speaker embedding path not provided. Pass spk_emb_path=... or set it in config "
                "(e.g., config['speaker_embedding']['path'])."
            )

        # Build spk_table [N,d] and keep it as a buffer (moved with the module, saved in state_dict)


        self.spk2idx, self.spk_ids, spk_table = build_spk_table_from_pt(path, sort_ids=True)
        self.register_buffer("spk_table", spk_table, persistent=True)  # [N,d]

        # Save hparams (do NOT include spk_table itself)
        self.default_monitor = "val_loss"
        self.save_hyperparameters(self.config_to_hparams(self.config))
        self.validation_step_outputs = []

        # TSE note: old SpeedAug was separation-specific; disable for now
        if self.config.get("training", {}).get("SpeedAug", False):
            self.print("[WARN] SpeedAug=True but TSE SpeedAug is not implemented yet. Disabling for now.")
            self.config["training"]["SpeedAug"] = False

    def forward(self, wav: torch.Tensor, spk_idx: torch.Tensor):
        """
        wav: [B,T] or [B,1,T] (TIGER accepts both)
        spk_idx: [B] LongTensor
        """
        spk_emb = F.embedding(spk_idx, self.spk_table)  # [B,d]
        return self.audio_model(wav, spk_emb=spk_emb)

    def training_step(self, batch, batch_nb):
        mixtures, target, spk_idx, _utt_id = batch  # target: [B,T]
        # Ensure target shape matches model output for num_sources=1
        targets = target.unsqueeze(1)  # [B,1,T]

        est = self(mixtures, spk_idx)  # expected [B,1,T] if TIGER num_sources=1
        loss = self.loss_func["train"](est, targets)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        mixtures, target, spk_idx, _utt_id = batch
        targets = target.unsqueeze(1)  # [B,1,T]

        est = self(mixtures, spk_idx)
        loss = self.loss_func["val"](est, targets)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)
        self.validation_step_outputs.append(loss)
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        val_loss = torch.mean(self.all_gather(avg_loss))

        self.log("lr", self.optimizer.param_groups[0]["lr"], on_epoch=True, prog_bar=True, sync_dist=True)

        # keep your wandb-style logging if logger supports it
        if getattr(self, "logger", None) is not None and hasattr(self.logger, "experiment"):
            try:
                self.logger.experiment.log({"learning_rate": self.optimizer.param_groups[0]["lr"], "epoch": self.current_epoch})
                self.logger.experiment.log({"val_se_sisdr_like": -val_loss, "epoch": self.current_epoch})
            except Exception:
                pass

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        if self.scheduler is None:
            return self.optimizer

        if not isinstance(self.scheduler, (list, tuple)):
            self.scheduler = [self.scheduler]

        epoch_schedulers = []
        for sched in self.scheduler:
            if not isinstance(sched, dict):
                if isinstance(sched, ReduceLROnPlateau):
                    sched = {"scheduler": sched, "monitor": self.default_monitor}
                epoch_schedulers.append(sched)
            else:
                sched.setdefault("monitor", self.default_monitor)
                sched.setdefault("frequency", 1)
                if sched.get("interval") == "batch":
                    sched["interval"] = "step"
                assert sched["interval"] in ["epoch", "step"]
                epoch_schedulers.append(sched)
        return [self.optimizer], epoch_schedulers

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def on_save_checkpoint(self, checkpoint):
        checkpoint["training_config"] = self.config
        checkpoint["spk_ids"] = self.spk_ids  # to debug mapping/order if needed
        return checkpoint

    @staticmethod
    def config_to_hparams(dic):
        dic = flatten_dict(dic)
        for k, v in dic.items():
            if v is None:
                dic[k] = str(v)
            elif isinstance(v, (list, tuple)):
                dic[k] = torch.tensor(v)
        return dic