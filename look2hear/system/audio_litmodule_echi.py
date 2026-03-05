import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections.abc import MutableMapping
from speechbrain.augment.time_domain import SpeedPerturb

"""
this AudioLightningModuleECHI is almost a copy of the AudioLightningModule 
but does not use the test set for validation during training

"""

def flatten_dict(d, parent_key="", sep="_"):
    """Flattens a dictionary into a single-level dictionary while preserving
    parent keys. Taken from
    `SO <https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys>`_

    Args:
        d (MutableMapping): Dictionary to be flattened.
        parent_key (str): String to use as a prefix to all subsequent keys.
        sep (str): String to use as a separator between two key levels.

    Returns:
        dict: Single-level dictionary, flattened.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class AudioLightningModuleECHI(pl.LightningModule):
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
        # Speed Aug
        self.speedperturb = SpeedPerturb(
            self.config["datamodule"]["data_config"]["sample_rate"],
            speeds=[95, 100, 105]
        )
        # Save lightning"s AttributeDict under self.hparams
        self.default_monitor = "val_loss"
        self.save_hyperparameters(self.config_to_hparams(self.config))
        # self.print(self.audio_model)
        self.validation_step_outputs = []
        training_cfg = self.config.get("training", {})
        self.lambda_res = float(training_cfg.get("lambda_res", 0.0))
        self.pit_activity_tau = float(training_cfg.get("pit_activity_tau", 1e-6))
        self.residual_eps = 1e-8

        

    def forward(self, wav, mouth=None):
        """Applies forward pass of the model.

        Returns:
            :class:`torch.Tensor`
        """
        return self.audio_model(wav)

    @staticmethod
    def _to_wave_2d(wav):
        if wav.ndim == 2:
            return wav
        if wav.ndim == 3:
            return wav.reshape(wav.shape[0] * wav.shape[1], wav.shape[2])
        raise ValueError(f"Expected mixture waveform with 2 or 3 dims, got {wav.shape}")

    def _forward_speech_and_residual(self, mixtures):
        try:
            model_out = self.audio_model(mixtures, return_residual=True)
        except TypeError:
            model_out = self.audio_model(mixtures)

        if isinstance(model_out, tuple):
            est_sources, residual_hat = model_out
        else:
            est_sources = model_out
            residual_hat = self._to_wave_2d(mixtures) - est_sources.sum(dim=1)

        return est_sources, residual_hat

    def _compute_speech_loss(self, est_sources, targets):
        reduce_kwargs = {
            "target_energy": (targets ** 2).mean(dim=2),
            "tau": self.pit_activity_tau,
        }
        train_loss_fn = self.loss_func["train"]
        try:
            return train_loss_fn(est_sources, targets, reduce_kwargs=reduce_kwargs)
        except TypeError:
            return train_loss_fn(est_sources, targets)

    def _compute_residual_loss(self, residual_hat, mixtures, targets):
        mixtures_2d = self._to_wave_2d(mixtures)
        residual_target = mixtures_2d - targets.sum(dim=1)
        num = ((residual_hat - residual_target) ** 2).mean(dim=1)
        den = (mixtures_2d ** 2).mean(dim=1) + self.residual_eps
        return (num / den).mean()

    def training_step(self, batch, batch_nb):
        mixtures, targets, _ = batch
        
        new_targets = []
        min_len = -1
        if self.config["training"]["SpeedAug"] == True:
            with torch.no_grad():
                for i in range(targets.shape[1]):
                    new_target = self.speedperturb(targets[:, i, :])
                    new_targets.append(new_target)
                    if i == 0:
                        min_len = new_target.shape[-1]
                    else:
                        if new_target.shape[-1] < min_len:
                            min_len = new_target.shape[-1]

                targets = torch.zeros(
                            targets.shape[0],
                            targets.shape[1],
                            min_len,
                            device=targets.device,
                            dtype=torch.float,
                        )
                for i, new_target in enumerate(new_targets):
                    targets[:, i, :] = new_targets[i][:, 0:min_len]
                    
                mixtures = targets.sum(1)
        # print(mixtures.shape)
        est_sources, residual_hat = self._forward_speech_and_residual(mixtures)
        speech_loss = self._compute_speech_loss(est_sources, targets)
        residual_loss = self._compute_residual_loss(residual_hat, mixtures, targets)
        loss = speech_loss + self.lambda_res * residual_loss

        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )
        self.log(
            "train_speech_loss",
            speech_loss,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            logger=True,
        )
        self.log(
            "train_residual_loss",
            residual_loss,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            logger=True,
        )

        return {"loss": loss}


    def validation_step(self, batch, batch_nb):
            mixtures, targets, _ = batch
            # print(mixtures.shape)
            est_sources = self(mixtures)
            loss = self.loss_func["val"](est_sources, targets)
            self.log(
                "val_loss",
                loss,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                logger=True,
            )
            
            self.validation_step_outputs.append(loss)
            
            return {"val_loss": loss}



    def on_validation_epoch_end(self):
        # val
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        val_loss = torch.mean(self.all_gather(avg_loss))
        self.log(
            "lr",
            self.optimizer.param_groups[0]["lr"],
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.logger.experiment.log(
            {"learning_rate": self.optimizer.param_groups[0]["lr"], "epoch": self.current_epoch}
        )
        self.logger.experiment.log(
            {"val_pit_se-sisdr": -val_loss, "epoch": self.current_epoch}
        )

        self.validation_step_outputs.clear()  # free memory


    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        if self.scheduler is None:
            return self.optimizer

        if not isinstance(self.scheduler, (list, tuple)):
            self.scheduler = [self.scheduler]  # support multiple schedulers

        epoch_schedulers = []
        for sched in self.scheduler:
            if not isinstance(sched, dict):
                if isinstance(sched, ReduceLROnPlateau):
                    sched = {"scheduler": sched, "monitor": self.default_monitor}
                epoch_schedulers.append(sched)
            else:
                sched.setdefault("monitor", self.default_monitor)
                sched.setdefault("frequency", 1)
                # Backward compat
                if sched["interval"] == "batch":
                    sched["interval"] = "step"
                assert sched["interval"] in [
                    "epoch",
                    "step",
                ], "Scheduler interval should be either step or epoch"
                epoch_schedulers.append(sched)
        return [self.optimizer], epoch_schedulers

    # def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
    #     if metric is None:
    #         scheduler.step()
    #     else:
    #         scheduler.step(metric)
    
    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader

    def on_save_checkpoint(self, checkpoint):
        """Overwrite if you want to save more things in the checkpoint."""
        checkpoint["training_config"] = self.config
        return checkpoint

    @staticmethod
    def config_to_hparams(dic):
        """Sanitizes the config dict to be handled correctly by torch
        SummaryWriter. It flatten the config dict, converts ``None`` to
        ``"None"`` and any list and tuple into torch.Tensors.

        Args:
            dic (dict): Dictionary to be transformed.

        Returns:
            dict: Transformed dictionary.
        """
        dic = flatten_dict(dic)
        for k, v in dic.items():
            if v is None:
                dic[k] = str(v)
            elif isinstance(v, (list, tuple)):
                dic[k] = torch.tensor(v)
        return dic
