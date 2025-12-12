from pathlib import Path
import pytorch_lightning as pl

class ExternalStopCallback(pl.Callback):
    def __init__(self, flag_path):
        super().__init__()
        self.flag_path = Path(flag_path)

    def on_train_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print(f"[ExternalStopCallback] Watching stop file: {self.flag_path}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_rank != 0:
            return
        if self.flag_path.exists():
            print("[ExternalStopCallback] STOP detected → setting trainer.should_stop = True")
            trainer.should_stop = True
