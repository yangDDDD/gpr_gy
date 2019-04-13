import logging
import torch

from src.utils.mybot import MyBot
from pathlib import Path

class GAPBot(MyBot):
    def __init__(self, model, train_loader, val_loader, *, optimizer, clip_grad=0,
                 avg_window=100, logger=None,
                 checkpoint_dir="./cache/model_cache/", batch_idx=0,
                 device="cuda:0"):
        super().__init__(
            model, train_loader, val_loader,
            optimizer=optimizer, clip_grad=clip_grad,
            logger=logger, checkpoint_dir=checkpoint_dir,
            batch_idx=batch_idx,
            device=device
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.loss_format = "%.6f"

    def extract_prediction(self, tensor):
        return tensor

    def snapshot(self):
        """Override the snapshot method because Kaggle kernel has limited local disk space."""
        loss = self.eval(self.val_loader)
        loss_str = self.loss_format % loss
        self.logger.info("Snapshot loss %s", loss_str)
        self.logger.tb_scalars(
            "losses", {"val": loss}, self.step)
        target_path = (
                self.checkpoint_dir / "best.pth")
        if not self.best_performers or (self.best_performers[0][0] > loss):
            torch.save(self.model.state_dict(), target_path)
            self.best_performers = [(loss, target_path, self.step)]
            self.logger.info("Saving checkpoint %s...", target_path)
        assert Path(target_path).exists()
        return loss