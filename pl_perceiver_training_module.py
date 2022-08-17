import torch

import torch.nn.functional as F
from torchmetrics.functional import accuracy
import pytorch_lightning as pl

from models.perceiver import Perceiver


class PerceiverTrainingModule(pl.LightningModule):
    """Classic Perceiver

    Args:
        input_shape (tuple of ints): Dimensions of input images
        latent_dim (int): Size of latent array
        embed_dim (int): Size of embedding output from linear projection layer
        attn_mlp_dim (int): Size of MLP
        trnfr_mlp_dim (int): Size transformer MLP
        trnfr_heads (int): Number of self-attention heads in the latent transformer
        dropout (float): dropout for network
        trnfr_layers (int): Number of decoders in the transformers
        n_blocks (int): Number of Perceiver blocks
        n_classes (int): Number of target classes
        batch_size (int): Batch size
        learning_rate (float): Learning Rate
    """

    def __init__(
        self,
        input_shape,
        latent_dim,
        embed_dim,
        attn_mlp_dim,
        trnfr_mlp_dim,
        trnfr_heads,
        dropout,
        trnfr_layers,
        n_blocks,
        n_classes,
        batch_size,
        learning_rate,
    ):
        super().__init__()

        # Key parameters
        self.save_hyperparameters()

        # Transformer with arbitrary number of encoders, heads, and hidden size

        self.model = Perceiver(
            input_shape=input_shape,
            latent_dim=latent_dim,
            embed_dim=embed_dim,
            attn_mlp_dim=attn_mlp_dim,
            trnfr_mlp_dim=trnfr_mlp_dim,
            trnfr_heads=trnfr_heads,
            dropout=dropout,
            trnfr_layers=trnfr_layers,
            n_blocks=n_blocks,
            n_classes=n_classes,
            batch_size=batch_size,
        )

    def forward(self, x):
        x = self.model(x)
        return x

    def evaluate(self, batch, stage=None):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        gamma = 0.1**0.5
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=gamma, last_epoch=-1, verbose=False
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
