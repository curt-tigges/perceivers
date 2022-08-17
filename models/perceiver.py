import torch
import math
from torch import nn
from positional_image_embedding import PositionalImageEmbedding


class PerceiverAttention(nn.Module):
    """Basic decoder block used both for cross-attention and the latent transformer"""

    def __init__(self, embed_dim, mlp_dim, n_heads, dropout=0.0):
        super().__init__()

        self.lnorm1 = nn.LayerNorm(embed_dim)
        self.lnormq = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads)

        self.lnorm2 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(mlp_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, q):
        # x will be of shape [PIXELS x BATCH_SIZE x EMBED_DIM]
        # q will be of shape [LATENT_DIM x BATCH_SIZE x EMBED_DIM] when this is
        # used for cross-attention; otherwise same as x

        # attention block
        out = self.lnorm1(x)
        # q = self.lnormq(q)
        out, _ = self.attn(query=q, key=x, value=x)
        # out will be of shape [LATENT_DIM x BATCH_SIZE x EMBED_DIM] after matmul
        # when used for cross-attention; otherwise same as x

        # first residual connection
        resid = out + q

        # dense block
        out = self.lnorm2(resid)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)
        out = self.drop(out)

        # second residual connection
        out = out + resid

        return out


class PerceiverBlock(nn.Module):
    """Block consisting of one cross-attention layer and one latent transformer"""

    def __init__(
        self, embed_dim, attn_mlp_dim, trnfr_mlp_dim, trnfr_heads, dropout, trnfr_layers
    ):
        super().__init__()

        self.cross_attention = PerceiverAttention(
            embed_dim, attn_mlp_dim, n_heads=1, dropout=dropout
        )

        self.latent_transformer = LatentTransformer(
            embed_dim, trnfr_mlp_dim, trnfr_heads, dropout, trnfr_layers
        )

    def forward(self, x, l):
        l = self.cross_attention(x, l)

        l = self.latent_transformer(l)

        return l


class Classifier(nn.Module):
    """Original Perceiver classification calculation"""

    def __init__(self, embed_dim, latent_dim, batch_size, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        # latent, batch, embed

        x = self.fc1(x)
        x = x.mean(dim=0)
        x = self.fc2(x)

        return x


class LatentTransformer(nn.Module):
    """Latent transformer module with n_layers count of decoders"""

    def __init__(self, embed_dim, mlp_dim, n_heads, dropout, n_layers):
        super().__init__()
        self.transformer = nn.ModuleList(
            [
                PerceiverAttention(
                    embed_dim=embed_dim,
                    mlp_dim=mlp_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                )
                for l in range(n_layers)
            ]
        )

    def forward(self, l):

        for trnfr in self.transformer:
            l = trnfr(l, l)

        return l


class Perceiver(nn.Module):
    """Complete Perceiver"""

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
    ):
        super().__init__()

        # Initialize latent array
        self.latent = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.zeros((latent_dim, 1, embed_dim)), mean=0, std=0.02, a=-2, b=2
            )
        )
        # In the paper, a truncated normal distribution was used for initialization,
        # so I used this hidden torch function to create it.

        # Initialize embedding with position encoding
        self.embed = PositionalImageEmbedding(input_shape, 1, embed_dim)

        # Initialize arbitrary number of Perceiver blocks
        self.perceiver_blocks = nn.ModuleList(
            [
                PerceiverBlock(
                    embed_dim=embed_dim,
                    attn_mlp_dim=attn_mlp_dim,
                    trnfr_mlp_dim=trnfr_mlp_dim,
                    trnfr_heads=trnfr_heads,
                    dropout=dropout,
                    trnfr_layers=trnfr_layers,
                )
                for b in range(n_blocks)
            ]
        )

        # learnable label embedding
        # self.label_emb = nn.Parameter(torch.rand(1, n_classes, latent_dim))

        # Initialize classification layer
        self.classifier = Classifier(
            embed_dim=embed_dim,
            latent_dim=latent_dim,
            batch_size=batch_size,
            n_classes=n_classes,
        )

    def forward(self, x):
        # First we expand our latent query matrix to size of batch
        batch_size = x.shape[0]
        latent = self.latent.expand(-1, batch_size, -1)

        # Next, we pass the image through the embedding module to get flattened input
        x = self.embed(x)

        # Next, we iteratively pass the latent matrix and image embedding through
        # perceiver blocks
        for pb in self.perceiver_blocks:
            latent = pb(x, latent)
        # print(latent.shape)

        # Finally, we project the output to the number of target classes
        latent = self.classifier(latent)

        return latent
