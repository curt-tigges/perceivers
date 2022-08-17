import torch
import math
from torch import nn
from positional_image_embedding import PositionalImageEmbedding
import torch.functional as F


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


class PerceiverIOBlock(nn.Module):
    """Block consisting of one latent transformer, preceded by an optional cross-attention"""

    def __init__(
        self,
        embed_dim,
        attn_mlp_dim,
        trnfr_mlp_dim,
        trnfr_heads,
        dropout,
        trnfr_layers,
        inner_ca=False,
    ):
        super().__init__()
        self.inner_ca = inner_ca

        # Optional cross-attention. Can be omitted
        if self.inner_ca:
            self.cross_attention = PerceiverAttention(
                embed_dim, attn_mlp_dim, n_heads=1, dropout=dropout
            )

        self.latent_transformer = LatentTransformer(
            embed_dim, trnfr_mlp_dim, trnfr_heads, dropout, trnfr_layers
        )

    def forward(self, x, l):
        if self.inner_ca:
            l = self.cross_attention(x, l)

        l = self.latent_transformer(l)

        return l


class ClassifierIO(nn.Module):
    """Perceiver IO classification calculation"""

    def __init__(self, embed_dim, output_dim, output_heads, n_classes, dropout=0.0):
        super().__init__()
        # learnable label embedding
        self.n_classes = n_classes
        self.output_dim = output_dim

        self.label_emb = nn.Parameter(torch.rand(n_classes, 1, output_dim))

        self.output_attn = PerceiverAttention(
            embed_dim, embed_dim * 4, output_heads, dropout=dropout
        )

        self.fc1 = nn.Linear(output_dim, output_dim)
        self.fc2 = nn.Linear(self.n_classes * self.output_dim, n_classes)

    def forward(self, x):
        # latent, batch, embed
        L, B, E = x.shape
        # print(f"Latent shape: {x.shape}")
        # print(f"Output emb shape: {self.label_emb.shape}")
        output_emb = self.label_emb.repeat(1, B, 1)
        # print(f"Output array shape: {output_emb.shape}")

        x = self.output_attn(x, output_emb)
        # print(x.shape)

        x = self.fc1(x)
        x = F.gelu(x)
        x = torch.reshape(x, (B, self.n_classes * self.output_dim))
        # x = x.mean(dim=0)
        x = self.fc2(x)

        return x


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


class PerceiverIO(nn.Module):
    """Full Perceiver IO model"""

    def __init__(
        self,
        input_shape,
        latent_dim,
        embed_dim,
        output_dim,
        attn_mlp_dim,
        trnfr_mlp_dim,
        trnfr_heads,
        dropout,
        trnfr_layers,
        n_blocks,
        n_classes,
        batch_size,
        inner_ca=False,
    ):
        super().__init__()

        # Initialize latent array
        self.latent = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.zeros((latent_dim, 1, embed_dim)), mean=0, std=0.02, a=-2, b=2
            )
        )

        # Initialize embedding with position encoding
        self.embed = PositionalImageEmbedding(input_shape, 1, embed_dim)

        # Initialize initial block with cross-attention
        self.initial_perceiver_block = PerceiverIOBlock(
            embed_dim=embed_dim,
            attn_mlp_dim=attn_mlp_dim,
            trnfr_mlp_dim=trnfr_mlp_dim,
            trnfr_heads=trnfr_heads,
            dropout=dropout,
            trnfr_layers=trnfr_layers,
            inner_ca=True,
        )

        # Initialize arbitrary number of Perceiver blocks; will be transformer
        # blocks unless inner_ca (inner cross-attention) is enabled
        self.perceiver_blocks = nn.ModuleList(
            [
                PerceiverIOBlock(
                    embed_dim=embed_dim,
                    attn_mlp_dim=attn_mlp_dim,
                    trnfr_mlp_dim=trnfr_mlp_dim,
                    trnfr_heads=trnfr_heads,
                    dropout=dropout,
                    trnfr_layers=trnfr_layers,
                    inner_ca=inner_ca,
                )
                for b in range(n_blocks)
            ]
        )

        # PerceiverIO classification layer
        self.classifier = ClassifierIO(
            embed_dim=embed_dim,
            output_dim=output_dim,
            output_heads=8,
            n_classes=n_classes,
        )

        # Original classification layer
        # self.classifier = Classifier(embed_dim=embed_dim, latent_dim=latent_dim, batch_size=batch_size, n_classes=n_classes)

    def forward(self, x):
        # First we expand our latent query matrix to size of batch
        batch_size = x.shape[0]
        latent = self.latent.expand(-1, batch_size, -1)

        # Next, we pass the image through the embedding module to get flattened input
        x = self.embed(x)

        #
        latent = self.initial_perceiver_block(x, latent)

        # Next, we iteratively pass the latent matrix and image embedding through
        # perceiver blocks
        for pb in self.perceiver_blocks:
            latent = pb(x, latent)

        # Finally, we project the output to the number of target classes
        latent = self.classifier(latent)

        return latent
