import torch
import torch.nn as nn

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class PatchEmbeddings(nn.Module):

    def __init__(self, in_channels, patch_size, embedding_dim):
        super().__init__()

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim

        self.patcher = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.embedding_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # Flatten 2D representation into 1D [B, T, Hidden_size]
        # Technically [B, T, 768] -> for sending it to transformer's attention layers
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):
        # (B, C, H, W)
        image_resolution = x.shape[-1]
        assert (
            image_resolution % self.patch_size == 0
        ), f"Image Size {(image_resolution, image_resolution)} should be divisible by patch size {(self.patch_size)}"

        print(
            f"Before Patch Embeddings x shape: {x.shape}"
        )  # torch.Size([1, 768, 224, 224])
        x = self.patcher(x)
        print(f"After Con2d layer x shape: {x.shape}")  # torch.Size([1, 768, 14, 14])
        x = self.flatten(x)
        print(f"After Flattening x shape: {x.shape}")  # torch.Size([1, 768, 196])
        # Reshape to change 768 from channel dim to image dim
        print(
            f"Final x shape to be sent to transformer encoder: {x.permute(0, 2, 1).shape}"
        )  # torch.Size([1, 768, 196])
        return x.permute(0, 2, 1)  # [1, 768, 196] -> [1, 196, 768]


class MLPBlock(nn.Module):

    def __init__(self, embedding_dim, mlp_size, dropout):
        super().__init__()

        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_size, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        print(f"Before Layer Norm x shape: {x.shape}")
        x = self.layer_norm(x)
        print(f"After Layer Norm x shape: {x.shape}")

        print(f"Before MLP Layer x shape: {x.shape}")
        x = self.mlp(x)
        print(f"After MLP Layer x shape: {x.shape}")

        return x


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, embedding_dim, num_heads, attn_dropout):
        super().__init__()

        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.multihead_attn = nn.MultiheadAttention(
            embedding_dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True
        )

    def forward(self, x):

        # as per paper layer_norm -> MHA
        x = self.layer_norm(x)

        print(f"After Layer-Norm x shape: {x.shape}")
        attn_output, _ = self.multihead_attn(
            query=x, key=x, value=x, need_weights=False
        )
        print(f"Self-Attn shape: {attn_output.shape}")
        return attn_output


class TransformerEncoderBlock(nn.Module):

    def __init__(self, embedding_dim, num_heads, attn_dropout, mlp_size, dropout):
        super().__init__()

        self.msa_block = MultiHeadSelfAttention(embedding_dim, num_heads, attn_dropout)

        self.mlp_block = MLPBlock(embedding_dim, mlp_size, dropout)

    def forward(self, x):

        print(f"Before MSA x shape: {x.shape}")
        x = self.msa_block(x) + x
        print(f"After MSA x shape: {x.shape}")

        print(f"Before MLP x shape: {x.shape}")
        x = self.mlp_block(x) + x
        print(f"After MLP x shape: {x.shape}")

        return x


class ViTBase(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        embedding_dim=768,
        dropout=0.1,
        in_channels=3,
        num_heads=12,
        mlp_size=3072,
        attn_dropout=0.1,
        num_transformer_layers=12,
        num_classes=1000,
    ):
        super(ViTBase, self).__init__()

        assert (
            image_size % patch_size == 0
        ), f"Image size {(image_size, image_size)} must be divisible by patch size: f{(patch_size)}"

        # embedded patches (as per diagram)
        # 1. Calculating num_patches
        # (224 * 224) / 16**2 = 196 patches
        # We then add the class token [CLS], so the total sequence length would be 1 + 196
        # 197 is the sequence length (for base-patch16-224)
        self.num_patches = (image_size * image_size) // patch_size**2

        # 2. Creating the class token
        # In config hidden_size refers to the embedding_dim
        self.class_embedding = nn.Parameter(
            data=torch.randn(1, 1, embedding_dim), requires_grad=True
        )

        # 3. Creating the positional encoding (as per dia Patch + PE goes to Transformer Encoder)
        self.positional_embedding = nn.Parameter(
            data=torch.randn(1, self.num_patches + 1, embedding_dim),
            requires_grad=True,  # 1 + 196
        )

        # 4.Creating Patch Embeddings + Dropout
        self.embedding_dropout = nn.Dropout(dropout)
        self.patch_embeddings = PatchEmbeddings(
            in_channels=in_channels,
            patch_size=patch_size,
            embedding_dim=embedding_dim,  # alias hidden size
        )

        self.transformer_encoder = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    embedding_dim, num_heads, attn_dropout, mlp_size, dropout
                )
                for _ in range(num_transformer_layers)
            ]
        )

        # Final Layer to do the classifications
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, out_features=num_classes),
        )

    def forward(self, x):
        # Given an input image of shape:
        # (B, C, H, W)
        batch_size = x.shape[0]
        class_token = self.class_embedding.expand(batch_size, -1, -1)
        x = self.patch_embeddings(x)
        # After Patch Embeddings x -> (B, 196, 768)
        x = torch.cat([class_token, x], dim=1)
        # After adding [CLS] token -> (B, 197, 768)
        x = self.positional_embedding + x
        x = self.embedding_dropout(x)

        # (B, 197, 768) -> Transformer Block
        x = self.transformer_encoder(x)
        x = self.classifier(
            x[:, 0]
        )  # return only the output class ! for calculating loss/acc

        return x


if __name__ == "__main__":

    patch = ViTBase(224, 16, 768, 0, 3, 12, 3072, 0, 12, 1000)

    x = torch.randn((1, 3, 224, 224))
    print("Input shape: ", x.shape)

    out = patch(x)

    print(f"Output shape: {out.shape}")
    print(out.shape)
