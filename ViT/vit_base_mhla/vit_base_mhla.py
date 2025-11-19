import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from rope import apply_rope, compute_rope_params


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

        # print(
        #     f"Before Patch Embeddings x shape: {x.shape}"
        # )  # torch.Size([1, 768, 224, 224])
        x = self.patcher(x)
        # print(f"After Con2d layer x shape: {x.shape}")  # torch.Size([1, 768, 14, 14])
        x = self.flatten(x)
        # print(f"After Flattening x shape: {x.shape}")  # torch.Size([1, 768, 196])
        # Reshape to change 768 from channel dim to image dim
        # print(
        #     f"Final x shape to be sent to transformer encoder: {x.permute(0, 2, 1).shape}"
        # )  # torch.Size([1, 768, 196])
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
        # print(f"Before Layer Norm x shape: {x.shape}")
        x = self.layer_norm(x)
        # print(f"After Layer Norm x shape: {x.shape}")

        # print(f"Before MLP Layer x shape: {x.shape}")
        x = self.mlp(x)
        # print(f"After MLP Layer x shape: {x.shape}")

        return x


class MultiHeadLatentAttention(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        attn_dropout,
        num_latent_heads,
        latent_dim,
        compression_ratio,
    ):
        super().__init__()

        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = self.embedding_dim // self.num_heads
        self.attn_dropout = attn_dropout

        self.num_latent_heads = num_latent_heads
        self.latent_dim = latent_dim // compression_ratio

        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.o_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.latent_q = nn.Parameter(
            torch.randn(self.num_latent_heads, self.latent_dim)
        )
        self.latent_k = nn.Linear(
            self.embedding_dim, self.num_latent_heads * self.latent_dim, bias=False
        )
        self.latent_v = nn.Linear(
            self.embedding_dim, self.num_latent_heads * self.latent_dim, bias=False
        )
        self.latent_o = nn.Linear(
            self.num_latent_heads * self.latent_dim, self.embedding_dim, bias=False
        )

        self.attn_dropout = attn_dropout
        self.layer_norm = nn.LayerNorm(embedding_dim)

        self.cos, self.sin = compute_rope_params(self.head_dim)

    def forward(self, x):

        B, N, D = x.shape
        x = self.layer_norm(x)

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # print(f"KQV shapes before applying RoPE (mid-matmul-operation): {q.shape}")
        # Apply RoPE
        q = apply_rope(q, self.cos.to(q.device), self.sin.to(q.device))
        k = apply_rope(k, self.cos.to(k.device), self.sin.to(k.device))

        # print(f"KQV shapes after applying RoPE (mid-matmul-operation): {q.shape}")

        # print(f"Q Dimension: {q.shape}")

        latent_k = self.latent_k(x)  # [B, T, ]
        latent_k = latent_k.view(B, N, self.num_latent_heads, self.latent_dim)

        latent_v = self.latent_v(x)  # [B, T, ]
        latent_v = latent_v.view(B, N, self.num_latent_heads, self.latent_dim)

        latent_q = self.latent_q.unsqueeze(0).unsqueeze(2)
        latent_q = latent_q.expand(B, self.num_latent_heads, N, self.latent_dim)

        # print(f"Latent Q shape: {latent_q.shape}")
        # print(f"Latent K/V shape: {latent_k.shape}")

        # Reshaping for mat-mul
        latent_k = latent_k.view(B, self.num_latent_heads, N, self.latent_dim)
        latent_v = latent_v.view(B, self.num_latent_heads, N, self.latent_dim)

        latent_scores = torch.matmul(latent_q, latent_k.transpose(-2, -1)) / math.sqrt(
            self.latent_dim
        )

        latent_probs = F.softmax(latent_scores, dim=-1)
        latent_out = torch.matmul(latent_probs, latent_v)

        # print(f"Latent out shape: {latent_out.shape}")

        latent_out = (
            latent_out.transpose(1, 2)
            .contiguous()
            .view(B, N, self.num_latent_heads * self.latent_dim)
        )

        main_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        main_out = main_out.transpose(1, 2).contiguous().view(B, N, D)

        latent_out = self.latent_o(latent_out)

        output = self.o_proj(main_out) + latent_out

        return output


class TransformerEncoderBlock(nn.Module):

    def __init__(
        self,
        embedding_dim=768,
        num_heads=12,
        attn_dropout=0.1,
        mlp_size=3072,
        dropout=0.1,
        num_latent_heads=4,
        latent_dim=64,
        compression_ratio=8,
    ):
        super().__init__()

        self.msa_block = MultiHeadLatentAttention(
            embedding_dim,
            num_heads,
            attn_dropout,
            num_latent_heads,
            latent_dim,
            compression_ratio,
        )

        self.mlp_block = MLPBlock(embedding_dim, mlp_size, dropout)

    def forward(self, x):

        # print(f"Before MSA x shape: {x.shape}")
        x = self.msa_block(x) + x
        # print(f"After MSA x shape: {x.shape}")

        # print(f"Before MLP x shape: {x.shape}")
        x = self.mlp_block(x) + x
        # print(f"After MLP x shape: {x.shape}")

        return x


class ViTMHLA(nn.Module):
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
        num_latent_heads=4,
        latent_dim=64,
        compression_ratio=8,
    ):
        super(ViTMHLA, self).__init__()

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
                    embedding_dim,
                    num_heads,
                    attn_dropout,
                    mlp_size,
                    dropout,
                    num_latent_heads,
                    latent_dim,
                    compression_ratio,
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
        # x = self.positional_embedding + x
        x = self.embedding_dropout(x)

        # (B, 197, 768) -> Transformer Block
        x = self.transformer_encoder(x)
        x = self.classifier(
            x[:, 0]
        )  # return only the output class ! for calculating loss/acc

        return x


if __name__ == "__main__":
    # num_latent_heads, latent_dim, compression_ratio
    patch = ViTMHLA(224, 16, 768, 0, 3, 12, 3072, 0, 12, 1000, 4, 64, 8)

    x = torch.randn((1, 3, 224, 224))
    # print("Input shape: ", x.shape)

    out = patch(x)

    # print(f"Output shape: {out.shape}")
    # print(out.shape)
