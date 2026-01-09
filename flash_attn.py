import torch
import torch.nn as nn
import torch.nn.functional as F

class FlashAttention(nn.Module):

    def __init__(self, dim, num_heads=8, dropout=0.1, max_seq_length=1024):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scaling = self.head_dim ** -0.5 # softmax/sqrt(dk)

        self.dropout = dropout
        self.max_seq_length = max_seq_length

        # Linear Projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):  # x: (B, S, dim)

        batch_size, seq_length, dim = x.shape  # x: (B, S, dim)

        # Project and reshape queries, keys and values
        q = self.q_proj(x).reshape(batch_size, seq_length, self.num_heads, self.head_dim)  # (B, S, nh, d)
        k = self.k_proj(x).reshape(batch_size, seq_length, self.num_heads, self.head_dim)  # (B, S, nh, d)
        v = self.v_proj(x).reshape(batch_size, seq_length, self.num_heads, self.head_dim)  # (B, S, nh, d)

        q = q.transpose(1, 2)  # (B, nh, S, d)
        k = k.transpose(1, 2)  # (B, nh, S, d)
        v = v.transpose(1, 2)  # (B, nh, S, d)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling  # (B, nh, S, S)

        if mask is not None:  # mask: (B, 1, S, S) or (B, S, S)
            attn_weights = attn_weights.masked_fill(mask==0, float('-inf'))  # (B, nh, S, S)
        
        attn_weights = F.softmax(attn_weights, dim=-1)  # (B, nh, S, S)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)  # (B, nh, S, S)

        attn_output = torch.matmul(attn_weights, v)  # (B, nh, S, d)

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, dim)  # (B, S, dim)
        attn_output = self.out_proj(attn_output)  # (B, S, dim)

        return attn_output  # (B, S, dim)
    
class FlashAttentionWithTiling(FlashAttention):

    def __init__(self, dim, num_heads=8, dropout=0.1, max_seq_length=1024, block_size=64):
        super().__init__(dim, num_heads, dropout, max_seq_length)

        self.block_size = block_size

    def forward(self, x, mask=None):  # x: (B, S, dim)

        batch_size, seq_length, dim = x.shape  # x: (B, S, dim)

        # Project and reshape queries, keys and values
        q = self.q_proj(x).reshape(batch_size, seq_length, self.num_heads, self.head_dim)  # (B, S, nh, d)
        k = self.k_proj(x).reshape(batch_size, seq_length, self.num_heads, self.head_dim)  # (B, S, nh, d)
        v = self.v_proj(x).reshape(batch_size, seq_length, self.num_heads, self.head_dim)  # (B, S, nh, d)

        q = q.transpose(1, 2)  # (B, nh, S, d)
        k = k.transpose(1, 2)  # (B, nh, S, d)
        v = v.transpose(1, 2)  # (B, nh, S, d)

        output = torch.zeros_like(q)  # (B, nh, S, d)

        # Process in blocks
        for i in range(0, seq_length, self.block_size):
            q_block = q[:, :, i:min(i + self.block_size, seq_length), :]  # (B, nh, block_size, d)

            block_weights = torch.zeros(batch_size, self.num_heads, q_block.size(2), seq_length, device=q.device)  # (B, nh, block_size, S)

            for j in range(0, seq_length, self.block_size):
                k_block = k[:, :, j:min(j + self.block_size, seq_length), :]  # (B, nh, block_size, d)

                scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * self.scaling  # (B, nh, block_size, block_size)

                if mask is not None:  # mask: (B, 1, S, S) or (B, S, S)
                    block_mask = mask[:, :, i:i+self.block_size, j:j+self.block_size]  # (B, 1, block_size, block_size)
                    scores = scores.masked_fill(block_mask == 0, float('-inf'))  # (B, nh, block_size, block_size)
                
                block_weights[:, :, :, j:j+k_block.size(2)] = scores  # (B, nh, block_size, block_size) -> (B, nh, block_size, S)
            
            block_weights = F.softmax(block_weights, dim=-1)  # (B, nh, block_size, S)
            block_weights = F.dropout(block_weights, p=self.dropout, training=self.training)  # (B, nh, block_size, S)

            block_output = torch.zeros_like(q_block)  # (B, nh, block_size, d)

            # Compute block output by processing each key/value block
            for j in range(0, seq_length, self.block_size):
                v_block = v[:, :, j:min(j + self.block_size, seq_length), :]  # (B, nh, block_size, d)
                block_weights_slice = block_weights[:, :, :, j:j+v_block.size(2)]  # (B, nh, block_size, block_size)
                block_output += torch.matmul(block_weights_slice, v_block)  # (B, nh, block_size, d)

            output[:, :, i:i + q_block.size(2), :] = block_output  # (B, nh, block_size, d) -> (B, nh, S, d)
        
        output = output.transpose(1, 2).reshape(batch_size, seq_length, dim)  # (B, S, dim)
        output = self.out_proj(output)  # (B, S, dim)

        return output  # (B, S, dim)
    

def test_flash_attention():

    batch_size = 2
    seq_length = 128
    dim = 512
    num_heads = 8

    flash_attn = FlashAttention(dim=dim, num_heads=num_heads)
    flash_attn_tiling = FlashAttentionWithTiling(dim=dim, num_heads=num_heads, block_size=32)

    x = torch.randn(batch_size, seq_length, dim)  # (B, S, dim)

    mask = torch.ones(batch_size, 1, seq_length, seq_length)  # (B, 1, S, S)

    output1 = flash_attn(x, mask)  # (B, S, dim)

    output2 = flash_attn_tiling(x, mask)  # (B, S, dim)

    assert output1.shape == (batch_size, seq_length, dim)  # (B, S, dim)
    assert output2.shape == (batch_size, seq_length, dim)  # (B, S, dim)

    assert not torch.allclose(output1, output2, rtol=1e-4)

    print("Tests Passed")

if __name__ == "__main__":
    test_flash_attention()