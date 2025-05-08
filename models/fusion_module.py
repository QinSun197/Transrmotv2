import torch
import torch.nn as nn
from typing import Optional

class MultiVisionLanguageFusion(nn.Module):
    def __init__(self, d_model, nhead, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionBlock(d_model, nhead, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, 
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None):
        """
        tgt: image features, shape [num_queries, batch_size, d_model]
        memory: text query features, shape [num_patches, batch_size, d_model]
        """
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, memory_key_padding_mask, pos, query_pos)
        
        output = self.norm(output)
        return output

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, hidden_dim=d_model*4, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None):
        # Cross Attention
        q = self.with_pos_embed(tgt, query_pos)
        k = self.with_pos_embed(memory, pos)
        v = memory
        tgt2, _ = self.cross_attn(query=q, key=k, value=v, key_padding_mask=memory_key_padding_mask)
        
        # Residual + Norm
        tgt = tgt + tgt2
        tgt = self.norm1(tgt)

        # FFN
        tgt2 = self.ffn(tgt)

        # Residual + Norm
        tgt = tgt + tgt2
        tgt = self.norm2(tgt)

        return tgt

class FFN(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


# Example usage
if __name__ == "__main__":
    batch_size = 1
    num_patches = 64   # 比如图像特征展平后的 patch 数
    num_words = 10     # 输入句子最多20个单词
    d_model = 256      # 特征维度
    nhead = 8          # 多头注意力头数

    # 图像特征 memory（一般来自 backbone 或 encoder）
    memory = torch.randn(num_patches, batch_size, d_model)  # [num_patches, B, C]
    print(f"memory shape: {memory.shape}")

    # 文本特征 tgt（比如每个单词的特征向量）
    text_features = torch.randn(num_words, batch_size, d_model)  # [num_words, B, C]
    print(f"text_features shape: {text_features.shape}")

    # 文本 mask（1表示padding，0表示有效）
    text_mask = torch.zeros(batch_size, num_words).bool()  # [B, num_words]
    print(f"text_mask: {text_mask}")

    # 修改：文本特征位置编码调整为与 memory (图像特征) 一致的形状
    text_pos = torch.randn(num_patches, batch_size, d_model)  # [num_patches, batch_size, d_model]
    print(f"text_pos shape: {text_pos.shape}")

    # 你想对文本特征做图文融合（图像作为memory，文本作为query）
    fusion_module = MultiVisionLanguageFusion(d_model=d_model, nhead=nhead, num_layers=2)

    # 前向
    fused_img_features = fusion_module(
        tgt=memory,                # 图像特征作为Query
        memory=text_features,      # 文本特征作为Key-Value
        memory_key_padding_mask=None,  # 可选，如果memory有mask的话可以加
        pos=None,                          # 可选，这里memory没有加位置信息就传None
        query_pos=text_pos         # Query加上位置编码，形状与 memory 匹配
    )

    print(f"fused_image_features shape: {fused_img_features.shape}")
