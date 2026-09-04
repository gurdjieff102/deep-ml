from tinygrad import Tensor, nn

class TransformerBlock:
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        assert d_model % num_heads == 0

        self.norm1 = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)

        self.norm2 = nn.LayerNorm(d_model)

        self.mlp_fc1 = nn.Linear(d_model, d_ff)
        self.mlp_fc2 = nn.Linear(d_ff, d_model)

        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.dropout_p = dropout

    def __call__(self, x):
        # ===== Attention =====
        residual = x

        x = self.norm1(x)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        B, T, _ = q.shape

        # (B, T, d_model)
        # -> (B, H, T, d_head)
        q = q.reshape(B, T, self.num_heads, self.d_head)
        k = k.reshape(B, T, self.num_heads, self.d_head)
        v = v.reshape(B, T, self.num_heads, self.d_head)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = q.scaled_dot_product_attention(k, v)

        # (B, H, T, d_head)
        # -> (B, T, d_model)
        attn = attn.transpose(1, 2)
        attn = attn.reshape(B, T, self.num_heads * self.d_head)

        attn = self.attn_out(attn)

        x = residual + attn.dropout(self.dropout_p)

        # ===== MLP =====
        residual = x

        x = self.norm2(x)

        x = self.mlp_fc1(x)
        x = x.gelu()
        x = self.mlp_fc2(x)

        x = residual + x.dropout(self.dropout_p)

        return x