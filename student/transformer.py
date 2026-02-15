import torch
import torch.nn as nn
from student.custom_modules import RMSNorm,SwiGLU,Embedding,Linear
from student.multihead_attention import MultiHeadSelfAttentionWithRoPE

class TransformerBlock(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,rope_theta,max_seq_len,
                 device=None,dtype=None):
        super().__init__()

        self.norm_1  = RMSNorm(d_model, device=device, dtype=dtype)

        self.attn = MultiHeadSelfAttentionWithRoPE(d_model,num_heads,rope_theta,max_seq_len,device=device,dtype=dtype)

        self.norm_2 = RMSNorm(d_model, device=device, dtype=dtype)

        self.ffn = SwiGLU(d_model,d_ff,device=device,dtype=dtype)
    
    def forward(self,x,token_positions=None):
        # sublayer 1 attn
        x_norm = self.norm_1(x)
        attn = self.attn(x_norm,token_positions)
        x = x + attn

        # sublayer 2 ffn
        x_norm = self.norm_2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out

        return x
    

class TransformerLM(nn.Module):
    def __init__(self,vocab_size, context_length,num_layers,d_model,num_heads,d_ff,rope_theta,
                 device=None,dtype=None):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        # token embedding
        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        # transformer block
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                rope_theta=rope_theta,
                max_seq_len=context_length,
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])

        # final norm
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)

        # output projection
        self.output_projection_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, input_tokens, token_positions=None):
        # create embeddings
        x = self.token_embedding(input_tokens)

        # transformer blicks
        for block in self.blocks:
            x = block(x, token_positions)

        # nrom blick
        x = self.final_norm(x)

        # vocab projection
        logits = self.output_projection_head(x)

        return logits



