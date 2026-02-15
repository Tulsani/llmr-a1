import torch
import torch.nn as nn
from student.custom_modules import scaled_dot_product_attention,RotatryPositionalEmbedding,Linear  

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,d_model,num_heads,
                 use_causal_mask=True,
                 device=None,dtype=None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.use_causal_mask = use_causal_mask

        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # linear projections # we can directly have d_model 
        self.W_q = Linear(d_model,num_heads * self.d_k,device=device,dtype=dtype)
        self.W_k = Linear(d_model,num_heads * self.d_k,device=device,dtype=dtype)
        self.W_v = Linear(d_model,num_heads * self.d_v,device=device,dtype=dtype)

        self.W_o = Linear(num_heads * self.d_v,d_model,device=device,dtype=dtype)

    
    def forward(self,x,mask=None):
        #unpack x
        x_batch, x_seq_len, d_model = x.shape

        # linear proj
        Q = self.W_q(x)  
        K = self.W_k(x)
        V = self.W_v(x)

        # reshape for heads
        Q = Q.view(x_batch, x_seq_len, self.num_heads, self.d_k).transpose(1,2)
        K = K.view(x_batch, x_seq_len, self.num_heads, self.d_k).transpose(1,2)
        V = V.view(x_batch, x_seq_len, self.num_heads, self.d_v).transpose(1,2)

        # causal mask
        if mask is None and self.use_causal_mask:
            mask = torch.tril(torch.ones(x_seq_len, x_seq_len, device=x.device, dtype=torch.bool))
        
        # applying attention
        attn_output = scaled_dot_product_attention(Q, K, V, mask)

        # reshape back
        attn_output = attn_output.transpose(1,2).contiguous()
        attn_output = attn_output.view(x_batch, x_seq_len, self.num_heads * self.d_v)
        
        # output
        output = self.W_o(attn_output)

        return output

class MultiHeadSelfAttentionWithRoPE(nn.Module):
    def __init__(self,d_model,num_heads,theta,max_seq_len,
                 use_causal_mask=True,
                 device=None,dtype=None):
          
          super().__init__()

          self.d_model = d_model
          self.num_heads = num_heads

          self.d_k = d_model // num_heads # k and q
          self.d_v = d_model // num_heads # v

          self.W_q = Linear(d_model,num_heads * self.d_k,device=device,dtype=dtype)
          self.W_k = Linear(d_model,num_heads * self.d_k,device=device,dtype=dtype)
          self.W_v = Linear(d_model,num_heads * self.d_v,device=device,dtype=dtype)

          self.W_o = Linear(num_heads * self.d_v,d_model,device=device,dtype=dtype)

          # rope
          self.rope = RotatryPositionalEmbedding(theta=theta,
                                                 d_k=self.d_k,
                                                 max_seq_len=max_seq_len,
                                                 device=device,
                                                 dtype=dtype)
          self.use_causal_mask = use_causal_mask

    def forward(self,x,token_positions,mask=None):
        x_batch, x_seq_len, d_model = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # reshape
        Q = Q.view(x_batch, x_seq_len, self.num_heads, self.d_k).transpose(1,2)
        K = K.view(x_batch, x_seq_len, self.num_heads, self.d_k).transpose(1,2)
        V = V.view(x_batch, x_seq_len, self.num_heads, self.d_v).transpose(1,2)

        #token_postion
        if token_positions is None:
            token_positions = torch.arange(x_seq_len, device=x.device).unsqueeze(0).expand(x_batch, x_seq_len)

        # apply rope
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)

        # apply causal mask

        if mask is None and self.use_causal_mask:
            mask = torch.tril(torch.ones(x_seq_len, x_seq_len, device=x.device, dtype=torch.bool))
        
        attn_output =scaled_dot_product_attention(Q,K,V,mask)

        attn_output = attn_output.transpose(1,2).contiguous()
        attn_output = attn_output.view(x_batch, x_seq_len, self.num_heads * self.d_v)

        output = self.W_o(attn_output)

        return output


        


          