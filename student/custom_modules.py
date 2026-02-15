import torch.nn as nn
import torch
import math

# Linear
class Linear(nn.Module):
    """
    We need to y = Wx
    """
    def __init__(self, in_features, out_features,device=None,dtype=None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # seting up weights matrix
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))

        # std
        std = math.sqrt(2.0 / (self.in_features + self.out_features))

        #initalizing
        nn.init.trunc_normal_(self.weight, std=std,mean=0.0,a= -3*std, b=3*std)
    
    def forward(self, x):
        return x @ self.weight.T
    
#Embedding
class Embedding(nn.Module):
    def __init__(self,num_embeddings,embedding_dim,device=None,dtype=None):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )

        std = math.sqrt(2.0 / (self.num_embeddings + self.embedding_dim))
        nn.init.trunc_normal_(self.weight, std=1, mean=0.0, a=-3, b=3)

    
    def forward(self, token_ids):
        return self.weight[token_ids]
    
#RMS Norm
class RMSNorm(nn.Module):
    def __init__(self,d_model,eps=1e-5,device=None,dtype=None):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        self.weight = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        nn.init.ones_(self.weight)

    def forward(self, x):

        input_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        x_scaled = x_normed * self.weight

        return x_scaled.to(input_dtype)

# SwiGLU
class SwiGLU(nn.Module):
    def __init__(self,d_model,d_ff,device=None,dtype=None):
        super().__init__()

        self.d_model = d_model
        if d_ff is None:
            d_ff = int(8 * d_model /3)
            # to nearest multiple of 64
            d_ff = ((d_ff + 63) // 64) * 64
        
        self.d_ff = d_ff

        # linear transformations
        #silu path
        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        # output path
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        # gate path
        self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x):
        #silu path
        silu_path = self.W1(x)
        silu_activation = silu_path * torch.sigmoid(silu_path)

        #gate path
        gate_path = self.W3(x)

        # elementwise product
        gated_silu = silu_activation * gate_path

        #output
        output = self.W2(gated_silu)
        return output

# Silu
class SiLU(nn.Module):
    def __init__(self, d_model=None, d_ff=None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        if d_model is not None:
            if d_ff is None:
                d_ff = 4 * d_model
            self.d_ff = d_ff
            self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)
            self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        else:
            self.W1 = None
            self.W2 = None

    def forward(self, x):
        if self.W1 is not None:
            h = self.W1(x)
            h = h * torch.sigmoid(h)
            return self.W2(h)
        else:
            # Just the activation
            return x * torch.sigmoid(x)

class RotatryPositionalEmbedding(nn.Module):
    def __init__(self,theta,d_k,max_seq_len,device=None,dtype=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        position = torch.arange(max_seq_len, device=device, dtype=torch.float32).unsqueeze(1)  # (max_seq_len, 1)

        half  = d_k // 2

        dim_indices = torch.arange(0, half, device=device, dtype=torch.float32)

        #angles
        inv_freq = 1.0 / (theta ** (2 * dim_indices / d_k)) 
        freq = position * inv_freq.unsqueeze(0) 

        cos_vals =torch.cos(freq)  
        sin_vals = torch.sin(freq)

        # register to buffers
        self.register_buffer("cos_vals", cos_vals,persistent=False)
        self.register_buffer("sin_vals", sin_vals,persistent=False)

    def forward(self, x, pos_ids):
        cos = self.cos_vals[pos_ids] 
        sin = self.sin_vals[pos_ids]

        # split into pairs
        x_reshaped = x.reshape(*x.shape[:-1], -1, 2)  # (..., d_k/2, 2)

        # extract pairs
        x1 = x_reshaped[..., 0]  
        x2 = x_reshaped[..., 1]

        # apply rotation
        x1_rotated = x1 * cos - x2 * sin
        x2_rotated = x1 * sin + x2 * cos

        # stacking back together
        x_rotated = torch.stack((x1_rotated, x2_rotated), dim=-1)  # (..., d_k/2, 2)
        
        #reshape back to original
        output = x_rotated.reshape(*x.shape)

        return output


def softmax(x,dim):
    # softmax
    #noramlization
    x_max = torch.max(x,dim=dim,keepdim=True).values
    x_shifted = x - x_max

    # exp
    exp_x = torch.exp(x_shifted)

    # sum along dim
    exp_sum = torch.sum(exp_x,dim=dim,keepdim=True)

    return exp_x / exp_sum

def scaled_dot_product_attention(Q,K,V,mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2,-1) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == False, float("-inf"))

    attn_weights = softmax(scores, dim=-1)
    output = attn_weights @ V
    return output

    

    
