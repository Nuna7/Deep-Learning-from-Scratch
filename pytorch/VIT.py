import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from typing import Optional
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, input_dim, qk_norm: bool = False ,qkv_bias: bool = False):
        
        super(MultiHeadAttention, self).__init__()
        assert input_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        self.qkv_layer = nn.Linear(input_dim, 3 * input_dim, qkv_bias)
        self.softmax_layer = nn.Softmax(dim=-1)

        nn.init.xavier_uniform_(self.qkv_layer.weight)

        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
    
    def forward(self, x):
        qkv = self.qkv_layer(x)
        qkv = qkv.view(x.size(0), self.num_heads, x.size(1), 3 * self.head_dim)  # [batch_size, num_heads, num_patches, 3*head_dim]
        q, k, v = qkv.chunk(3, dim=-1)
        q = self.q_norm(q)
        k = self.k_norm(k)
        multi_head_attention = torch.matmul(q, k.transpose(-2, -1)) / (q.size(3) ** 0.5)
    
        output = torch.matmul(self.softmax_layer(multi_head_attention), v).reshape(v.size(0), v.size(2), -1)
        return output

class PositionalEmbedding(nn.Module):
    def __init__(self, patch_shape, d_model):
        super(PositionalEmbedding, self).__init__()
        H, W, P, C = patch_shape

        self.embedding_projection = nn.Conv2d(in_channels = C, 
                       out_channels = d_model, kernel_size = P, stride = P)#P**2 * C
        
        self.positional_embeddings = nn.Parameter(torch.rand((int(H/P) ** 2) + 1, P**2 * C))
        nn.init.xavier_uniform_(self.embedding_projection.weight)
        
        self.classification_token = nn.Parameter(torch.rand((1,  d_model)))#P**2 * C
        nn.init.xavier_uniform_(self.classification_token)
        
        self.flatten_layer = nn.Flatten(start_dim=1, end_dim=2)
    

    def forward(self, x):
        image = x.permute((0, 3, 1, 2))
        image = self.embedding_projection(image)
        image = image.permute((0, 2, 3, 1))
        image = self.flatten_layer(image)
        
        token = self.classification_token.expand(image.size(0), self.classification_token.size(0), self.classification_token.size(1))
        
        image = torch.cat([token, image], dim=1)
        return image

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, patch_shape, num_heads, init_values : Optional[float] = None,mlp_ratio : int = 4, mlp_dropout : float = 0):
        H, W, P, C = patch_shape
        self.num_heads = num_heads
        super(TransformerEncoder, self).__init__()

        self.norm_1 = nn.LayerNorm([int(H/P)**2 + 1])
        self.norm_2 = nn.LayerNorm([int(H/P)**2 + 1])

        self.attention = MultiHeadAttention(num_heads, d_model)

        self.ls1 = LayerScale(d_model, init_values=init_values) if init_values else nn.Identity()
        self.ls2 = LayerScale(d_model, init_values=init_values) if init_values else nn.Identity()

        self.mlp = nn.Sequential(
            nn.Linear(in_features = d_model, out_features = mlp_ratio * d_model),
            nn.GELU(),
            nn.Dropout(p = mlp_dropout),
            nn.Linear(in_features = mlp_ratio * d_model, out_features = d_model),
            nn.Dropout(p = mlp_dropout)
        )

    def forward(self, x):
        skip_1 = x.clone()
        x_1 = self.norm_1(x.reshape(x.size(0), x.size(2), x.size(1))).reshape(x.size(0), x.size(1), x.size(2))
        output_1 = self.ls1(self.attention.forward(x_1))
        out_1 = output_1 + skip_1

        skip_2 = out_1.clone()
        norm_2 = self.norm_2(out_1.reshape(out_1.size(0), out_1.size(2), out_1.size(1))).reshape(out_1.size(0), out_1.size(1), out_1.size(2))
        mlp_output = self.ls2(self.mlp(norm_2))
        out_2 = mlp_output + skip_2

        return out_2


"""
The LayerScale class implements layer scaling.

Layer scaling is a technique used to scale the activations of a layer by a learnable parameter (gamma).
"""
class LayerScale(nn.Module):
    def __init__(self, dim : int, init_values : float = 1e-5, inplace : bool = False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
        
class VIT(nn.Module):
    def __init__(self, d_model, data, y, num_heads, P, num_class,num_encoder=1, init_value=1e-5, mlp_ratio=4, dropout=0):
        super(VIT, self).__init__()
        self.P = P
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_class = num_class
        self.y = y
        
        self.data = self.process_data(data)
        patch_shape = [self.H, self.W, P, self.C]
         
        self.positional_embedding = PositionalEmbedding(patch_shape, self.d_model)
        
        self.all_encoder = nn.ModuleList([TransformerEncoder(self.d_model, patch_shape, self.num_heads, init_value,mlp_ratio,dropout)
                                     for _ in range(num_encoder)])

        self.classification_head = nn.Linear(self.d_model, self.num_class)

        # VIT also used hybrid architecture where they take the feature map output by stages 4 of resnet 50 or stages 3 and add another layer
    
    def process_data(self, x):
        all_data = []
        for instance in x:
            image = torch.Tensor(list(map(int, instance.split()))).view(48, 48, 1)
            H, W, C = image.shape
            all_data.append(image)
            
        data = torch.stack(all_data)
        self.H, self.W, self.C = H, W, C
        return data

    def forward(self):
        x = self.positional_embedding(self.data)
        for i in range(len(self.all_encoder)):
            x = self.all_encoder[i](x)

        x = x[:, 0, :]
        logits = self.classification_head(x)
        return logits

    def calculate_loss(self, logits):
        targets = self.y.view(-1).long()
        return F.cross_entropy(logits, targets)

vit_model = VIT(d_model=256, data=train_x, y=train_y, num_heads=8, P=16, num_class=8, num_encoder=6)

optimizer = optim.Adam(vit_model.parameters(), lr=0.001) 

num_epochs = 600

for epoch in range(num_epochs):
    vit_model.train() 

    logits = vit_model()

    loss = F.cross_entropy(logits, vit_model.y.view(-1).long())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() 

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')