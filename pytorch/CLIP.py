import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchvision import models, transforms
import numpy as np

class CLIP(nn.Module):
    def __init__(self, 
                 embed_size, 
                 transformer_width, 
                 transformer_heads, 
                 transformer_layers,
                 context_length
                ):
        super(CLIP, self).__init__()
        self.embed_size = embed_size
        
        # Text encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_width, nhead=transformer_heads, activation='gelu')
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, transformer_width))
        
        # Vision encoder (ViT)
        self.image_encoder = models.vit_b_16(pretrained=False)
        
        # Projection layers
        self.text_projection = nn.Linear(transformer_width, embed_size)
        self.vision_projection = nn.Linear(self.image_encoder.heads.head.out_features, embed_size)
        
        # Image transformations
        self.image_transformation = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def encode_image(self, x):
        x = self.image_transformation(x)
        image_features = self.image_encoder(x)
        return self.vision_projection(image_features)
    
    def encode_text(self, x):
        x = x + self.positional_embedding 
        text_features = self.text_encoder(x)
        text_features = text_features[:, -1, :]
        return self.text_projection(text_features)
    
    def forward(self, text, image):
        text_features = self.encode_text(text)
        image_features = self.encode_image(image)
        
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        logits = text_features @ image_features.t() * self.logit_scale.exp()

        losses = self.loss(logits)
        return logits, losses

    def loss(self, logit):
        batch_size = logit.size(0)
        labels = torch.arange(batch_size)

        loss_i = F.cross_entropy(logit, labels)
        loss_t = F.cross_entropy(logit.t(), labels)
        
        # Total loss
        total_loss = (loss_i + loss_t) / 2
        
        return total_loss