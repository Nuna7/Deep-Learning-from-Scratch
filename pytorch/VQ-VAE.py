import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantization(nn.Module):
    def __init__(self, dim, d_model, decay=0.99, training=True):
        super(VectorQuantization, self).__init__()
        self.dim = dim
        self.d_model = d_model
        self.decay = decay
        self.eps = 1e-5
        self.training = training

        embedding = torch.randn(dim, d_model)

        # No gradient updating onto this 
        self.register_buffer("embedding", embedding)
        self.register_buffer("N", torch.zeros(d_model))
        self.register_buffer("M", embedding.clone())

    def code_book(self, id):
        return F.embedding(id, self.embedding.transpose(0,1))

    def forward(self, x):
        flatten = x.reshape(-1, self.dim)

        # || a - b || = ||a|| - 2ab + ||b||
        distance = flatten.pow(2).sum(1, keepdim=True) - 2*flatten @ self.embedding + self.embedding.pow(2).sum(0, keepdim=True)

        _, max_index = (-distance).max(1)
        onehot = F.one_hot(max_index, self.d_model)
        indices = max_index.view(*x.shape[:-1])
        quantize = self.code_book(indices)
        
        if self.training:
            ni = onehot.sum(0)
            self.N = (self.N * self.decay) + ((1 - self.decay) * ni)

            eofx = flatten.transpose(0, 1) @ onehot.float()
            self.M = self.M * self.decay + ((1 - self.decay) * eofx)

            n = self.N.sum()
            N = ((self.N + self.eps) / (n + self.d_model * self.eps) * n)

            e = self.M / N
            self.embedding.data.copy_(e)

        quant_loss = (x.detach() - quantize).pow(2).mean()
        commitment_loss = (quantize.detach() - x).pow(2).mean()
        
        # This allows gradients to flow through x
        quantize = x + (quantize - x).detach()

        return quantize, quant_loss, indices, commitment_loss


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1)
        )

    def forward(self, x):
        out = self.conv(x)
        out += x
        return out

class Encoder(nn.Module):
    def __init__(self, in_channel, channel, num_blocks, num_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1)
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1)
            ]

        for i in range(num_blocks):
            blocks.append(ResidualBlock(channel, num_channel))

        blocks.append(nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, channel, num_block, num_channel, stride):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(num_block):
            blocks.append(ResidualBlock(channel, num_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

class VQVAE(nn.Module):
    """
    1. In first training step, the given loss is optimized to tune the encoder, decoder and codebook
    2. In second training step, each x pass through encoder top and bottom, are both quantized,
       then the top quantized is trained on PixelCNN while the bottom quantized is trained on CondPixelCNN
    3. For sampling, we get sample from both PixelCNN, concat and pass through decoder.
    """
    def __init__(self, channel, num_bocks, num_channel, dim=64, d_model=512, decay=0.99, beta=0.9):
        super().__init__()

        self.beta = beta
        
        self.encoder_bottom = Encoder(3, channel, num_bocks, num_channel, stride=4)
        self.encoder_top = Encoder(channel, channel, num_bocks, num_channel, stride=2)

        self.conv_top = nn.Conv2d(channel, dim, 1)
        self.quantize_top = VectorQuantization(dim, d_model, decay, training=True)

        self.decoder_top = Decoder(dim, dim, channel, num_bocks, num_channel, stride=2)

        self.conv_bottom = nn.Conv2d(dim + channel, dim, 1)
        self.quantize_bottom = VectorQuantization(dim, d_model, decay, training=True)

        self.upsampling = nn.ConvTranspose2d(dim, dim, 4, stride=2, padding=1)

        self.decoder = Decoder(dim + dim, 3, channel, num_bocks, num_channel, stride=4)

    def encode(self, x):
        encoded_bottom = self.encoder_bottom(x)
        encoded_top = self.encoder_top(encoded_bottom)

        quant_top = self.conv_top(encoded_top).permute(0, 2, 3, 1)
        quant_t, loss_t, id_t, commitment_loss_1 = self.quantize_top(quant_top)
        quant_t = quant_t.permute(0, 3, 1, 2)
        loss_t = loss_t.unsqueeze(0)

        decoded_top = self.decoder_top(quant_t)
        encoded_bottom = torch.cat([decoded_top, encoded_bottom], 1)

        quant_b = self.conv_bottom(encoded_bottom).permute(0, 2, 3, 1)
        quant_b, lose_b, id_b, commitment_loss_2 = self.quantize_bottom(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        loss_b = lose_b.unsqueeze(0)

        commitment_loss = commitment_loss_1 + commitment_loss_2

        return quant_t, quant_b, loss_t + loss_b, id_t, id_b, commitment_loss

    def decode(self, quant_t, quant_b):
        upsample = self.upsampling(quant_t)
        quant = torch.cat([upsample, quant_b], 1)
        decoded = self.decoder(quant)

        return decoded

    def forward(self, x):
        quant_t, quant_b, quant_loss, id_t, id_b, commitment_loss = self.encode(x)
        dec = self.decode(quant_t, quant_b)
        total_loss = self.loss(quant_loss, x, dec, commitment_loss)
        return dec, total_loss

    def loss(self, quant_loss, x, dec, commitment_loss):
        # Reconstruction loss
        recon_loss = F.mse_loss(dec, x)

        return recon_loss + quant_loss + self.beta * commitment_loss