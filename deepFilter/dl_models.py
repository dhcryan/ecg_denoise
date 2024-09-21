# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # Conv1DTranspose in PyTorch
# class Conv1DTranspose(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=1, activation='relu'):
#         super(Conv1DTranspose, self).__init__()
#         self.conv_transpose = nn.ConvTranspose2d(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=(kernel_size, 1),
#             stride=(stride, 1),
#             padding=(padding, 0),
#         )
#         self.activation = getattr(F, activation)

#     def forward(self, x):
#         x = x.unsqueeze(2)  # Adding the extra dimension (like Lambda in TensorFlow)
#         x = self.conv_transpose(x)
#         x = x.squeeze(2)  # Removing the added dimension
#         return self.activation(x)

# # Module replacements from Keras to PyTorch
# class LFilterModule(nn.Module):
#     def __init__(self, layers):
#         super(LFilterModule, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=layers, out_channels=layers // 4, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=layers, out_channels=layers // 4, kernel_size=5, padding=1)
#         self.conv3 = nn.Conv1d(in_channels=layers, out_channels=layers // 4, kernel_size=9, padding=1)
#         self.conv4 = nn.Conv1d(in_channels=layers, out_channels=layers // 4, kernel_size=15, padding=1)

#     def forward(self, x):
#         lb0 = self.conv1(x)
#         lb1 = self.conv2(x)
#         lb2 = self.conv3(x)
#         lb3 = self.conv4(x)
#         return torch.cat([lb0, lb1, lb2, lb3], dim=1)  # Concatenation along the channel dimension

# class NLFilterModule(nn.Module):
#     def __init__(self, layers):
#         super(NLFilterModule, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=layers, out_channels=layers // 4, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=layers, out_channels=layers // 4, kernel_size=5, padding=1)
#         self.conv3 = nn.Conv1d(in_channels=layers, out_channels=layers // 4, kernel_size=9, padding=1)
#         self.conv4 = nn.Conv1d(in_channels=layers, out_channels=layers // 4, kernel_size=15, padding=1)

#     def forward(self, x):
#         nlb0 = F.relu(self.conv1(x))
#         nlb1 = F.relu(self.conv2(x))
#         nlb2 = F.relu(self.conv3(x))
#         nlb3 = F.relu(self.conv4(x))
#         return torch.cat([nlb0, nlb1, nlb2, nlb3], dim=1)

# class LANLFilterModule(nn.Module):
#     def __init__(self, layers):
#         super(LANLFilterModule, self).__init__()
#         # Linear blocks
#         self.conv1 = nn.Conv1d(in_channels=layers, out_channels=layers // 8, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=layers, out_channels=layers // 8, kernel_size=5, padding=1)
#         self.conv3 = nn.Conv1d(in_channels=layers, out_channels=layers // 8, kernel_size=9, padding=1)
#         self.conv4 = nn.Conv1d(in_channels=layers, out_channels=layers // 8, kernel_size=15, padding=1)
#         # Non-linear blocks
#         self.nconv1 = nn.Conv1d(in_channels=layers, out_channels=layers // 8, kernel_size=3, padding=1)
#         self.nconv2 = nn.Conv1d(in_channels=layers, out_channels=layers // 8, kernel_size=5, padding=1)
#         self.nconv3 = nn.Conv1d(in_channels=layers, out_channels=layers // 8, kernel_size=9, padding=1)
#         self.nconv4 = nn.Conv1d(in_channels=layers, out_channels=layers // 8, kernel_size=15, padding=1)

#     def forward(self, x):
#         lb0 = self.conv1(x)
#         lb1 = self.conv2(x)
#         lb2 = self.conv3(x)
#         lb3 = self.conv4(x)

#         nlb0 = F.relu(self.nconv1(x))
#         nlb1 = F.relu(self.nconv2(x))
#         nlb2 = F.relu(self.nconv3(x))
#         nlb3 = F.relu(self.nconv4(x))

#         return torch.cat([lb0, lb1, lb2, lb3, nlb0, nlb1, nlb2, nlb3], dim=1)

# class LANLFilterModuleDilated(nn.Module):
#     def __init__(self, layers):
#         super(LANLFilterModuleDilated, self).__init__()
#         # Linear blocks with dilation
#         self.conv1 = nn.Conv1d(in_channels=layers, out_channels=layers // 6, kernel_size=5, dilation=3, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=layers, out_channels=layers // 6, kernel_size=9, dilation=3, padding=1)
#         self.conv3 = nn.Conv1d(in_channels=layers, out_channels=layers // 6, kernel_size=15, dilation=3, padding=1)
#         # Non-linear blocks with dilation
#         self.nconv1 = nn.Conv1d(in_channels=layers, out_channels=layers // 6, kernel_size=5, dilation=3, padding=1)
#         self.nconv2 = nn.Conv1d(in_channels=layers, out_channels=layers // 6, kernel_size=9, dilation=3, padding=1)
#         self.nconv3 = nn.Conv1d(in_channels=layers, out_channels=layers // 6, kernel_size=15, dilation=3, padding=1)

#     def forward(self, x):
#         lb1 = self.conv1(x)
#         lb2 = self.conv2(x)
#         lb3 = self.conv3(x)

#         nlb1 = F.relu(self.nconv1(x))
#         nlb2 = F.relu(self.nconv2(x))
#         nlb3 = F.relu(self.nconv3(x))

#         return torch.cat([lb1, lb2, lb3, nlb1, nlb2, nlb3], dim=1)


# ###### MODELS #######
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class DeepFilterVanillaLinear(nn.Module):
#     def __init__(self):
#         super(DeepFilterVanillaLinear, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=9, padding=4)
#         self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=9, padding=4)
#         self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=9, padding=4)
#         self.conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=9, padding=4)
#         self.conv5 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=9, padding=4)
#         self.conv6 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=9, padding=4)
#         self.conv7 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=9, padding=4)

#     def forward(self, x):
#         x = F.linear(self.conv1(x))
#         x = F.linear(self.conv2(x))
#         x = F.linear(self.conv3(x))
#         x = F.linear(self.conv4(x))
#         x = F.linear(self.conv5(x))
#         x = F.linear(self.conv6(x))
#         x = F.linear(self.conv7(x))
#         return x


# class DeepFilterVanillaNLinear(nn.Module):
#     def __init__(self):
#         super(DeepFilterVanillaNLinear, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=9, padding=4)
#         self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=9, padding=4)
#         self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=9, padding=4)
#         self.conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=9, padding=4)
#         self.conv5 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=9, padding=4)
#         self.conv6 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=9, padding=4)
#         self.conv7 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=9, padding=4)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))
#         x = F.relu(self.conv5(x))
#         x = F.relu(self.conv6(x))
#         x = F.linear(self.conv7(x))
#         return x


# # Deep Filter using LFilterModule and NLFilterModule (defined previously)
# class DeepFilterILinear(nn.Module):
#     def __init__(self):
#         super(DeepFilterILinear, self).__init__()
#         self.layer1 = LFilterModule(layers=64)
#         self.layer2 = LFilterModule(layers=64)
#         self.layer3 = LFilterModule(layers=32)
#         self.layer4 = LFilterModule(layers=32)
#         self.layer5 = LFilterModule(layers=16)
#         self.layer6 = LFilterModule(layers=16)
#         self.conv_final = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=9, padding=4)

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.layer5(x)
#         x = self.layer6(x)
#         x = self.conv_final(x)
#         return x


# class DeepFilterINLinear(nn.Module):
#     def __init__(self):
#         super(DeepFilterINLinear, self).__init__()
#         self.layer1 = NLFilterModule(layers=64)
#         self.layer2 = NLFilterModule(layers=64)
#         self.layer3 = NLFilterModule(layers=32)
#         self.layer4 = NLFilterModule(layers=32)
#         self.layer5 = NLFilterModule(layers=16)
#         self.layer6 = NLFilterModule(layers=16)
#         self.conv_final = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=9, padding=4)

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.layer5(x)
#         x = self.layer6(x)
#         x = self.conv_final(x)
#         return x


# # Deep filter with LANLFilterModule and BatchNormalization
# class DeepFilterILANL(nn.Module):
#     def __init__(self):
#         super(DeepFilterILANL, self).__init__()
#         self.layer1 = LANLFilterModule(layers=64)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.layer2 = LANLFilterModule(layers=64)
#         self.bn2 = nn.BatchNorm1d(64)
#         self.layer3 = LANLFilterModule(layers=32)
#         self.bn3 = nn.BatchNorm1d(32)
#         self.layer4 = LANLFilterModule(layers=32)
#         self.bn4 = nn.BatchNorm1d(32)
#         self.layer5 = LANLFilterModule(layers=16)
#         self.bn5 = nn.BatchNorm1d(16)
#         self.layer6 = LANLFilterModule(layers=16)
#         self.bn6 = nn.BatchNorm1d(16)
#         self.conv_final = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=9, padding=4)

#     def forward(self, x):
#         x = self.bn1(self.layer1(x))
#         x = self.bn2(self.layer2(x))
#         x = self.bn3(self.layer3(x))
#         x = self.bn4(self.layer4(x))
#         x = self.bn5(self.layer5(x))
#         x = self.bn6(self.layer6(x))
#         x = self.conv_final(x)
#         return x


# # Deep filter with LANLFilterModule and dilated convolutions
# class DeepFilterModelILANLDilated(nn.Module):
#     def __init__(self):
#         super(DeepFilterModelILANLDilated, self).__init__()
#         self.layer1 = LANLFilterModule(layers=64)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.layer2 = LANLFilterModuleDilated(layers=64)
#         self.bn2 = nn.BatchNorm1d(64)
#         self.layer3 = LANLFilterModule(layers=32)
#         self.bn3 = nn.BatchNorm1d(32)
#         self.layer4 = LANLFilterModuleDilated(layers=32)
#         self.bn4 = nn.BatchNorm1d(32)
#         self.layer5 = LANLFilterModule(layers=16)
#         self.bn5 = nn.BatchNorm1d(16)
#         self.layer6 = LANLFilterModuleDilated(layers=16)
#         self.bn6 = nn.BatchNorm1d(16)
#         self.conv_final = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=9, padding=4)

#     def forward(self, x):
#         x = self.bn1(self.layer1(x))
#         x = self.bn2(self.layer2(x))
#         x = self.bn3(self.layer3(x))
#         x = self.bn4(self.layer4(x))
#         x = self.bn5(self.layer5(x))
#         x = self.bn6(self.layer6(x))
#         x = self.conv_final(x)
#         return x

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class FCN_DAE(nn.Module):
#     def __init__(self):
#         super(FCN_DAE, self).__init__()
        
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=40, kernel_size=16, stride=2, padding=7)
#         self.bn1 = nn.BatchNorm1d(40)
        
#         self.conv2 = nn.Conv1d(in_channels=40, out_channels=20, kernel_size=16, stride=2, padding=7)
#         self.bn2 = nn.BatchNorm1d(20)
        
#         self.conv3 = nn.Conv1d(in_channels=20, out_channels=20, kernel_size=16, stride=2, padding=7)
#         self.bn3 = nn.BatchNorm1d(20)
        
#         self.conv4 = nn.Conv1d(in_channels=20, out_channels=20, kernel_size=16, stride=2, padding=7)
#         self.bn4 = nn.BatchNorm1d(20)
        
#         self.conv5 = nn.Conv1d(in_channels=20, out_channels=40, kernel_size=16, stride=2, padding=7)
#         self.bn5 = nn.BatchNorm1d(40)
        
#         self.conv6 = nn.Conv1d(in_channels=40, out_channels=1, kernel_size=16, stride=1, padding=7)
#         self.bn6 = nn.BatchNorm1d(1)
        
#         self.deconv1 = nn.ConvTranspose1d(in_channels=1, out_channels=40, kernel_size=16, stride=2, padding=7)
#         self.bn7 = nn.BatchNorm1d(40)
        
#         self.deconv2 = nn.ConvTranspose1d(in_channels=40, out_channels=20, kernel_size=16, stride=2, padding=7)
#         self.bn8 = nn.BatchNorm1d(20)
        
#         self.deconv3 = nn.ConvTranspose1d(in_channels=20, out_channels=20, kernel_size=16, stride=2, padding=7)
#         self.bn9 = nn.BatchNorm1d(20)
        
#         self.deconv4 = nn.ConvTranspose1d(in_channels=20, out_channels=20, kernel_size=16, stride=2, padding=7)
#         self.bn10 = nn.BatchNorm1d(20)
        
#         self.deconv5 = nn.ConvTranspose1d(in_channels=20, out_channels=40, kernel_size=16, stride=2, padding=7)
#         self.bn11 = nn.BatchNorm1d(40)
        
#         self.deconv6 = nn.ConvTranspose1d(in_channels=40, out_channels=1, kernel_size=16, stride=1, padding=7)

#     def forward(self, x):
#         # Encoder
#         x = F.elu(self.bn1(self.conv1(x)))
#         x = F.elu(self.bn2(self.conv2(x)))
#         x = F.elu(self.bn3(self.conv3(x)))
#         x = F.elu(self.bn4(self.conv4(x)))
#         x = F.elu(self.bn5(self.conv5(x)))
#         x = F.elu(self.bn6(self.conv6(x)))
        
#         # Decoder
#         x = F.elu(self.bn7(self.deconv1(x)))
#         x = F.elu(self.bn8(self.deconv2(x)))
#         x = F.elu(self.bn9(self.deconv3(x)))
#         x = F.elu(self.bn10(self.deconv4(x)))
#         x = F.elu(self.bn11(self.deconv5(x)))
#         x = self.deconv6(x)  # Final output (linear activation)
        
#         return x

# class DRRN_Denoising(nn.Module):
#     def __init__(self):
#         super(DRRN_Denoising, self).__init__()
#         self.lstm = nn.LSTM(input_size=1, hidden_size=64, batch_first=True, num_layers=1)
#         self.fc1 = nn.Linear(in_features=64, out_features=64)
#         self.fc2 = nn.Linear(in_features=64, out_features=64)
#         self.fc3 = nn.Linear(in_features=64, out_features=1)

#     def forward(self, x):
#         # LSTM layer
#         x, _ = self.lstm(x)
        
#         # Fully connected layers with ReLU activations
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
        
#         # Final linear layer
#         x = self.fc3(x)
#         return x

# ##########################################################################

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.fft

# # Conv1DTranspose in PyTorch
# class Conv1DTranspose(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=1, activation='relu'):
#         super(Conv1DTranspose, self).__init__()
#         self.conv_transpose = nn.ConvTranspose2d(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=(kernel_size, 1),
#             stride=(stride, 1),
#             padding=(padding, 0),
#         )
#         self.activation = getattr(F, activation)

#     def forward(self, x):
#         x = x.unsqueeze(2)  # Adding the extra dimension (like Lambda in TensorFlow)
#         x = self.conv_transpose(x)
#         x = x.squeeze(2)  # Removing the added dimension
#         return self.activation(x)

# # Gated Noise Addition
# class AddGatedNoise(nn.Module):
#     def __init__(self):
#         super(AddGatedNoise, self).__init__()

#     def forward(self, x, training=True):
#         if training:
#             noise = torch.rand_like(x) * 2 - 1  # Random noise between -1 and 1
#             return x * (1 + noise)
#         return x

# # Transformer Encoder Layer
# class TransformerEncoder(nn.Module):
#     def __init__(self, head_size, num_heads, ff_dim, dropout=0):
#         super(TransformerEncoder, self).__init__()
#         self.attention = nn.MultiheadAttention(embed_dim=head_size * num_heads, num_heads=num_heads, dropout=dropout)
#         self.norm1 = nn.LayerNorm(head_size * num_heads)
#         self.norm2 = nn.LayerNorm(head_size * num_heads)
#         self.ff = nn.Sequential(
#             nn.Conv1d(head_size * num_heads, ff_dim, kernel_size=1),
#             nn.ReLU(),
#             nn.Conv1d(ff_dim, head_size * num_heads, kernel_size=1)
#         )
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         x2 = x.permute(1, 0, 2)  # (seq_len, batch_size, num_features)
#         attn_output, _ = self.attention(x2, x2, x2)
#         attn_output = attn_output.permute(1, 0, 2)  # Back to (batch_size, seq_len, num_features)
#         x = self.norm1(x + self.dropout(attn_output))
#         ff_output = self.ff(x.permute(0, 2, 1)).permute(0, 2, 1)
#         x = self.norm2(x + self.dropout(ff_output))
#         return x

# # Positional Encoding Layer
# class PositionalEncoding1D(nn.Module):
#     def __init__(self, channels: int):
#         super(PositionalEncoding1D, self).__init__()
#         self.channels = int((channels + 1) // 2 * 2)
#         self.inv_freq = 1 / (10000 ** (torch.arange(0, self.channels, 2).float() / self.channels))

#     def forward(self, inputs):
#         batch_size, x, _ = inputs.shape
#         device = inputs.device  # Get the device of the inputs
#         pos_x = torch.arange(x, dtype=torch.float32, device=device)  # Ensure pos_x is on the same device
#         inv_freq = self.inv_freq.to(device)  # Move inv_freq to the same device as inputs
#         sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq)  # Element-wise product for position and inverse frequency
#         emb = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)  # Concatenate sin and cos embeddings
#         emb = emb.unsqueeze(0)  # Add a batch dimension to the embedding
#         return emb.expand(batch_size, -1, -1)  # Repeat along the batch dimension

# # Main Transformer-based Autoencoder Model
# class TransformerDAE(nn.Module):
#     def __init__(self, signal_size=512, head_size=64, num_heads=8, ff_dim=64, num_transformer_blocks=6, dropout=0):
#         super(TransformerDAE, self).__init__()

#         self.conv1 = nn.Conv1d(1, 16, kernel_size=13, stride=2, padding=6)
#         self.gated_noise1 = AddGatedNoise()
#         self.conv1_ = nn.Conv1d(1, 16, kernel_size=13, stride=2, padding=6)
#         self.mul1 = nn.Multiply()

#         self.conv2 = nn.Conv1d(16, 32, kernel_size=13, stride=2, padding=6)
#         self.gated_noise2 = AddGatedNoise()
#         self.conv2_ = nn.Conv1d(16, 32, kernel_size=13, stride=2, padding=6)
#         self.mul2 = nn.Multiply()

#         self.conv3 = nn.Conv1d(32, 64, kernel_size=13, stride=2, padding=6)
#         self.gated_noise3 = AddGatedNoise()
#         self.conv3_ = nn.Conv1d(32, 64, kernel_size=13, stride=2, padding=6)
#         self.mul3 = nn.Multiply()

#         self.position_encoding = PositionalEncoding1D(signal_size)
#         self.transformer_blocks = nn.ModuleList([TransformerEncoder(head_size, num_heads, ff_dim, dropout) for _ in range(num_transformer_blocks)])

#         self.deconv1 = Conv1DTranspose(64, 64, kernel_size=13, stride=1, padding=6)
#         self.deconv2 = Conv1DTranspose(64, 32, kernel_size=13, stride=2, padding=6)
#         self.deconv3 = Conv1DTranspose(32, 16, kernel_size=13, stride=2, padding=6)
#         self.deconv4 = Conv1DTranspose(16, 1, kernel_size=13, stride=2, padding=6)

#         self.batchnorm1 = nn.BatchNorm1d(16)
#         self.batchnorm2 = nn.BatchNorm1d(32)
#         self.batchnorm3 = nn.BatchNorm1d(64)
#         self.batchnorm4 = nn.BatchNorm1d(64)

#     def forward(self, x):
#         # Encoder
#         x0 = F.elu(self.conv1(x))
#         x0 = self.gated_noise1(x0)
#         x0_ = self.conv1_(x)
#         xmul0 = x0 * x0_
#         xmul0 = self.batchnorm1(xmul0)

#         x1 = F.elu(self.conv2(xmul0))
#         x1 = self.gated_noise2(x1)
#         x1_ = self.conv2_(xmul0)
#         xmul1 = x1 * x1_
#         xmul1 = self.batchnorm2(xmul1)

#         x2 = F.elu(self.conv3(xmul1))
#         x2 = self.gated_noise3(x2)
#         x2_ = F.elu(self.conv3_(xmul1))
#         xmul2 = x2 * x2_
#         xmul2 = self.batchnorm3(xmul2)

#         # Positional Encoding and Transformer blocks
#         pos_enc = self.position_encoding(xmul2)
#         x3 = xmul2 + pos_enc

#         for transformer in self.transformer_blocks:
#             x3 = transformer(x3)

#         # Decoder
#         x5 = F.elu(self.deconv1(x3))
#         x5 = x5 + xmul2
#         x5 = self.batchnorm4(x5)

#         x6 = F.elu(self.deconv2(x5))
#         x6 = x6 + xmul1
#         x6 = self.batchnorm3(x6)

#         x7 = F.elu(self.deconv3(x6))
#         x7 = x7 + xmul0
#         x8 = self.deconv4(x7)

#         return x8


# # FFT Layer
# def fft_layer(x):
#     fft = torch.fft.fft(x, dim=-1)
#     amplitude = torch.abs(fft)
#     return amplitude.float()


# # FFT Layer
# def fft_layer(x):
#     # FFT 수행
#     fft = torch.fft.fft(x, dim=-1)
#     # Amplitude spectrum 계산
#     amplitude = torch.abs(fft)
#     return amplitude.float()

# # Modified GCL Layer
# class ModifiedGCL(nn.Module):
#     def __init__(self, in_channels, filters, kernel_size, strides):
#         super(ModifiedGCL, self).__init__()
#         # Conv1d expects the correct input channel size, which we will pass dynamically
#         self.conv_time = nn.Conv1d(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size, stride=strides, padding=(kernel_size // 2))
#         self.conv_freq = nn.Conv1d(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size, stride=strides, padding=(kernel_size // 2))
#         self.gate = nn.Conv1d(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size, stride=strides, padding=(kernel_size // 2))
#         self.bn = nn.BatchNorm1d(filters)

#     def forward(self, inputs):
#         # 시간 도메인 처리
#         x_time = self.conv_time(inputs)

#         # 주파수 도메인 처리
#         x_freq = fft_layer(inputs)
#         x_freq = self.conv_freq(x_freq)

#         # 게이팅 메커니즘
#         gate = torch.sigmoid(self.gate(inputs))

#         # 시간 및 주파수 도메인 정보 결합
#         out = gate * x_time + (1 - gate) * x_freq
#         return self.bn(out)

# # Main Transformer-based Autoencoder Model
# class TransformerFDAE(nn.Module):
#     def __init__(self, signal_size=512, head_size=64, num_heads=8, ff_dim=64, num_transformer_blocks=6, dropout=0):
#         super(TransformerFDAE, self).__init__()

#         self.gcl1 = ModifiedGCL(in_channels=1, filters=16, kernel_size=13, strides=2)
#         self.gcl2 = ModifiedGCL(in_channels=16, filters=32, kernel_size=13, strides=2)
#         self.gcl3 = ModifiedGCL(in_channels=32, filters=64, kernel_size=13, strides=2)

#         self.position_encoding = PositionalEncoding1D(signal_size)
#         self.transformer_blocks = nn.ModuleList([TransformerEncoder(head_size, num_heads, ff_dim, dropout) for _ in range(num_transformer_blocks)])

#         self.deconv1 = Conv1DTranspose(64, 64, kernel_size=13, stride=1, padding=6)
#         self.deconv2 = Conv1DTranspose(64, 32, kernel_size=13, stride=2, padding=6)
#         self.deconv3 = Conv1DTranspose(32, 16, kernel_size=13, stride=2, padding=6)
#         self.deconv4 = Conv1DTranspose(16, 1, kernel_size=13, stride=2, padding=6)

#         self.batchnorm1 = nn.BatchNorm1d(16)
#         self.batchnorm2 = nn.BatchNorm1d(32)
#         self.batchnorm3 = nn.BatchNorm1d(64)
#         self.batchnorm4 = nn.BatchNorm1d(64)

#     def forward(self, x):
#         # Encoder
#         x = self.gcl1(x)  # Input channel is 1
#         x = self.gcl2(x)  # Input channel is 16 after gcl1
#         x = self.gcl3(x)  # Input channel is 32 after gcl2

#         # Positional Encoding and Transformer blocks
#         pos_enc = self.position_encoding(x)
#         x = x + pos_enc

#         for transformer in self.transformer_blocks:
#             x = transformer(x)

#         # Decoder
#         x = F.elu(self.deconv1(x))
#         x = self.batchnorm4(x)

#         x = F.elu(self.deconv2(x))
#         x = self.batchnorm3(x)

#         x = F.elu(self.deconv3(x))
#         x = self.batchnorm2(x)

#         x = self.deconv4(x)
#         return x

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization, concatenate, Activation, Input, Conv2DTranspose, Lambda, LSTM, Reshape, Embedding
import tensorflow as tf 
import keras.backend as K

def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, activation='relu', padding='same'):
    """
        https://stackoverflow.com/a/45788699

        input_tensor: tensor, with the shape (batch_size, time_steps, dims)
        filters: int, output dimension, i.e. the output tensor will have the shape of (batch_size, time_steps, filters)
        kernel_size: int, size of the convolution kernel
        strides: int, convolution step size
        padding: 'same' | 'valid'
    """
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters,
                        kernel_size=(kernel_size, 1),
                        activation=activation,
                        strides=(strides, 1),
                        padding=padding)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x

##########################################################################

###### MODULES #######

def LFilter_module(x, layers):
    LB0 = Conv1D(filters=int(layers / 4),
                 kernel_size=3,
                 activation='linear',
                 strides=1,
                 padding='same')(x)
    LB1 = Conv1D(filters=int(layers / 4),
                kernel_size=5,
                activation='linear',
                strides=1,
                padding='same')(x)
    LB2 = Conv1D(filters=int(layers / 4),
                kernel_size=9,
                activation='linear',
                strides=1,
                padding='same')(x)
    LB3 = Conv1D(filters=int(layers / 4),
                kernel_size=15,
                activation='linear',
                strides=1,
                padding='same')(x)


    x = concatenate([LB0, LB1, LB2, LB3])

    return x


def NLFilter_module(x, layers):

    NLB0 = Conv1D(filters=int(layers / 4),
                  kernel_size=3,
                  activation='relu',
                  strides=1,
                  padding='same')(x)
    NLB1 = Conv1D(filters=int(layers / 4),
                kernel_size=5,
                activation='relu',
                strides=1,
                padding='same')(x)
    NLB2 = Conv1D(filters=int(layers / 4),
                kernel_size=9,
                activation='relu',
                strides=1,
                padding='same')(x)
    NLB3 = Conv1D(filters=int(layers / 4),
                kernel_size=15,
                activation='relu',
                strides=1,
                padding='same')(x)


    x = concatenate([NLB0, NLB1, NLB2, NLB3])

    return x


def LANLFilter_module(x, layers):
    LB0 = Conv1D(filters=int(layers / 8),
                 kernel_size=3,
                 activation='linear',
                 strides=1,
                 padding='same')(x)
    LB1 = Conv1D(filters=int(layers / 8),
                kernel_size=5,
                activation='linear',
                strides=1,
                padding='same')(x)
    LB2 = Conv1D(filters=int(layers / 8),
                kernel_size=9,
                activation='linear',
                strides=1,
                padding='same')(x)
    LB3 = Conv1D(filters=int(layers / 8),
                kernel_size=15,
                activation='linear',
                strides=1,
                padding='same')(x)

    NLB0 = Conv1D(filters=int(layers / 8),
                  kernel_size=3,
                  activation='relu',
                  strides=1,
                  padding='same')(x)
    NLB1 = Conv1D(filters=int(layers / 8),
                 kernel_size=5,
                 activation='relu',
                 strides=1,
                 padding='same')(x)
    NLB2 = Conv1D(filters=int(layers / 8),
                 kernel_size=9,
                 activation='relu',
                 strides=1,
                 padding='same')(x)
    NLB3 = Conv1D(filters=int(layers / 8),
                 kernel_size=15,
                 activation='relu',
                 strides=1,
                 padding='same')(x)

    x = concatenate([LB0, LB1, LB2, LB3, NLB0, NLB1, NLB2, NLB3])

    return x


def LANLFilter_module_dilated(x, layers):
    LB1 = Conv1D(filters=int(layers / 6),
                kernel_size=5,
                activation='linear',
                dilation_rate=3,
                padding='same')(x)
    LB2 = Conv1D(filters=int(layers / 6),
                kernel_size=9,
                activation='linear',
                dilation_rate=3,
                padding='same')(x)
    LB3 = Conv1D(filters=int(layers / 6),
                kernel_size=15,
                dilation_rate=3,
                activation='linear',
                padding='same')(x)

    NLB1 = Conv1D(filters=int(layers / 6),
                 kernel_size=5,
                 activation='relu',
                 dilation_rate=3,
                 padding='same')(x)
    NLB2 = Conv1D(filters=int(layers / 6),
                 kernel_size=9,
                 activation='relu',
                 dilation_rate=3,
                 padding='same')(x)
    NLB3 = Conv1D(filters=int(layers / 6),
                 kernel_size=15,
                 dilation_rate=3,
                 activation='relu',
                 padding='same')(x)

    x = concatenate([LB1, LB2, LB3, NLB1, NLB2, NLB3])
    # x = BatchNormalization()(x)

    return x


###### MODELS #######

def deep_filter_vanilla_linear():

    model = Sequential()

    model.add(Conv1D(filters=64,
                     kernel_size=9,
                     activation='linear',
                     input_shape=(512, 1),
                     strides=1,
                     padding='same'))
    model.add(Conv1D(filters=64,
                     kernel_size=9,
                     activation='linear',
                     strides=1,
                     padding='same'))


    model.add(Conv1D(filters=32,
                     kernel_size=9,
                     activation='linear',
                     strides=1,
                     padding='same'))
    model.add(Conv1D(filters=32,
                     kernel_size=9,
                     activation='linear',
                     strides=1,
                     padding='same'))


    model.add(Conv1D(filters=16,
                     kernel_size=9,
                     activation='linear',
                     strides=1,
                     padding='same'))
    model.add(Conv1D(filters=16,
                     kernel_size=9,
                     activation='linear',
                     strides=1,
                     padding='same'))


    model.add(Conv1D(filters=1,
                     kernel_size=9,
                     activation='linear',
                     strides=1,
                     padding='same'))
    return model


def deep_filter_vanilla_Nlinear():
    model = Sequential()

    model.add(Conv1D(filters=64,
                     kernel_size=9,
                     activation='relu',
                     input_shape=(512, 1),
                     strides=1,
                     padding='same'))
    model.add(Conv1D(filters=64,
                     kernel_size=9,
                     activation='relu',
                     strides=1,
                     padding='same'))


    model.add(Conv1D(filters=32,
                     kernel_size=9,
                     activation='relu',
                     strides=1,
                     padding='same'))
    model.add(Conv1D(filters=32,
                     kernel_size=9,
                     activation='relu',
                     strides=1,
                     padding='same'))


    model.add(Conv1D(filters=16,
                     kernel_size=9,
                     activation='relu',
                     strides=1,
                     padding='same'))
    model.add(Conv1D(filters=16,
                     kernel_size=9,
                     activation='relu',
                     strides=1,
                     padding='same'))


    model.add(Conv1D(filters=1,
                     kernel_size=9,
                     activation='linear',
                     strides=1,
                     padding='same'))
    return model


def deep_filter_I_linear():
    input_shape = (512, 1)
    input = Input(shape=input_shape)

    tensor = LFilter_module(input, 64)
    tensor = LFilter_module(tensor, 64)
    tensor = LFilter_module(tensor, 32)
    tensor = LFilter_module(tensor, 32)
    tensor = LFilter_module(tensor, 16)
    tensor = LFilter_module(tensor, 16)
    predictions = Conv1D(filters=1,
                         kernel_size=9,
                         activation='linear',
                         strides=1,
                         padding='same')(tensor)

    model = Model(inputs=[input], outputs=predictions)

    return model


def deep_filter_I_Nlinear():
    input_shape = (512, 1)
    input = Input(shape=input_shape)

    tensor = NLFilter_module(input, 64)
    tensor = NLFilter_module(tensor, 64)
    tensor = NLFilter_module(tensor, 32)
    tensor = NLFilter_module(tensor, 32)
    tensor = NLFilter_module(tensor, 16)
    tensor = NLFilter_module(tensor, 16)
    predictions = Conv1D(filters=1,
                         kernel_size=9,
                         activation='linear',
                         strides=1,
                         padding='same')(tensor)

    model = Model(inputs=[input], outputs=predictions)

    return model


def deep_filter_I_LANL():
    # TODO: Make the doc

    input_shape = (512, 1)
    input = Input(shape=input_shape)

    tensor = LANLFilter_module(input, 64)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module(tensor, 64)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module(tensor, 32)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module(tensor, 32)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module(tensor, 16)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module(tensor, 16)
    tensor = BatchNormalization()(tensor)
    predictions = Conv1D(filters=1,
                    kernel_size=9,
                    activation='linear',
                    strides=1,
                    padding='same')(tensor)

    model = Model(inputs=[input], outputs=predictions)

    return model


def deep_filter_model_I_LANL_dilated():
    # TODO: Make the doc

    input_shape = (512, 1)
    input = Input(shape=input_shape)

    tensor = LANLFilter_module(input, 64)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module_dilated(tensor, 64)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module(tensor, 32)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module_dilated(tensor, 32)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module(tensor, 16)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module_dilated(tensor, 16)
    tensor = BatchNormalization()(tensor)
    predictions = Conv1D(filters=1,
                    kernel_size=9,
                    activation='linear',
                    strides=1,
                    padding='same')(tensor)

    model = Model(inputs=[input], outputs=predictions)

    return model


def FCN_DAE():
    # Implementation of FCN_DAE approach presented in
    # Chiang, H. T., Hsieh, Y. Y., Fu, S. W., Hung, K. H., Tsao, Y., & Chien, S. Y. (2019).
    # Noise reduction in ECG signals using fully convolutional denoising autoencoders.
    # IEEE Access, 7, 60806-60813.

    input_shape = (512, 1)
    input = Input(shape=input_shape)

    x = Conv1D(filters=40,
               input_shape=(512, 1),
               kernel_size=16,
               activation='elu',
               strides=2,
               padding='same')(input)

    x = BatchNormalization()(x)

    x = Conv1D(filters=20,
               kernel_size=16,
               activation='elu',
               strides=2,
               padding='same')(x)

    x = BatchNormalization()(x)

    x = Conv1D(filters=20,
               kernel_size=16,
               activation='elu',
               strides=2,
               padding='same')(x)

    x = BatchNormalization()(x)

    x = Conv1D(filters=20,
               kernel_size=16,
               activation='elu',
               strides=2,
               padding='same')(x)

    x = BatchNormalization()(x)

    x = Conv1D(filters=40,
               kernel_size=16,
               activation='elu',
               strides=2,
               padding='same')(x)

    x = BatchNormalization()(x)

    x = Conv1D(filters=1,
               kernel_size=16,
               activation='elu',
               strides=1,
               padding='same')(x)

    x = BatchNormalization()(x)

    # Keras has no 1D Traspose Convolution, instead we use Conv2DTranspose function
    # in a souch way taht is mathematically equivalent
    x = Conv1DTranspose(input_tensor=x,
                        filters=1,
                        kernel_size=16,
                        activation='elu',
                        strides=1,
                        padding='same')

    x = BatchNormalization()(x)

    x = Conv1DTranspose(input_tensor=x,
                        filters=40,
                        kernel_size=16,
                        activation='elu',
                        strides=2,
                        padding='same')

    x = BatchNormalization()(x)

    x = Conv1DTranspose(input_tensor=x,
                        filters=20,
                        kernel_size=16,
                        activation='elu',
                        strides=2,
                        padding='same')

    x = BatchNormalization()(x)

    x = Conv1DTranspose(input_tensor=x,
                        filters=20,
                        kernel_size=16,
                        activation='elu',
                        strides=2,
                        padding='same')

    x = BatchNormalization()(x)

    x = Conv1DTranspose(input_tensor=x,
                        filters=20,
                        kernel_size=16,
                        activation='elu',
                        strides=2,
                        padding='same')

    x = BatchNormalization()(x)

    x = Conv1DTranspose(input_tensor=x,
                        filters=40,
                        kernel_size=16,
                        activation='elu',
                        strides=2,
                        padding='same')

    x = BatchNormalization()(x)

    predictions = Conv1DTranspose(input_tensor=x,
                        filters=1,
                        kernel_size=16,
                        activation='linear',
                        strides=1,
                        padding='same')

    model = Model(inputs=[input], outputs=predictions)
    return model


def DRRN_denoising():
    # Implementation of DRNN approach presented in
    # Antczak, K. (2018). Deep recurrent neural networks for ECG signal denoising.
    # arXiv preprint arXiv:1807.11551.    

    model = Sequential()
    model.add(LSTM(64, input_shape=(512, 1), return_sequences=True))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))

    return model

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization,\
                         concatenate, Activation, Input, Conv2DTranspose, Lambda, LSTM, GRU,Reshape, Embedding, GlobalAveragePooling1D,\
                         Multiply,Bidirectional


import keras.backend as K
from keras import layers
import tensorflow as tf
import numpy as np
from scipy import signal


sigLen = 512
def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, activation='relu', padding='same'):
    """
        https://stackoverflow.com/a/45788699

        input_tensor: tensor, with the shape (batch_size, time_steps, dims)
        filters: int, output dimension, i.e. the output tensor will have the shape of (batch_size, time_steps, filters)
        kernel_size: int, size of the convolution kernel
        strides: int, convolution step size
        padding: 'same' | 'valid'
    """
    x = Lambda(lambda x: tf.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters,
                        kernel_size=(kernel_size, 1),
                        activation=activation,
                        strides=(strides, 1),
                        padding=padding)(x)
    x = Lambda(lambda x: tf.squeeze(x, axis=2))(x)
    return x

##########################################################################


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = tf.stack((tf.sin(sin_inp), tf.cos(sin_inp)), -1)
    emb = tf.reshape(emb, (*emb.shape[:-2], -1))
    return emb
class TFPositionalEncoding1D(tf.keras.layers.Layer):
    def __init__(self, channels: int, dtype=tf.float32):
        """
        Args:
            channels int: The last dimension of the tensor you want to apply pos emb to.
        Keyword Args:
            dtype: output type of the encodings. Default is "tf.float32".
        """
        super(TFPositionalEncoding1D, self).__init__()

        #self.channels = int(np.ceil(channels / 2) * 2)
        self.channels = int(np.ceil(channels // 2) * 2)
        self.inv_freq = np.float32(
            1
            / np.power(
                10000, np.arange(0, self.channels, 2) / np.float32(self.channels)
            )
        )
        self.cached_penc = None

    @tf.function
    def call(self, inputs):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(inputs.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == inputs.shape:
            return self.cached_penc

        self.cached_penc = None
        _, x, org_channels = inputs.shape

        dtype = self.inv_freq.dtype
        pos_x = tf.range(x, dtype=dtype)
        sin_inp_x = tf.einsum("i,j->ij", pos_x, self.inv_freq)
        emb = tf.expand_dims(get_emb(sin_inp_x), 0)
        emb = emb[0]  # A bit of a hack
        self.cached_penc = tf.repeat(
            emb[None, :, :org_channels], tf.shape(inputs)[0], axis=0
        )

        return self.cached_penc
def transformer_encoder(inputs,head_size,num_heads,ff_dim,dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x= layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)  ##之前用的sigmoid, 可以试下gelu
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

ks = 13   #orig 13
ks1 = 7


def spatial_attention(inputs):
    attention = tf.keras.layers.Dense(1, activation='tanh')(inputs)
    attention = tf.keras.layers.Flatten()(attention)
    attention = tf.keras.layers.Activation('softmax')(attention)
    attention = tf.keras.layers.Reshape((-1, 1))(attention)
    return attention
def attention_module(inputs, filters):
    x = tf.keras.layers.Conv1D(filters, kernel_size=1, activation='relu')(inputs)
    x = tf.keras.layers.Conv1D(filters, kernel_size=3, padding='same', activation='relu')(x)
    attention = tf.keras.layers.GlobalAveragePooling1D()(x)
    attention = tf.keras.layers.Dense(filters, activation='sigmoid')(attention)
    attention = tf.keras.layers.Reshape((1, filters))(attention)
    scaled_inputs = tf.keras.layers.Multiply()([inputs, attention])
    return scaled_inputs

# 학습 중 노이즈를 추가하여 강건성을 높이는 역할
class AddGatedNoise(layers.Layer):
    def __init__(self, **kwargs):
        super(AddGatedNoise, self).__init__(**kwargs)

    def call(self, x, training=None):
        # 在训练时，使用随机噪声
        noise = tf.random.uniform(shape=tf.shape(x), minval=-1, maxval=1)
        return tf.keras.backend.in_train_phase(x * (1 + noise), x, training=training)
    
def Transformer_DAE(signal_size = sigLen,head_size=64,num_heads=8,ff_dim=64,num_transformer_blocks=6, dropout=0):   ###paper 1 model

    input_shape = (signal_size, 1)
    input = Input(shape=input_shape)

    x0 = Conv1D(filters=16,
                input_shape=(input_shape, 1),
                kernel_size=ks,
                activation='linear',  # 使用线性激活函数
                strides=2,
                padding='same')(input)

    # 使用自定义层添加乘性噪声，仅在训练时
    x0 = AddGatedNoise()(x0)

    # 应用sigmoid激活函数
    x0 = layers.Activation('sigmoid')(x0)
    # x0 = Dropout(0.3)(x0)
    x0_ = Conv1D(filters=16,
               input_shape=(input_shape, 1),
               kernel_size=ks,
               activation=None,
               strides=2,
               padding='same')(input)
    # x0_ = Dropout(0.3)(x0_)
    xmul0 = Multiply()([x0,x0_])

    xmul0 = BatchNormalization()(xmul0)

    x1 = Conv1D(filters=32,
                kernel_size=ks,
                activation='linear',  # 使用线性激活函数
                strides=2,
                padding='same')(xmul0)

    # 使用自定义层添加乘性噪声，仅在训练时
    x1 = AddGatedNoise()(x1)

    # 应用sigmoid激活函数
    x1 = layers.Activation('sigmoid')(x1)

    # x1 = Dropout(0.3)(x1)
    x1_ = Conv1D(filters=32,
               kernel_size=ks,
               activation=None,
               strides=2,
               padding='same')(xmul0)
    # x1_ = Dropout(0.3)(x1_)
    xmul1 = Multiply()([x1, x1_])
    xmul1 = BatchNormalization()(xmul1)

    x2 = Conv1D(filters=64,
               kernel_size=ks,
               activation='linear',
               strides=2,
               padding='same')(xmul1)
    x2 = AddGatedNoise()(x2)
    # 应用sigmoid激活函数
    x2 = layers.Activation('sigmoid')(x2)
    # x2 = Dropout(0.3)(x2)
    x2_ = Conv1D(filters=64,
               kernel_size=ks,
               activation='elu',
               strides=2,
               padding='same')(xmul1)
    # x2_ = Dropout(0.3)(x2_)
    xmul2 = Multiply()([x2, x2_])

    xmul2 = BatchNormalization()(xmul2)
    #位置编码
    position_embed = TFPositionalEncoding1D(signal_size)
    x3 = xmul2+position_embed(xmul2)
    #
    for _ in range(num_transformer_blocks):
        x3 = transformer_encoder(x3,head_size,num_heads,ff_dim, dropout)
    # x = layers.GlobalAvgPool1D(data_format='channels_first')(x)
    # x4 = x4+xmul2
    x4 = x3
    x5 = Conv1DTranspose(input_tensor=x4,
                        filters=64,
                        kernel_size=ks,
                        activation='elu',
                        strides=1,
                        padding='same')
    x5 = x5+xmul2
    x5 = BatchNormalization()(x5)

    x6 = Conv1DTranspose(input_tensor=x5,
                        filters=32,
                        kernel_size=ks,
                        activation='elu',
                        strides=2,
                        padding='same')
    x6 = x6+xmul1
    x6 = BatchNormalization()(x6)

    x7 = Conv1DTranspose(input_tensor=x6,
                        filters=16,
                        kernel_size=ks,
                        activation='elu',
                        strides=2,
                        padding='same')

    x7 = x7 + xmul0 #res

    x8 = BatchNormalization()(x7)
    predictions = Conv1DTranspose(
                        input_tensor=x8,
                        filters=1,
                        kernel_size=ks,
                        activation='linear',
                        strides=2,
                        padding='same')

    model = Model(inputs=[input], outputs=predictions)
    return model

def to_frequency_domain(x):
    return torch.rfft(x, signal_ndim=1, normalized=True, onesided=True)

import tensorflow as tf
from keras import layers

def fft_layer(x):
    # FFT 수행
    fft = tf.signal.fft(tf.cast(x, tf.complex64))
    # 절대값 취하여 amplitude spectrum 얻기
    amplitude = tf.abs(fft)
    # 실수 부분만 사용
    return tf.cast(amplitude, tf.float32)

class ModifiedGCL(layers.Layer):
    def __init__(self, filters, kernel_size, strides, **kwargs):
        super(ModifiedGCL, self).__init__(**kwargs)
        self.conv_time = layers.Conv1D(filters, kernel_size, strides=strides, padding='same')
        self.conv_freq = layers.Conv1D(filters, kernel_size, strides=strides, padding='same')
        self.gate = layers.Conv1D(filters, kernel_size, strides=strides, padding='same', activation='sigmoid')
        self.bn = layers.BatchNormalization()

    def call(self, inputs):
        # 시간 도메인 처리
        x_time = self.conv_time(inputs)
        
        # 주파수 도메인 처리
        x_freq = fft_layer(inputs)
        x_freq = self.conv_freq(x_freq)
        
        # 게이팅 메커니즘
        gate = self.gate(inputs)
        
        # 시간 및 주파수 도메인 정보 결합
        out = gate * x_time + (1 - gate) * x_freq
        return self.bn(out)

def Transformer_FDAE(signal_size=sigLen, head_size=64, num_heads=8, ff_dim=64, num_transformer_blocks=6, dropout=0):
    input_shape = (signal_size, 1)
    inputs = Input(shape=input_shape)

    # GCL 레이어 적용
    x = ModifiedGCL(16, ks, strides=2)(inputs)
    x = ModifiedGCL(32, ks, strides=2)(x)
    x = ModifiedGCL(64, ks, strides=2)(x)

    # 위치 인코딩
    position_embed = TFPositionalEncoding1D(x.shape[-2])
    x = x + position_embed(x)

    # Transformer 블록
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    # 디코더 부분
    x = Conv1DTranspose(input_tensor=x, filters=64, kernel_size=ks, activation='elu', strides=1, padding='same')
    x = BatchNormalization()(x)
    x = Conv1DTranspose(input_tensor=x, filters=32, kernel_size=ks, activation='elu', strides=2, padding='same')
    x = BatchNormalization()(x)
    x = Conv1DTranspose(input_tensor=x, filters=16, kernel_size=ks, activation='elu', strides=2, padding='same')
    x = BatchNormalization()(x)
    
    predictions = Conv1DTranspose(input_tensor=x, filters=1, kernel_size=ks, activation='linear', strides=2, padding='same')

    model = Model(inputs=inputs, outputs=predictions)
    return model


