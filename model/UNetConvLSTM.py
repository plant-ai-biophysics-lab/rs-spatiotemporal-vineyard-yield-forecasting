import torch
import torch.nn as nn
from model.cnn_modules import *
from torch.nn import init
    
class UNet2DConvLSTM(nn.Module):
    def __init__(self,         
                in_channels: int = 6, 
                out_channels: int = 1, 
                num_filters: int = 16, 
                embd_channels: int = 4,
                dropout: float = 0.1, 
                batch_size: int = 64, 
                bottelneck_size: int = 2
        ):
        super(UNet2DConvLSTM, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = 16
        self.dropout = dropout
        self.Emb_Channels = 4
        self.batch_size = 512
        self.botneck_size = 2

        # Down sampling
        self.Encoder1 = Encoder2D(self.in_channels, self.num_filters, self.num_filters, self.dropout)
        self.Pool1 = MaxPooling2D()
        self.Encoder2 = Encoder2D(self.num_filters, self.num_filters, self.num_filters * 2, self.dropout)
        self.Pool2 = MaxPooling2D()
        self.Encoder3 = Encoder2D(self.num_filters * 2, self.num_filters * 2, self.num_filters * 4, self.dropout)
        self.Pool3 = MaxPooling2D()

        # LSTM at the bottleneck for capturing temporal dependencies
        self.LSTM = ConvLSTM(img_size=(self.batch_size, 15, (self.num_filters * 4) + self.Emb_Channels, self.botneck_size, self.botneck_size),
                             img_width=self.botneck_size,
                             input_dim=(self.num_filters * 4) + self.Emb_Channels,
                             hidden_dim=(self.num_filters * 4),
                             kernel_size=(3, 3), cnn_dropout=self.dropout, rnn_dropout=self.dropout,
                             batch_first=True, bias=True, peephole=False, layer_norm=False,
                             return_sequence=True, bidirectional=False)

        # Up sampling
        self.Up3 = Decoder2D((self.num_filters * 4), (self.num_filters * 4), self.dropout)
        self.Encoder3Up = Encoder2D((self.num_filters * 8), self.num_filters * 8, self.num_filters * 6, self.dropout)
        self.Up2 = Decoder2D(self.num_filters * 6, self.num_filters * 6, self.dropout)
        self.Encoder2Up = Encoder2D(self.num_filters * 8, self.num_filters * 4, self.num_filters * 3, self.dropout)
        self.Up1 = Decoder2D(self.num_filters * 3, self.num_filters * 3, self.dropout)
        self.Encoder1Up = Encoder2D(self.num_filters * 4, self.num_filters * 2, self.num_filters, self.dropout)

        # Output layer for pixel-wise regression
        self.out1 = OutConv2D(self.num_filters, 1) 


        # Apply weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # Initialize weights for Convolutional and BatchNorm layers
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight, mode='fan_out',  nonlinearity='leaky_relu', a=0.1) #,
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

    def forward(self, x, e):

        # Encoder outputs for skip connections
        enc_outputs = []

        for i in range(x.shape[-1]):
            img = x[:, :, :, :, i]

            # Down sampling
            Encoder1 = self.Encoder1(img)
            Pool1 = self.Pool1(Encoder1)
            Encoder2 = self.Encoder2(Pool1)
            Pool2 = self.Pool2(Encoder2)
            Encoder3 = self.Encoder3(Pool2)
            Pool3 = self.Pool3(Encoder3)

            # Collect encoder outputs for later skip connections
            enc_outputs.append((Encoder1, Encoder2, Encoder3))

            if self.Emb_Channels == 0:
                Conc4 = Pool3.unsqueeze(1)
            else:
                Conc4 = torch.cat([Pool3, e], dim=1).float()
                Conc4 = Conc4.unsqueeze(1)

            # Collect time series data
            if i == 0:
                time_series_stacked = Conc4
            else:
                time_series_stacked = torch.cat([time_series_stacked, Conc4], dim=1)

        # LSTM processing
        layer_output, last_state = self.LSTM(time_series_stacked)

        # Decoder outputs
        reg_outputs = []

        for i in range(15):
            in_ = layer_output[:, i, :, :, :]

            # Up sampling with skip connections
            Up3 = self.Up3(in_)
            Conc3 = torch.cat([Up3, enc_outputs[i][2]], dim=1)  # Skip connection from Encoder3
            Encoder3Up = self.Encoder3Up(Conc3)

            Up2 = self.Up2(Encoder3Up)
            Conc2 = torch.cat([Up2, enc_outputs[i][1]], dim=1)  # Skip connection from Encoder2
            Encoder2Up = self.Encoder2Up(Conc2)

            Up1 = self.Up1(Encoder2Up)
            Conc1 = torch.cat([Up1, enc_outputs[i][0]], dim=1)  # Skip connection from Encoder1
            Encoder1Up = self.Encoder1Up(Conc1)

            # Output
            reg = self.out1(Encoder1Up)
            reg_outputs.append(reg)

        return reg_outputs

