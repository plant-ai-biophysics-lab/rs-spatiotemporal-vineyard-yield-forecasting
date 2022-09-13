import torch
import torch.nn as nn
from src.ModelEngine import *

    
class UNet2DConvLSTM(nn.Module):
    
    def __init__(self, in_channels, out_channels, num_filters, dropout = None, Emb_Channels = None, batch_size = None, botneck_size = None):
        super(UNet2DConvLSTM, self).__init__()
        
        self.in_channels   = in_channels
        self.out_channels  = out_channels
        self.num_filters   = num_filters
        self.dropout       = dropout
        self.Emb_Channels  = Emb_Channels
        self.batch_size    = batch_size
        self.botneck_size  = botneck_size
        

        # Down sampling
        self.Encoder1    = Encoder2D(self.in_channels, self.num_filters, self.num_filters, self.dropout)   
        self.Pool1       = MaxPolling2D() 
        self.Encoder2    = Encoder2D(self.num_filters, self.num_filters, self.num_filters * 2, self.dropout)
        self.Pool2       = MaxPolling2D() 
        self.Encoder3    = Encoder2D(self.num_filters * 2, self.num_filters * 2, self.num_filters * 4, self.dropout)
        self.Pool3       = MaxPolling2D()
        
        
        # LSTM
        self.LSTM        = ConvLSTM(img_size= ((self.batch_size, 15, (self.num_filters * 4) + self.Emb_Channels, self.botneck_size, self.botneck_size)), 
                                    img_width = self.botneck_size,
                                    input_dim = (self.num_filters * 4) + self.Emb_Channels, 
                                    hidden_dim = (self.num_filters * 4),
                                    kernel_size=(3,3), cnn_dropout=self.dropout, rnn_dropout=self.dropout, 
                                    batch_first=True,  bias=True, peephole=False,  layer_norm=False,  
                                    return_sequence=True, bidirectional=False) 
    

        # Up sampling
        self.Up3          = Decoder2D((self.num_filters * 4), (self.num_filters * 4) , self.dropout)   
        self.Encoder3Up   = Encoder2D((self.num_filters * 8 ), self.num_filters * 8, self.num_filters * 6, self.dropout)  
        self.Up2          = Decoder2D(self.num_filters * 6, self.num_filters * 6, self.dropout) 
        self.Encoder2Up   = Encoder2D(self.num_filters * 8, self.num_filters * 4, self.num_filters * 3, self.dropout)
        self.Up1          = Decoder2D(self.num_filters * 3, self.num_filters*3, self.dropout) 
        self.Encoder1Up   = Encoder2D(self.num_filters * 4, self.num_filters * 2, self.num_filters, self.dropout) 
      
            
        self.out4 = OutConv2D((self.num_filters*4)+ self.Emb_Channels, out_channels) 
        self.out1 = OutConv2D(self.num_filters, out_channels)
        
    def forward(self, x, e):
        f, b = [], []
        for i in range(x.shape[-1]):
            img = x[:, :, :, :, i]
            # Down sampling
            Encoder1 = self.Encoder1(img)
            Pool1    = self.Pool1(Encoder1)
            Encoder2 = self.Encoder2(Pool1)
            Pool2    = self.Pool2(Encoder2)
            Encoder3 = self.Encoder3(Pool2)
            Pool3    = self.Pool3(Encoder3)
            #Pool3    = Pool3.unsqueeze(1)
            Conc4    = torch.cat([Pool3, e], dim=1).float() 
            Conc4    = Conc4.unsqueeze(1)
            #out4     = self.out4(Conc4)      # -> [1, 10, 10, 1]
            
            b.append(Conc4)
        
        time_series_stacked = torch.cat(b, dim=1).float() 
        #print(time_series_stacked.shape)
        layer_output, last_state = self.LSTM(time_series_stacked) 
        #print(layer_output.shape)
        for i in range(15):
            in_        = layer_output[:, i, :, :, :] 
            #print(in_.shape)
            #Up3        = self.Up3(in_) 
            #Conc4      = torch.cat([in_, e], dim=1).float() 
            Up3        = self.Up3(in_) 
            Conc3      = torch.cat([Up3, Encoder3], dim=1) 
            Encoder3Up = self.Encoder3Up(Conc3) 

            Up2        = self.Up2(Encoder3Up) 
            Conc2      = torch.cat([Up2, Encoder2], dim=1) 
            Encoder2Up = self.Encoder2Up(Conc2) 

            Up1        = self.Up1(Encoder2Up) 
            Conc1      = torch.cat([Up1, Encoder1], dim=1) 
            Encoder1Up = self.Encoder1Up(Conc1)

            # Output
            out1     = self.out1(Encoder1Up) # -> [1, 80, 80, 1]
            f.append(out1)

            
        return f


class UNet2DConvLSTM_S1(nn.Module):
    
    def __init__(self, in_channels, out_channels, num_filters, dropout = None, Emb_Channels = None, batch_size = None, botneck_size = None):
        super(UNet2DConvLSTM_S1, self).__init__()
        
        self.in_channels   = in_channels
        self.out_channels  = out_channels
        self.num_filters   = num_filters
        self.dropout       = dropout
        self.Emb_Channels  = Emb_Channels
        self.batch_size    = batch_size
        self.botneck_size  = botneck_size
        

        # Down sampling
        self.Encoder1    = Encoder2D(self.in_channels, self.num_filters, self.num_filters, self.dropout)   
        self.Pool1       = MaxPolling2D() 
        self.Encoder2    = Encoder2D(self.num_filters, self.num_filters, self.num_filters * 2, self.dropout)
        self.Pool2       = MaxPolling2D() 
        self.Encoder3    = Encoder2D(self.num_filters * 2, self.num_filters * 2, self.num_filters * 4, self.dropout)
        self.Pool3       = MaxPolling2D()
        
        
        # LSTM
        self.LSTM        = ConvLSTM(img_size= ((self.batch_size, 15, (self.num_filters * 4) + self.Emb_Channels, self.botneck_size, self.botneck_size)), 
                                    img_width = self.botneck_size,
                                    input_dim = (self.num_filters * 4) + self.Emb_Channels, 
                                    hidden_dim = (self.num_filters * 4),
                                    kernel_size=(3,3), cnn_dropout=self.dropout, rnn_dropout=self.dropout, 
                                    batch_first=True,  bias=True, peephole=False,  layer_norm=False,  
                                    return_sequence=True, bidirectional=False) 
    

        # Up sampling
        self.Up3          = Decoder2D((self.num_filters * 4), (self.num_filters * 4) , self.dropout)   
        self.Encoder3Up   = Encoder2D((self.num_filters * 8 ), self.num_filters * 8, self.num_filters * 6, self.dropout)  
        self.Up2          = Decoder2D(self.num_filters * 6, self.num_filters * 6, self.dropout) 
        self.Encoder2Up   = Encoder2D(self.num_filters * 8, self.num_filters * 4, self.num_filters * 3, self.dropout)
        self.Up1          = Decoder2D(self.num_filters * 3, self.num_filters*3, self.dropout) 
        self.Encoder1Up   = Encoder2D(self.num_filters * 4, self.num_filters * 2, self.num_filters, self.dropout) 
      
            
        self.out4 = OutConv2D((self.num_filters*4)+ self.Emb_Channels, out_channels) 
        self.out1 = OutConv2D(self.num_filters, out_channels)
        
    def forward(self, x):
        f, b = [], []
        for i in range(x.shape[-1]):
            img = x[:, :, :, :, i]
            # Down sampling
            Encoder1 = self.Encoder1(img)
            Pool1    = self.Pool1(Encoder1)
            Encoder2 = self.Encoder2(Pool1)
            Pool2    = self.Pool2(Encoder2)
            Encoder3 = self.Encoder3(Pool2)
            Pool3    = self.Pool3(Encoder3)
            Pool3    = Pool3.unsqueeze(1)
            b.append(Pool3)
        
        time_series_stacked = torch.cat(b, dim=1).float() 

        layer_output, last_state = self.LSTM(time_series_stacked) 

        for i in range(15):
            in_        = layer_output[:, i, :, :, :] 

            Up3        = self.Up3(in_) 
            Conc3      = torch.cat([Up3, Encoder3], dim=1) 
            Encoder3Up = self.Encoder3Up(Conc3) 

            Up2        = self.Up2(Encoder3Up) 
            Conc2      = torch.cat([Up2, Encoder2], dim=1) 
            Encoder2Up = self.Encoder2Up(Conc2) 

            Up1        = self.Up1(Encoder2Up) 
            Conc1      = torch.cat([Up1, Encoder1], dim=1) 
            Encoder1Up = self.Encoder1Up(Conc1)

            # Output
            out1     = self.out1(Encoder1Up)
            f.append(out1)
            
        return f
