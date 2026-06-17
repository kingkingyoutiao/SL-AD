import torch
import torch.nn as nn
from backbone.CNN_Auto.Autoformer import Model as Autoformer

from plugin.Plugin.model import Plugin
class CNNDownsample(nn.Module):
    def __init__(self, in_channels, kernel_size=6,stride=2, num_layers=3):
        super(CNNDownsample, self).__init__()
        layers = []
        for i in range(num_layers-1):
            layers.append(nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=2, padding=kernel_size // 2))
            layers.append(nn.ReLU())
        layers.append(nn.Conv1d(in_channels, in_channels, kernel_size=7, stride=2, padding=44))
        layers.append(nn.ReLU())
        self.conv_blocks = nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_blocks(x)
        x = x.permute(0, 2, 1)
        return x
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.args = configs
        self.flag=self.args.flag
        self.pred_len=self.args.pred_len
        self.cnn_downsample = CNNDownsample(
            in_channels=configs.enc_in,
            kernel_size=144,
            stride=2,
            num_layers=configs.num_cnn_layers
        )
        self.plugin=Plugin(configs,self.args.enc_in)
        self.autoformer = Autoformer(configs)
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        x_enc = self.cnn_downsample(x_enc) 

        x_enc_copy, x_mark_enc_copy, x_mark_dec_copy = x_enc.clone(), x_mark_enc.clone(), x_mark_dec.clone()
        if self.args.output_attention:
            pred, attn, cross_attn, enc_delay, dec_delay, enc_delay_stren, dec_delay_stren= self.autoformer(x_enc, x_mark_enc, x_dec, x_mark_dec)
        else:
            pred= self.autoformer(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.flag == 'Plugin':
            pred = self.plugin(x_enc_copy, x_mark_enc_copy, pred, x_mark_dec_copy[:, -self.pred_len:, :])
        if self.args.output_attention:
            return pred, attn, cross_attn, enc_delay, dec_delay, enc_delay_stren, dec_delay_stren
        else:
            return pred