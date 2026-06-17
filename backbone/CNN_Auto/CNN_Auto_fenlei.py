import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve, auc
from backbone.CNN_Auto.Autoformer import Model
from backbone.CNN_Auto.layers.Embed import DataEmbedding_wo_pos
from backbone.CNN_Auto.layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from backbone.CNN_Auto.layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
from plugin.Plugin.model import Plugin
class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, num_classes,gpu):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.lstm.flatten_parameters = lambda: None
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.gpu = gpu

    def forward(self, x):
        x = x.float()  # 确保输入为 torch.float32
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim, device=x.device).float()
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim, device=x.device).float()
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc1(out)
        out = nn.functional.relu(out)
        out = self.fc2(out)
        return out

class Fenlei_Model(nn.Module):
    def __init__(self,ts, configs):
        super(Fenlei_Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Decomp
        kernel_size = configs.moving_avg
        self.configs=configs

        if len(ts)>0:
            self.cnn_downsample=ts[0].module.cnn_downsample
            self.decomp = ts[0].module.autoformer.decomp
            self.enc_embedding=ts[0].module.autoformer.enc_embedding
            self.encoder=ts[0].module.autoformer.encoder
        else:
            self.decomp = series_decomp(kernel_size)
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,configs.dropout)
            self.encoder = Encoder(
                [
                    EncoderLayer(
                        AutoCorrelationLayer(
                            AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=configs.output_attention),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        moving_avg=configs.moving_avg,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for l in range(configs.e_layers)
                ],
                norm_layer=my_Layernorm(configs.d_model)
            )
        self.fenlei=BiLSTM(configs.d_model,configs.hidden_nc,configs.num_layers,configs.num_classes,configs.gpu)
    def forward(self, x_enc, x_mark_enc,enc_self_mask=None):
        x_enc = self.cnn_downsample(x_enc)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=enc_self_mask)
        enc_out.requires_grad_(True)
        enc_out.retain_grad()
        cls_out = self.fenlei(enc_out)
        return cls_out,enc_out
    def get_importance(self, x_enc, x_mark_enc, target_class=None):
        cls_out, enc_out = self.forward(x_enc, x_mark_enc)
        if target_class is None:
            target_class = cls_out[0].argmax()
        score = cls_out[0, target_class]
        self.zero_grad()
        score.backward(retain_graph=True)
        grads = enc_out.grad[0]  # [seq_len, d_model]
        importance = grads.abs().mean(dim=-1)  # [seq_len]
        return importance.detach().cpu().numpy()
class Fenlei_Model_Free1_0(nn.Module):
    def __init__(self, configs):
        super(Fenlei_Model_Free1_0, self).__init__()
        self.ts = Model(configs)
        self.pred_len = configs.pred_len
        self.fim = 3
        self.args=configs
        self.bilstm_head = BiLSTM(
            embedding_dim=self.fim,
            hidden_dim=configs.hidden_nc,
            num_layers=configs.num_layers,
            num_classes=configs.num_classes,
            gpu=configs.gpu
        )
        self.fc_head = nn.Sequential(
            nn.Linear(self.fim * self.pred_len, 128),
            nn.ReLU(),
            nn.Linear(128, configs.num_classes)
        )
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, use_bilstm=True):

        dec_inp = torch.zeros_like(x_dec[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([x_dec[:, :self.args.label_len, :], dec_inp], dim=1).float()
        ts_output= self.ts(x_enc.double(), x_mark_enc.double(), dec_inp.double(), x_mark_dec.double())
        if use_bilstm:
            out = self.bilstm_head(ts_output)
        else:
            batch_size = ts_output.size(0)
            flattened_output = ts_output.view(batch_size, -1)
            out = self.fc_head(flattened_output)
        return out,ts_output

