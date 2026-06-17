import torch
import torch.nn as nn
import torch.nn.functional as F


class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        if self.attention.inner_correlation.output_attention:
            new_x, attn, delay, delaystength = self.attention(
                x, x, x,
                attn_mask=attn_mask
            )
        else:
            new_x, attn = self.attention(
                x, x, x,
                attn_mask=attn_mask
            )
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        if self.attention.inner_correlation.output_attention:
            return res, attn, delay, delaystength
        return res, attn


class Encoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        dela=[]
        delaysten=[]
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                if attn_layer.attention.inner_correlation.output_attention:
                    x, attn, delay, delaystrength = attn_layer(x, attn_mask=attn_mask)
                    dela.append(delay)
                    delaysten.append(delaystrength)
                else:
                    x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)
        if attn_layer.attention.inner_correlation.output_attention:
            return x, attns, dela, delaysten
        return x, attns


class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x, trend1 = self.decomp1(x)
        if self.cross_attention.inner_correlation.output_attention:
            cross_out, cross_corr, delay, delaystrength = self.cross_attention(
                x, cross, cross,
                attn_mask=cross_mask
            )
        else:
            cross_out, cross_corr = self.cross_attention(
                x, cross, cross,
                attn_mask=cross_mask
            )
        x = x + self.dropout(cross_out)
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        if self.cross_attention.inner_correlation.output_attention:
            return x, residual_trend, cross_corr,delay, delaystrength
        return x, residual_trend, cross_corr



class Decoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        attns = []
        dela=[]
        delaysten=[]
        for layer in self.layers:
            if layer.cross_attention.inner_correlation.output_attention:
                x, residual_trend, cross_corr, delay, delaystrength = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
                dela.append(delay)
                delaysten.append(delaystrength)
            else:
                x, residual_trend, cross_corr = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend
            attns.append(cross_corr)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        if layer.cross_attention.inner_correlation.output_attention:
            return x, trend, attns, dela, delaysten
        return x, trend, attns

if __name__ == '__main__':
    def test_moving_avg():
        # 假设原始输入序列为一个 batch_size = 1 的时间序列，包含 5 个时间步，每个时间步 3 个特征
        input_seq = torch.tensor([[[1.0, 2.0, 3.0],
                                   [2.0, 3.0, 4.0],
                                   [3.0, 4.0, 5.0],
                                   [4.0, 5.0, 6.0],
                                   [5.0, 6.0, 7.0]]])  # Shape: (1, 5, 3)

        print("Input sequence:")
        print(input_seq,input_seq.shape)

        # 设置 kernel_size = 4 和 stride = 1
        kernel_size = 4
        stride = 1

        # 创建 moving_avg 层并将输入数据传入
        model = moving_avg(kernel_size=kernel_size, stride=stride)
        output = model(input_seq)

        print("\nOutput sequence after applying moving average:")
        print(output,output.shape)


    # 运行测试程序
    test_moving_avg()