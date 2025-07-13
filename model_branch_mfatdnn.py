import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.nnet.CNN import Conv1d as _Conv1d
from speechbrain.nnet.normalization import BatchNorm1d as _BatchNorm1d
from Muti_Scale_TimeAtt import MutiScaleTimeAttention
from merge import merge


# Skip transpose as much as possible for efficiency
class Conv1d(_Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(skip_transpose=True, *args, **kwargs)


class BatchNorm1d(_BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(skip_transpose=True, *args, **kwargs)


class TDNNBlock(nn.Module):


    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        activation=nn.ReLU,
    ):
        super(TDNNBlock, self).__init__()
        self.conv = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.activation = activation()
        self.norm = BatchNorm1d(input_size=out_channels)

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))

class Conv2D_Basic_Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        stride=(0,0),
    ):
        super(Conv2D_Basic_Block, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride
        )
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class Res2NetBlock(torch.nn.Module):


    def __init__(self, in_channels, out_channels, scale=8, dilation=1, dtype='TDNN'):
        super(Res2NetBlock, self).__init__()
        assert in_channels % scale == 0
        assert out_channels % scale == 0
        self.dtype = dtype
        in_channel = in_channels // scale  
        hidden_channel = out_channels // scale
        if self.dtype == 'Conv2D':
            self.blocks = nn.ModuleList(
                [
                    Conv2D_Basic_Block(
                        in_channel, hidden_channel, kernel_size=(3,3), padding=(1,1), stride=(1,1)
                    )
                    for i in range(scale - 1)  
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    TDNNBlock(
                        in_channel, hidden_channel, kernel_size=3, dilation=dilation
                    )
                    for i in range(scale - 1)
                ]
            )
        self.scale = scale
    def forward(self, x):
        y = []
        for i, x_i in enumerate(torch.chunk(x, self.scale, dim=1)):  
            if i == 0:
                y_i = x_i
            elif i == 1:
                y_i = self.blocks[i - 1](x_i)
            else:
                y_i = self.blocks[i - 1](x_i + y_i)
            if self.dtype == 'TDNN':
                y.append(y_i)
            else:
                y.append(torch.flatten(y_i, start_dim=1, end_dim=2))
        if self.dtype == 'TDNN':
            y = torch.cat(y, dim=1)
        return y

class SEBlock(nn.Module):

    def __init__(self, in_channels, se_channels, out_channels):
        super(SEBlock, self).__init__()

        self.conv1 = Conv1d(
            in_channels=in_channels, out_channels=se_channels, kernel_size=1
        )
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = Conv1d(
            in_channels=se_channels, out_channels=out_channels, kernel_size=1
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, lengths=None):
        L = x.shape[-1]
        if lengths is not None:
            mask = length_to_mask(lengths * L, max_len=L, device=x.device)
            mask = mask.unsqueeze(1)
            total = mask.sum(dim=2, keepdim=True)
            s = (x * mask).sum(dim=2, keepdim=True) / total
        else:
            s = x.mean(dim=2, keepdim=True)

        s = self.relu(self.conv1(s))
        s = self.sigmoid(self.conv2(s))

        return s * x


class AttentiveStatisticsPooling(nn.Module):

    def __init__(self, channels, attention_channels=128, global_context=True):
        super().__init__()

        self.eps = 1e-12
        self.global_context = global_context
        if global_context:
            self.tdnn = TDNNBlock(channels * 3, attention_channels, 1, 1)
        else:
            self.tdnn = TDNNBlock(channels, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = Conv1d(
            in_channels=attention_channels, out_channels=channels, kernel_size=1
        )

    def forward(self, x, lengths=None):
        """Calculates mean and std for a batch (input tensor).

        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape [N, C, L].
        """
        L = x.shape[-1]

        def _compute_statistics(x, m, dim=2, eps=self.eps):
            mean = (m * x).sum(dim)
            std = torch.sqrt(
                (m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps)
            )
            return mean, std

        if lengths is None:
            lengths = torch.ones(x.shape[0], device=x.device)

        # Make binary mask of shape [N, 1, L]
        mask = length_to_mask(lengths * L, max_len=L, device=x.device)
        mask = mask.unsqueeze(1)

        # Expand the temporal context of the pooling layer by allowing the
        # self-attention to look at global properties of the utterance.
        if self.global_context:
            # torch.std is unstable for backward computation
            # https://github.com/pytorch/pytorch/issues/4320
            total = mask.sum(dim=2, keepdim=True).float()
            mean, std = _compute_statistics(x, mask / total)
            mean = mean.unsqueeze(2).repeat(1, 1, L)
            std = std.unsqueeze(2).repeat(1, 1, L)
            attn = torch.cat([x, mean, std], dim=1)
        else:
            attn = x

        # Apply layers
        attn = self.conv(self.tanh(self.tdnn(attn)))

        # Filter out zero-paddings
        attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=2)
        mean, std = _compute_statistics(x, attn)
        # Append mean and std of the batch
        pooled_stats = torch.cat((mean, std), dim=1)
        pooled_stats = pooled_stats.unsqueeze(2)

        return pooled_stats


class SERes2NetBlock(nn.Module):


    def __init__(
        self,
        in_channels,
        out_channels,
        res2net_scale=8,
        se_channels=128,
        kernel_size=1,
        dilation=1,
        activation=torch.nn.ReLU,
    ):
        super().__init__()
        self.out_channels = out_channels


        self.tdnn1 = TDNNBlock(
            in_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            activation=activation,
        )
        self.res2net_block = Res2NetBlock(
            out_channels, out_channels, res2net_scale, dilation, dtype='TDNN'
        )
        self.tdnn2 = TDNNBlock(
            out_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            activation=activation,
        )
        self.se_block = SEBlock(out_channels, se_channels, out_channels)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

    def forward(self, x, lengths=None):
        residual = x
        if self.shortcut:
            residual = self.shortcut(x)

        x = self.tdnn1(x)
        x = self.res2net_block(x)
        x = self.tdnn2(x)
        x = self.se_block(x, lengths)
        return x + residual


class Att_layer(torch.nn.Module):
    def __init__(
            self,
            c_1st_conv,
            input_res2net_scale,
            SE_neur=8,  
            sub_channel=160,
    ):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.att_se = nn.Sequential(
            nn.Linear(sub_channel, SE_neur),
            nn.ReLU(inplace=True),
            nn.Linear(SE_neur, sub_channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x).squeeze()
        y = self.att_se(y)
        y = y.unsqueeze(dim=-1)
        return x * y


class Att_Block(torch.nn.Module):
    def __init__(
            self,
            c_1st_conv,
            outchannel,
            input_res2net_scale,
            SE_neur=8,
            se_channel=32,
            last_layer=False,
            res=True,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.res = res
        assert (outchannel%input_res2net_scale)==0, 'in attention part, the outchannel%input_res2net_scale!=0'
        self.sub_channel = outchannel//input_res2net_scale
        print(self.sub_channel, outchannel, input_res2net_scale)
        self.blocks_att = nn.ModuleList(
            [
                Att_layer(c_1st_conv, input_res2net_scale, SE_neur, self.sub_channel
                )
                for i in range(input_res2net_scale)
            ]
        )
        self.blocks_TDNN = nn.ModuleList(
            [
                TDNNBlock(
                    self.sub_channel, self.sub_channel, kernel_size=3, dilation=1
                )
                for i in range(input_res2net_scale)
            ]
        )
        if self.last_layer:
            self.conv1D = nn.Conv1d(outchannel, outchannel, kernel_size=1)
        self.se_block = SEBlock(outchannel, se_channel, outchannel)

    def forward(self, x):
        y = []

        for i in range(len(x)):
            if i == 0:
                y_i = self.blocks_att[i](x[i])
            else:
                y_i = self.blocks_att[i](x[i]+y_i)
            y_i = self.blocks_TDNN[i](y_i)
            y.append(y_i)
        if self.last_layer:
            y = torch.cat(y, dim=1)
            y = self.conv1D(y)
            if self.res:
                x = torch.cat(x, dim=1)
                y = y + x  
        else:
            if self.res:
                for i in range(len(x)):
                    y[i] = y[i] + x[i]  # res
            y = torch.cat(y, dim=1)
        return y


class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)


class FbankAug(nn.Module):

    def __init__(self, freq_mask_width=(0, 8), time_mask_width=(0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)

        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x



class branch_mfatdnn(torch.nn.Module):
    
    '''
    MFA TDNN
    '''
    def __init__(
        self,
        dims=192,
        activation=torch.nn.ReLU,
        channels=[640, 1024, 1024, 1024, 3072],
        kernel_sizes=[5, 3, 3, 3, 1],
        dilations=[1, 2, 3, 4, 1],
        attention_channels=256,
        res2net_scale=8,
        se_channels=128,
        global_context=True,
        c_1st_conv=32,  # expend channel at the 1st step
        input_res2net_scale=4,  # feature extract in different scales at the 2nd step
        att_channel=640,
        time_att_choice="half",
        merge_choice=1
    ):

        super().__init__()
        print("branch_mfa_tdnn")
        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
                                                 f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=80),
        )

        self.specaug = FbankAug()  # Spec augmentation
        assert len(channels) == len(kernel_sizes)
        assert len(channels) == len(dilations)
        self.channels = channels
        self.blocks = nn.ModuleList()
        self.att_channel = att_channel
        # network
        self.first_conv = nn.Sequential(
            Conv2D_Basic_Block(in_channels=1, out_channels=c_1st_conv, kernel_size=(3,3), padding=(1,1), stride=(2,1)),
            Conv2D_Basic_Block(in_channels=c_1st_conv, out_channels=c_1st_conv, kernel_size=(3, 3), padding=(1, 1), stride=(2, 1))
        )  # first step. expend feature channels
        # 2nd step. time_attention
        self.time_attention = MutiScaleTimeAttention(time_att_choice=time_att_choice)
        self.res2conv2D = nn.Sequential(
            Res2NetBlock(c_1st_conv, c_1st_conv, scale=input_res2net_scale, dtype='Conv2D'),
        ) 
        self.freq_att_TDNN = nn.Sequential(
            Att_Block(
                c_1st_conv,
                outchannel=self.att_channel,
                input_res2net_scale=input_res2net_scale,
                SE_neur=int(20*c_1st_conv/input_res2net_scale//5),
                se_channel=se_channels,
                last_layer=True,
                res=True,
                ),
        )  # 3rd step. Att-TDNN

        self.merge = merge(choice=merge_choice)

        # ----------------
        # SE-Res2Net layers
        for i in range(1, len(channels) - 1):
            print(i, channels[i - 1],channels[i])
            self.blocks.append(
                SERes2NetBlock(
                    channels[i - 1],
                    channels[i],
                    res2net_scale=res2net_scale,
                    se_channels=se_channels,
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                    activation=activation,

                )
            )

        # Multi-layer feature aggregation
        self.mfa = TDNNBlock(
            channels[-1],
            channels[-1],
            kernel_sizes[-1],
            dilations[-1],
            activation,
        )

        # Attentive Statistical Pooling
        self.asp = AttentiveStatisticsPooling(
            channels[-1],
            attention_channels=attention_channels,
            global_context=global_context,
        )
        self.asp_bn = BatchNorm1d(input_size=channels[-1] * 2)

        # Final linear transformation
        self.fc = Conv1d(
            in_channels=channels[-1] * 2,
            out_channels=dims,
            kernel_size=1,
        )

    def forward(self, x, aug=True, lengths=None):
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            x = x.log()
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug:
                x = self.specaug(x)

        # shape:batch_size*80*201
        x = x.unsqueeze(dim=1)
        # batch_size 1 80 201
        x = self.first_conv(x)
        # batch_size 32 20 201

        branch = x
        branch = self.time_attention.forward(branch)
        # batch_size 32 20 201

        residual = torch.flatten(x, 1, 2)  
        # batch_size 640 201

        x = self.res2conv2D(x)  # list: batch_size c_1st/scale*20 201
        x = self.freq_att_TDNN(x)  # list: batch_size c_1st/scale*20 201
        # batch_size 640 201
        x = x.view(-1, 32, 20, x.size(-1))
        # batch_size 32 20 201

        x = self.merge(branch, x)
        # batch_size 640 201
        x = residual+x

        xl = []
        for layer in self.blocks:
            try:
                x = layer(x, lengths=lengths)
                # batch_size 1024 202
            except TypeError:
                x = layer(x)
            xl.append(x)

        # Multi-layer feature aggregation
        x = torch.cat(xl[0:], dim=1)
        # batch_size 3072 202
        x = self.mfa(x)
        # batch_size 3072 202
        # Attentive Statistical Pooling
        x = self.asp(x, lengths=lengths)
        # batch_size 6144 1
        x = self.asp_bn(x)
        # batch_size 6144 1
        # Final linear transformation
        x = self.fc(x)
        # batch_size 192 1
        x = x.squeeze(dim=2)
        # x = x.transpose(1, 2)
        # print("x.shape5:")
        # print(x.shape)
        return x



