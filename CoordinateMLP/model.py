import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean([2,3], keepdim=True)
        std = x.std([2,3], keepdim=True) + self.eps
        x = (x - mean) / std
        x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x

class CoordGate(nn.Module):
    def __init__(self, num_in, num_hidden, num_out, cnn_in_channels, cnn_out_channels, stride=1):
        super(CoordGate, self).__init__()
        self.mask_mlp = nn.Sequential(
            nn.Linear(num_in, num_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(num_hidden, num_out),
            nn.Sigmoid()
        )
        self.conv = nn.Conv2d(cnn_in_channels, cnn_out_channels, kernel_size=3, padding=1, stride=stride)

    def forward(self, x, index):
        # x: [batch_size, channels, H, W]
        # index: [batch_size, H, W, num_in]
        batch_size, channels, H, W = x.size()
        index = index.view(batch_size, -1, index.size(-1))  # [batch_size, H*W, num_in]
        mask = self.mask_mlp(index)  # [batch_size, H*W, num_out]
        mask = mask.view(batch_size, -1, H, W)  # [batch_size, num_out, H, W]

        x = self.conv(x)
        if x.size(1) != mask.size(1):
            mask = mask.repeat(1, x.size(1) // mask.size(1), 1, 1)
        x = x * mask
        return x

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class SCA(nn.Module):
    def __init__(self, channels):
        super(SCA, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        y = self.global_pool(x)
        y = self.conv(y)
        return x * y

class NAFBlock(nn.Module):
    def __init__(self, c, num_in, num_hidden):
        super(NAFBlock, self).__init__()
        dw_channel = c * 2

        self.conv1 = CoordGate(num_in, num_hidden, dw_channel, c, dw_channel)
        self.conv2 = CoordGate(num_in, num_hidden, dw_channel, dw_channel, dw_channel)
        self.conv3 = CoordGate(num_in, num_hidden, c, dw_channel // 2, c)

        self.sca = SCA(dw_channel // 2)
        self.sg = SimpleGate()

        ffn_channel = c * 2
        self.conv4 = CoordGate(num_in, num_hidden, ffn_channel, c, ffn_channel)
        self.conv5 = CoordGate(num_in, num_hidden, c, ffn_channel // 2, c)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x, index):
        identity = x

        x = self.norm1(x)

        x = self.conv1(x, index)
        x = self.conv2(x, index)
        x = self.sg(x)
        x = self.sca(x)
        x = self.conv3(x, index)

        y = identity + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x, index)
        x = self.sg(x)
        x = self.conv5(x, index)

        return y + x * self.gamma

class SequentialMultiInput(nn.Sequential):
    def forward(self, x, index):
        for module in self._modules.values():
            x = module(x, index)
        return x

class NAFNet(nn.Module):
    def __init__(self, img_channel=1, width=32, middle_blk_num=1,
                 enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1],
                 num_in=2, num_hidden=64, num_views=9):
        super(NAFNet, self).__init__()

        self.num_views = num_views
        self.width = width

        self.intro = nn.ModuleList([
            CoordGate(num_in, num_hidden, width, img_channel, width)
            for _ in range(num_views)
        ])

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = SequentialMultiInput()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            encoder_layers = []
            for _ in range(num):
                encoder_layers.append(NAFBlock(chan, num_in, num_hidden))
            self.encoders.append(SequentialMultiInput(*encoder_layers))
            self.downs.append(nn.Conv2d(chan, chan * 2, kernel_size=2, stride=2))
            chan *= 2

        middle_layers = []
        for _ in range(middle_blk_num):
            middle_layers.append(NAFBlock(chan, num_in, num_hidden))
        self.middle_blks = SequentialMultiInput(*middle_layers)

        for num in dec_blk_nums:
            self.ups.append(nn.ConvTranspose2d(chan, chan // 2, kernel_size=2, stride=2))
            chan = chan // 2
            decoder_layers = []
            for _ in range(num):
                decoder_layers.append(NAFBlock(chan, num_in, num_hidden))
            self.decoders.append(SequentialMultiInput(*decoder_layers))

        self.ending = CoordGate(num_in, num_hidden, img_channel, chan, img_channel)

    def forward(self, x, index_list):
        # x: [Batch, num_views, 1, H, W]
        # index_list: [Batch, num_views, H, W, 2]

        batch_size, num_views, _, H, W = x.size()
        feats = []
        for i in range(num_views):
            xi = x[:, i, :, :, :]  # [Batch, 1, H, W]
            index_i = index_list[:, i, :, :, :]  # [Batch, H, W, 2]
            xi = self.intro[i](xi, index_i)
            feats.append(xi)  # [Batch, C, H, W]

        x = torch.stack(feats, dim=1)  # [Batch, num_views, C, H, W]
        x, _ = x.max(dim=1)  # [Batch, C, H, W]

        index = index_list.mean(dim=1)  # [Batch, H, W, 2]
        index_pyramid = [index]
        enc_features = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x, index)
            enc_features.append(x)
            x = down(x)
            index = F.interpolate(index.permute(0, 3, 1, 2), scale_factor=0.5, mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            index_pyramid.append(index)

        x = self.middle_blks(x, index)

        for decoder, up, skip, idx in zip(self.decoders, self.ups, reversed(enc_features), reversed(index_pyramid[:-1])):
            x = up(x)
            x = x + skip
            x = decoder(x, idx)

        x = self.ending(x, index_pyramid[0])

        return x


class FPNet(nn.Module):
    def __init__(self):
        super(FPNet, self).__init__()
        self.model = NAFNet(
            img_channel=1,
            width=32,
            middle_blk_num=1,
            enc_blk_nums=[1, 1, 1, 28],
            dec_blk_nums=[1, 1, 1, 1],
            num_in=2,
            num_hidden=64,
            num_views=9
        )
        self.activation = nn.Sigmoid()

    def forward(self, x, index_list):
        # x: [Batch, 9, H, W]
        # index_list: [Batch, 9, H, W, 2]
        x = x.unsqueeze(2)  # [Batch, 9, 1, H, W]
        output = self.model(x, index_list)  # [Batch, 1, H, W]
        output = self.activation(output)
        return output
