from collections import OrderedDict

import segmentation_models_pytorch as smp
from torch.nn import init
from torch.utils.checkpoint import checkpoint

from source.cbam import CBAM
from source.compact_bilinear_pooling import CompactBilinearPooling
from source.unet_parts import *


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=False)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class Encoder(nn.Module):
    def __init__(self, in_channels, bilinear=True, use_bn=False):
        super(Encoder, self).__init__()
        self.n_channels = in_channels
        self.bilinear = bilinear
        self.inc = (DoubleConv(in_channels, 64))
        self.down1 = (Down(64, 128, use_bn=use_bn))
        self.down2 = (Down(128, 256, use_bn=use_bn))
        self.down3 = (Down(256, 512, use_bn=use_bn))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, use_bn=True):
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        # nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=False, use_cbam=True, use_res=True):
        super(Bottleneck, self).__init__()
        self.use_res = use_res
        m = OrderedDict()
        m['conv1'] = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        if use_norm:
            m['bc1'] = nn.BatchNorm2d(out_channels)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        if use_norm:
            m['bc2'] = nn.BatchNorm2d(out_channels)
        m['relu2'] = nn.ReLU(inplace=True)
        m['conv3'] = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.group1 = nn.Sequential(m)
        self.relu = nn.ReLU(inplace=True)
        for name, module in self.group1.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x1 = self.group1(x)
        if self.use_res:
            x = self.relu(x1 + x)
        return x


class Fusion_Blocks_Plus(nn.Module):
    def __init__(self, in_feat, out_feat, cuda=True):
        super(Fusion_Blocks_Plus, self).__init__()
        self.cbp = CompactBilinearPooling(in_feat // 2, in_feat // 2, out_feat, sum_pool=False, cuda=cuda)
        self.cbam1 = CBAM(in_feat, no_spatial=False)
        self.cbam2 = CBAM(in_feat, no_spatial=False)
        self.conv1 = conv1x1(in_feat, out_feat)
        self.conv2 = conv1x1(in_feat, out_feat)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.Hardswish(inplace=True)
        init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.cbam1(x)
        x = self.conv1(x)
        x = self.relu(x)
        _x = self.cbp(x1, x2).permute(0, 3, 1, 2)
        x = torch.cat((x, _x), dim=1)
        x = self.cbam2(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x


class Fusion_Block(nn.Module):
    def __init__(self, in_feat, out_feat, cuda=True):
        super(Fusion_Block, self).__init__()
        self.conv3x3 = conv3x3(in_feat, in_feat)
        self.bn = nn.BatchNorm2d(in_feat)
        self.cbam = CBAM(in_feat, no_spatial=False)
        self.conv1x1 = conv1x1(in_feat, out_feat)
        self.relu = nn.Hardswish(inplace=True)
        init.kaiming_normal_(self.conv3x3.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.conv1x1.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.conv3x3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.cbam(x)
        x = self.conv1x1(x)
        x = self.relu(x)
        return x


class Translation(nn.Module):
    def __init__(self, in_ch, feat=32, num_block=18, use_bn=True):
        super(Translation, self).__init__()
        self.conv_in = conv3x3(in_ch, feat)
        self.res_blocks = nn.Sequential(
            *[BasicBlock(feat, feat, use_bn=use_bn) for _ in range(num_block)])

    def forward(self, x):
        x = self.conv_in(x)
        x = self.res_blocks(x)
        return x


class UAF(nn.Module):
    def __init__(self,
                 in_channels_sar=1,
                 in_channels_single=1,
                 classes=8,
                 ensemble_num=2,
                 bilinear=True,
                 cuda=True):
        super(UAF, self).__init__()
        self.soft_trans = Translation(in_ch=in_channels_single * ensemble_num, num_block=18, feat=64, use_bn=False)
        self.sar_trans = Translation(in_ch=in_channels_sar, num_block=18, feat=64, use_bn=True)  # simply resnet18
        self.encoder_sar = Encoder(64, bilinear, use_bn=True)  # official Unet Encoder
        self.encoder_s2 = Encoder(64, bilinear, use_bn=False)  # official Unet Encoder
        factor = 2 if bilinear else 1
        self.fb1 = Fusion_Blocks_Plus(128, 64, cuda=cuda)
        self.fb2 = Fusion_Blocks_Plus(256, 128, cuda=cuda)
        self.fb3 = Fusion_Blocks_Plus(512, 256, cuda=cuda)
        self.fb4 = Fusion_Blocks_Plus(1024, 512, cuda=cuda)
        self.fb5 = Fusion_Blocks_Plus(2048 // factor, 1024 // factor, cuda=cuda)
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64 // factor, bilinear))

        ch = 64 // factor + 128 // factor + 256 // factor + 512 // factor
        self.logit = nn.Sequential(
            nn.Conv2d(ch, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, classes, kernel_size=1, padding=0),
        )

    def forward(self, x1, x2):
        # Encode Feat
        x1 = checkpoint(self.soft_trans, x1)
        x2 = checkpoint(self.sar_trans, x2)

        # Encoder
        x11, x12, x13, x14, x15 = checkpoint(self.encoder_s2, x1)
        x21, x22, x23, x24, x25 = checkpoint(self.encoder_sar, x2)

        # Feat Fusion
        x1 = self.fb1(x11, x21)
        x2 = self.fb2(x12, x22)
        x3 = self.fb3(x13, x23)
        x4 = self.fb4(x14, x24)
        x5 = self.fb5(x15, x25)

        # decoder
        d4 = self.up1(x5, x4)
        d3 = self.up2(d4, x3)
        d2 = self.up3(d3, x2)
        d1 = self.up4(d2, x1)

        # seg out
        f = torch.cat((
            d1,
            F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=False),
            F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=False),
            F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=False),
        ), 1)

        f = F.dropout2d(f, p=0.50)
        logit = self.logit(f)
        return logit


class Efficient_UAF(nn.Module):
    def __init__(self,
                 encoder_name='efficientnet-b4',
                 in_channels_sar=1,
                 in_channels_single=1,
                 classes=8,
                 ensemble_num=2,
                 bilinear=True,
                 cuda=True):
        super(Efficient_UAF, self).__init__()
        self.encoder_sar = smp.encoders.get_encoder(
            name=encoder_name,
            in_channels=in_channels_sar,
            depth=5,
            weights='imagenet'
        )
        self.encoder_s2 = smp.encoders.get_encoder(
            name=encoder_name,
            in_channels=in_channels_single * ensemble_num,
            depth=5,
            weights='imagenet'
        )
        ch_list = self.encoder_s2.out_channels
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(in_channels_sar, 64, kernel_size=3, padding=1),
        #     nn.ELU(inplace=True),
        # )
        self.fb1 = Fusion_Blocks_Plus(ch_list[1] * 2, ch_list[1], cuda=cuda)
        self.fb2 = Fusion_Blocks_Plus(ch_list[2] * 2, ch_list[2], cuda=cuda)
        self.fb3 = Fusion_Blocks_Plus(ch_list[3] * 2, ch_list[3], cuda=cuda)
        self.fb4 = Fusion_Blocks_Plus(ch_list[4] * 2, ch_list[4], cuda=cuda)
        # self.fb5 = Fusion_Blocks_Plus(ch_list[5] * 2, ch_list[5], cuda=cuda)
        self.fb5 = Fusion_Blocks_Plus(ch_list[5] * 2, ch_list[5], cuda=cuda)
        # self.aspp = BasicRFB(ch_list[5] * 2, ch_list[5], map_reduce=8)
        # self.center = nn.Sequential(
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        # )

        self.up1 = (Up(ch_list[5] + ch_list[4], 256, bilinear))
        self.up2 = (Up(256 + ch_list[3], 128, bilinear))
        self.up3 = (Up(128 + ch_list[2], 64, bilinear))
        self.up4 = (Up(64 + ch_list[1], 64, bilinear))

        ch = 256 + 128 + 64 + 64
        self.logit = nn.Sequential(
            nn.Conv2d(ch, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, classes, kernel_size=1, padding=0),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

    def forward(self, x1, x2):
        x1, x11, x12, x13, x14, x15 = checkpoint(self.encoder_s2, x1)
        x2, x21, x22, x23, x24, x25 = checkpoint(self.encoder_sar, x2)

        # Feat
        x1 = self.fb1(x11, x21)
        x2 = self.fb2(x12, x22)
        x3 = self.fb3(x13, x23)
        x4 = self.fb4(x14, x24)
        x5 = self.fb5(x15, x25)
        # x5 = self.aspp(torch.cat((x15, x25), dim=1))

        # decoder
        d4 = self.up1(x5, x4)
        d3 = self.up2(d4, x3)
        d2 = self.up3(d3, x2)
        d1 = self.up4(d2, x1)

        # seg out
        f = torch.cat((
            d1,
            F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=False),
            F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=False),
            F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=False),
        ), 1)

        f = F.dropout2d(f, p=0.50)
        logit = self.logit(f)
        return logit


if __name__ == '__main__':
    bs = 1
    nc = 8 + 1
    size = 256
    sar_array = torch.randn(bs, 1, size, size)
    tensor_array1 = torch.randn(bs, 1, size, size)
    tensor_array2 = torch.randn(bs, 1, size, size)
    tensor_array3 = torch.randn(bs, 1, size, size)
    tensor_array4 = torch.randn(bs, 1, size, size)
    tensor_array5 = torch.randn(bs, 1, size, size)
    tensor_stack = torch.cat([tensor_array1, tensor_array2, tensor_array3, tensor_array4, tensor_array5], dim=1)
    model = Efficient_UAF(in_channels_sar=1,
                          in_channels_single=1,
                          ensemble_num=5,
                          encoder_name='tu-convnext_small',
                          classes=nc)
    model.eval().cuda()
    result = model(tensor_stack.cuda(), sar_array.cuda())
    print(result.shape)
    # torch.save(model.state_dict(), "test.pth")
