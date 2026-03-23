import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class Encoder(nn.Module):
    def __init__(self, input_nc, ngf=64):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(input_nc, ngf, 7, 1, 3), nn.InstanceNorm2d(ngf), nn.ReLU(True))
        self.down1 = nn.Sequential(nn.Conv2d(ngf, ngf * 2, 3, 2, 1), nn.InstanceNorm2d(ngf * 2), nn.ReLU(True))
        self.down2 = nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1), nn.InstanceNorm2d(ngf * 4), nn.ReLU(True))

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        return [f1, f2, f3]


class Decoder(nn.Module):
    def __init__(self, output_nc, ngf=64):
        super(Decoder, self).__init__()
        # Replace ConvTranspose2d with Upsample + Conv2d to avoid checkerboard artifacts
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ngf * 4, ngf * 2, 3, 1, 1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ngf * 4, ngf, 3, 1, 1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        )
        self.final = nn.Sequential(nn.Conv2d(ngf * 2, output_nc, 7, 1, 3), nn.Tanh())

    def forward(self, feat3, feat2, feat1):
        u1 = self.up1(feat3)
        u1 = torch.cat([u1, feat2], dim=1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, feat1], dim=1)
        out = self.final(u2)
        return out


class Corrector(nn.Module):
    def __init__(self, channels, ngf=32):
        super(Corrector, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, ngf, 3, 1, 1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf, 3, 1, 1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return x + self.net(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.conv(scale)
        return x * self.sigmoid(scale)


class HeadCECTGANGenerator(nn.Module):
    """HeadCECT-GAN generator: shared encoder -> residual bottleneck -> two decoders (image, deformation)

    Returns (image_out, flow_out)
    """
    def __init__(self, input_nc=1, output_nc=1, ngf=64, n_res=6):
        super(HeadCECTGANGenerator, self).__init__()
        self.encoder = Encoder(input_nc, ngf=ngf)
        # bottleneck residuals
        blocks = []
        for _ in range(n_res):
            blocks.append(ResidualBlock(ngf * 4))
        self.resblocks = nn.Sequential(*blocks)
        self.se = SEBlock(ngf * 4)
        self.sa = SpatialAttention()

        # decoders
        self.img_decoder = Decoder(output_nc, ngf=ngf)
        # deformation head outputs 2-channel flow at same spatial res as image
        self.flow_decoder = Decoder(2, ngf=ngf)

        # correctors
        self.img_corrector = Corrector(output_nc, ngf=ngf // 2)
        self.flow_corrector = Corrector(2, ngf=ngf // 2)

    def forward(self, x):
        f1, f2, f3 = self.encoder(x)
        b = self.resblocks(f3)
        b = self.se(b)
        b = self.sa(b)
        img = self.img_decoder(b, f2, f1)
        flow = self.flow_decoder(b, f2, f1)

        img = self.img_corrector(img)
        flow = self.flow_corrector(flow)
        
        # Clamp output to [-1, 1] to avoid artifacts during visualization
        img = torch.clamp(img, -1, 1)
        
        return img, flow


class HeadCECTGANDiscriminator(nn.Module):
    def __init__(self, input_nc=1):
        super(HeadCECTGANDiscriminator, self).__init__()
        def conv(in_c, out_c, stride=2):
            return nn.Sequential(spectral_norm(nn.Conv2d(in_c, out_c, 4, stride, 1)), nn.LeakyReLU(0.2, inplace=True))

        self.model = nn.Sequential(
            conv(input_nc, 64, stride=2),
            conv(64, 128, stride=2),
            conv(128, 256, stride=2),
            conv(256, 512, stride=1),
            spectral_norm(nn.Conv2d(512, 1, 4, 1, 1))
        )

    def forward(self, x):
        # return patch map [B,1,H,W]
        return self.model(x)


if __name__ == '__main__':
    g = HeadCECTGANGenerator(1,1)
    d = HeadCECTGANDiscriminator(1)
    x = torch.randn(2,1,256,256)
    img, flow = g(x)
    print(img.shape, flow.shape)
    out = d(img)
    print(out.shape)
