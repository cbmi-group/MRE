import torch.nn as nn


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Multi_Resolution_Encoder(nn.Module):
    def __init__(self, n_channels, bilinear=False):
        super(Multi_Resolution_Encoder, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        # encoder
        self.inc = DoubleConv(n_channels, 64)
        self.inc1 = DoubleConv(n_channels, 64)
        self.inc2 = DoubleConv(n_channels, 128)
        self.inc3 = DoubleConv(n_channels, 256)
        self.inc4 = DoubleConv(n_channels, 512)

    def forward(self, x, cropped_imgs):
        # encoder
        y1_feature = self.inc1(cropped_imgs[0])
        y2_feature = self.inc2(cropped_imgs[1])
        y3_feature = self.inc3(cropped_imgs[2])
        y4_feature = self.inc4(cropped_imgs[3])

        x1 = self.inc(x)
        features = [x1, y1_feature, y2_feature, y3_feature, y4_feature]

        return features
