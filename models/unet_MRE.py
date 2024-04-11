import torch.nn.functional as F
import torch
import torch.nn as nn
import glob
import os
import numpy as np
import cv2


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


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=(1, 1), stride=1)
            )

        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        cat_x = torch.cat((x1, x2), 1)
        output = self.conv(cat_x)
        return output


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UpOut(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=(1, 1), stride=1)
            )

        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, out_channels, kernel_size=kernel_size, stride=stride)

        self.conv = DoubleConv(in_channels, in_channels // 2)

    def forward(self, x1):
        x1_1 = self.conv(x1)
        output = self.up(x1_1)
        return output


class UpOutBoth(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=(1, 1), stride=1)
            )

        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, out_channels, kernel_size=kernel_size, stride=stride)

        self.conv1 = DoubleConv(in_channels, in_channels // 2)
        self.conv2 = nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)

    def forward(self, x1):
        x1_1 = self.conv1(x1)
        output1 = self.up(x1_1)
        output2 = self.conv2(x1_1)
        return output1, output2


'''
    Different combinations of losses in U-Net with MRE. 
    Single Loss which means we only use the top layer prediction-label pair to compute losses.
'''


class UNetMRE_Single_Loss(nn.Module):
    def __init__(self, n_channels, n_classes=1, bilinear=False):
        super(UNetMRE_Single_Loss, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # encoder
        self.inc = DoubleConv(n_channels, 64)
        self.inc1 = DoubleConv(n_channels, 128)
        self.inc2 = DoubleConv(n_channels, 256)
        self.inc3 = DoubleConv(n_channels, 512)
        self.inc4 = DoubleConv(n_channels, 1024)
        self.inc5 = DoubleConv(256, 128)
        self.inc6 = DoubleConv(n_channels, 128)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # decoder
        self.up1 = Up(2048, 1024, bilinear)
        self.up2 = Up(1024, 512, bilinear)
        self.up3 = Up(512, 256, bilinear)
        self.up4 = Up(256, 128, bilinear)
        self.out = OutConv(128, n_classes)

    def forward(self, x, cropped_imgs):
        # encoder
        y1_feature = self.inc1(cropped_imgs[0])
        y2_feature = self.inc2(cropped_imgs[1])
        y3_feature = self.inc3(cropped_imgs[2])
        y4_feature = self.inc4(cropped_imgs[3])

        x1 = self.inc(x)
        x1_1 = self.inc6(x)

        x2 = self.down1(x1)
        cat_y12 = torch.cat((y1_feature, x2), 1)
        x3 = self.down2(x2)
        cat_y23 = torch.cat((y2_feature, x3), 1)
        x4 = self.down3(x3)
        cat_y34 = torch.cat((y3_feature, x4), 1)
        x5 = self.down4(x4)
        cat_y45 = torch.cat((y4_feature, x5), 1)

        # decoder
        o_4 = self.up1(cat_y45, cat_y34)
        o_3 = self.up2(o_4, cat_y23)
        o_2 = self.up3(o_3, cat_y12)

        o_1 = self.up4(o_2, x1_1)
        o_seg = self.out(o_1)

        if self.n_classes > 1:
            seg = F.softmax(o_seg, dim=1)
            return seg
        elif self.n_classes == 1:
            seg = torch.sigmoid(o_seg)
            return seg


'''
    Different combinations of losses in U-Net with MRE. 
    Upsampling Loss which means each layer output is first up-sampled to 256×256 and then compute losses with the original 256×256 label.
'''


class UNetMRE_Up_Sampling_Loss(nn.Module):
    def __init__(self, n_channels, n_classes=1, bilinear=False):
        super(UNetMRE_Up_Sampling_Loss, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # encoder
        self.inc = DoubleConv(n_channels, 64)
        self.inc1 = DoubleConv(n_channels, 128)

        self.inc2 = DoubleConv(n_channels, 256)
        self.inc3 = DoubleConv(n_channels, 512)
        self.inc4 = DoubleConv(n_channels, 1024)
        self.inc5 = DoubleConv(256, 128)
        self.inc6 = DoubleConv(n_channels, 128)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # decoder
        self.up1 = Up(2048, 1024, bilinear)
        self.up2 = Up(1024, 512, bilinear)
        self.up3 = Up(512, 256, bilinear)
        self.up4 = Up(256, 128, bilinear)
        self.out1 = UpOut(2048, n_classes, 16, 16)
        self.out2 = UpOut(1024, n_classes, 8, 8)
        self.out3 = UpOut(512, n_classes, 4, 4)
        self.out4 = UpOut(256, n_classes, 2, 2)
        self.out5 = OutConv(128, n_classes)

    def forward(self, x, cropped_imgs):
        # encoder
        y1_feature = self.inc1(cropped_imgs[0])
        y2_feature = self.inc2(cropped_imgs[1])
        y3_feature = self.inc3(cropped_imgs[2])
        y4_feature = self.inc4(cropped_imgs[3])

        x1 = self.inc(x)
        x1_1 = self.inc6(x)

        x2 = self.down1(x1)
        cat_y12 = torch.cat((y1_feature, x2), 1)
        x3 = self.down2(x2)
        cat_y23 = torch.cat((y2_feature, x3), 1)
        x4 = self.down3(x3)
        cat_y34 = torch.cat((y3_feature, x4), 1)
        x5 = self.down4(x4)
        cat_y45 = torch.cat((y4_feature, x5), 1)

        # decoder
        o_seg_1 = self.out1(cat_y45)
        o_4 = self.up1(cat_y45, cat_y34)
        o_seg_2 = self.out2(o_4)
        o_3 = self.up2(o_4, cat_y23)
        o_seg_3 = self.out3(o_3)
        o_2 = self.up3(o_3, cat_y12)
        o_seg_4 = self.out4(o_2)

        o_1 = self.up4(o_2, x1_1)
        o_seg5 = self.out5(o_1)
        o_segs = [o_seg5, o_seg_4, o_seg_3, o_seg_2, o_seg_1]
        segs = []
        if self.n_classes > 1:
            seg = F.softmax(o_seg5, dim=1)
            return seg
        elif self.n_classes == 1:
            for i in range(len(o_segs)):
                seg_now = torch.sigmoid(o_segs[i])
                segs.append(seg_now)
            return segs


'''
    Different combinations of losses in U-Net with MRE. 
    Multi-layer Loss which means we independently compute losses in different layers. 
    The labels of each layer are randomly cropped from the original label, which is the same as we cropped the input images.
'''


class UNetMRE_Multi_Layer_Loss(nn.Module):
    def __init__(self, n_channels, n_classes=1, bilinear=False):
        super(UNetMRE_Multi_Layer_Loss, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # encoder
        self.inc = DoubleConv(n_channels, 64)
        self.inc1 = DoubleConv(n_channels, 128)

        self.inc2 = DoubleConv(n_channels, 256)
        self.inc3 = DoubleConv(n_channels, 512)
        self.inc4 = DoubleConv(n_channels, 1024)
        self.inc5 = DoubleConv(256, 128)
        self.inc6 = DoubleConv(n_channels, 128)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # decoder
        self.up1 = Up(2048, 1024, bilinear)
        self.up2 = Up(1024, 512, bilinear)
        self.up3 = Up(512, 256, bilinear)
        self.up4 = Up(256, 128, bilinear)
        self.out1 = OutConv(2048, n_classes)
        self.out2 = OutConv(1024, n_classes)
        self.out3 = OutConv(512, n_classes)
        self.out4 = OutConv(256, n_classes)
        self.out5 = OutConv(128, n_classes)

    def forward(self, x, cropped_imgs):
        # encoder
        y1_feature = self.inc1(cropped_imgs[0])
        y2_feature = self.inc2(cropped_imgs[1])
        y3_feature = self.inc3(cropped_imgs[2])
        y4_feature = self.inc4(cropped_imgs[3])

        x1 = self.inc(x)
        x1_1 = self.inc6(x)

        x2 = self.down1(x1)
        cat_y12 = torch.cat((y1_feature, x2), 1)
        x3 = self.down2(x2)
        cat_y23 = torch.cat((y2_feature, x3), 1)
        x4 = self.down3(x3)
        cat_y34 = torch.cat((y3_feature, x4), 1)
        x5 = self.down4(x4)
        cat_y45 = torch.cat((y4_feature, x5), 1)

        # decoder

        o_seg_1 = self.out1(cat_y45)
        o_4 = self.up1(cat_y45, cat_y34)
        o_seg_2 = self.out2(o_4)
        o_3 = self.up2(o_4, cat_y23)
        o_seg_3 = self.out3(o_3)
        o_2 = self.up3(o_3, cat_y12)
        o_seg_4 = self.out4(o_2)

        o_1 = self.up4(o_2, x1_1)
        o_seg5 = self.out5(o_1)
        o_segs = [o_seg5, o_seg_4, o_seg_3, o_seg_2, o_seg_1]
        segs = []
        if self.n_classes > 1:
            seg = F.softmax(o_seg5, dim=1)
            return seg
        elif self.n_classes == 1:
            for i in range(len(o_segs)):
                seg_now = torch.sigmoid(o_segs[i])
                segs.append(seg_now)
            return segs


'''
    Different combinations of losses in U-Net with MRE.
    The Hierarchical Fusing Loss used in UNet&MRE is the combinations of Upsampling Loss and Multi-layer Loss.
'''


class UNetMRE_Hierarchical_Fusing_Loss(nn.Module):
    def __init__(self, n_channels, n_classes=1, bilinear=False):
        super(UNetMRE_Hierarchical_Fusing_Loss, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # encoder
        self.inc = DoubleConv(n_channels, 64)
        self.inc1 = DoubleConv(n_channels, 128)
        self.inc2 = DoubleConv(n_channels, 256)
        self.inc3 = DoubleConv(n_channels, 512)
        self.inc4 = DoubleConv(n_channels, 1024)
        self.inc5 = DoubleConv(256, 128)
        self.inc6 = DoubleConv(n_channels, 128)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # decoder
        self.up1 = Up(2048, 1024, bilinear)
        self.up2 = Up(1024, 512, bilinear)
        self.up3 = Up(512, 256, bilinear)
        self.up4 = Up(256, 128, bilinear)

        self.out1 = UpOutBoth(2048, n_classes, 16, 16)
        self.out2 = UpOutBoth(1024, n_classes, 8, 8)
        self.out3 = UpOutBoth(512, n_classes, 4, 4)
        self.out4 = UpOutBoth(256, n_classes, 2, 2)
        self.out5 = OutConv(128, n_classes)

    def forward(self, x, cropped_imgs):
        # encoder
        y1_feature = self.inc1(cropped_imgs[0])
        y2_feature = self.inc2(cropped_imgs[1])
        y3_feature = self.inc3(cropped_imgs[2])
        y4_feature = self.inc4(cropped_imgs[3])

        x1 = self.inc(x)
        x1_1 = self.inc6(x)

        x2 = self.down1(x1)
        cat_y12 = torch.cat((y1_feature, x2), 1)
        x3 = self.down2(x2)
        cat_y23 = torch.cat((y2_feature, x3), 1)
        x4 = self.down3(x3)
        cat_y34 = torch.cat((y3_feature, x4), 1)
        x5 = self.down4(x4)
        cat_y45 = torch.cat((y4_feature, x5), 1)

        # decoder
        o_seg_1_256, o_seg_1 = self.out1(cat_y45)
        o_4 = self.up1(cat_y45, cat_y34)
        o_seg_2_256, o_seg_2 = self.out2(o_4)
        o_3 = self.up2(o_4, cat_y23)
        o_seg_3_256, o_seg_3 = self.out3(o_3)
        o_2 = self.up3(o_3, cat_y12)
        o_seg_4_256, o_seg_4 = self.out4(o_2)

        o_1 = self.up4(o_2, x1_1)
        o_seg5 = self.out5(o_1)
        o_segs = [o_seg5, o_seg_4_256, o_seg_3_256, o_seg_2_256, o_seg_1_256, o_seg_4, o_seg_3, o_seg_2, o_seg_1]
        segs = []
        if self.n_classes > 1:
            seg = F.softmax(o_seg5, dim=1)
            return seg
        elif self.n_classes == 1:
            for i in range(len(o_segs)):
                seg_now = torch.sigmoid(o_segs[i])
                segs.append(seg_now)
            return segs
