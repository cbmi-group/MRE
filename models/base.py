import torch
import torch.nn as nn
import torch.nn.functional as F


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


class MS_PVT_SegModel_SingleLoss(torch.nn.Module):
    def __init__(self, n_classes=1, bilinear=False, isUpsample=False):
        super(MS_PVT_SegModel_SingleLoss, self).__init__()
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.isUpsample = isUpsample

        # decoder
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.out5 = OutConv(64, n_classes)

    def upsample_features(self, features):
        up_features = []
        for feature in features:
            feature = self.up1(feature)
            up_features.append(feature)

        return up_features

    def forward(self, x, cropped_imgs):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        # self.check_input_shape(x)
        features = self.encoder1(x)
        multi_features = self.encoder2(x, cropped_imgs)

        if self.isUpsample:
            features = self.upsample_features(features)

        cat_y12 = torch.cat((features[0], multi_features[1]), 1)
        cat_y23 = torch.cat((features[1], multi_features[2]), 1)
        cat_y34 = torch.cat((features[2], multi_features[3]), 1)
        cat_y45 = torch.cat((features[3], multi_features[4]), 1)

        o_4 = self.up1(cat_y45, cat_y34)
        o_3 = self.up2(o_4, cat_y23)
        o_2 = self.up3(o_3, cat_y12)
        o_1 = self.up4(o_2, multi_features[0])
        o_seg5 = self.out5(o_1)
        if self.n_classes > 1:
            seg = F.softmax(o_seg5, dim=1)
            return seg
        elif self.n_classes == 1:
            seg = torch.sigmoid(o_seg5)
            return seg

    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()
        x = self.forward(x)
        return x


class MS_PVT_SegModel_Up_Sampling_Loss(torch.nn.Module):
    def __init__(self, n_classes=1, bilinear=False, isUpsample=False):
        super(MS_PVT_SegModel_Up_Sampling_Loss, self).__init__()
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.isUpsample = isUpsample

        # decoder
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.out1 = UpOut(1024, n_classes, 16, 16)
        self.out2 = UpOut(512, n_classes, 8, 8)
        self.out3 = UpOut(256, n_classes, 4, 4)
        self.out4 = UpOut(128, n_classes, 2, 2)
        self.out5 = OutConv(64, n_classes)

    def upsample_features(self, features):
        up_features = []
        for feature in features:
            feature = self.up1(feature)
            up_features.append(feature)

        return up_features

    def forward(self, x, cropped_imgs):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        # self.check_input_shape(x)
        features = self.encoder1(x)
        multi_features = self.encoder2(x, cropped_imgs)

        if self.isUpsample:
            features = self.upsample_features(features)

        cat_y12 = torch.cat((features[0], multi_features[1]), 1)
        cat_y23 = torch.cat((features[1], multi_features[2]), 1)
        cat_y34 = torch.cat((features[2], multi_features[3]), 1)
        cat_y45 = torch.cat((features[3], multi_features[4]), 1)
        # decoder
        o_seg_1 = self.out1(cat_y45)
        o_4 = self.up1(cat_y45, cat_y34)
        o_seg_2 = self.out2(o_4)
        o_3 = self.up2(o_4, cat_y23)
        o_seg_3 = self.out3(o_3)
        o_2 = self.up3(o_3, cat_y12)
        o_seg_4 = self.out4(o_2)
        o_1 = self.up4(o_2, multi_features[0])
        o_seg5 = self.out5(o_1)
        o_segs = [o_seg5, o_seg_4, o_seg_3, o_seg_2, o_seg_1]
        segs = []
        for i in range(len(o_segs)):
            if self.n_classes > 1:
                seg_now = F.softmax(o_seg5, dim=1)
            elif self.n_classes == 1:
                seg_now = torch.sigmoid(o_segs[i])
            segs.append(seg_now)
        return segs

    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()
        x = self.forward(x)
        return x


class MS_PVT_SegModel_Multi_Layer_Loss(torch.nn.Module):
    def __init__(self, n_classes=1, bilinear=False, isUpsample=False):
        super(MS_PVT_SegModel_Multi_Layer_Loss, self).__init__()
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.isUpsample = isUpsample

        # decoder
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.out1 = OutConv(1024, n_classes)
        self.out2 = OutConv(512, n_classes)
        self.out3 = OutConv(256, n_classes)
        self.out4 = OutConv(128, n_classes)
        self.out5 = OutConv(64, n_classes)

    def upsample_features(self, features):
        up_features = []
        for feature in features:
            feature = self.up1(feature)
            up_features.append(feature)

        return up_features

    def forward(self, x, cropped_imgs):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        # self.check_input_shape(x)
        features = self.encoder1(x)
        multi_features = self.encoder2(x, cropped_imgs)

        if self.isUpsample:
            features = self.upsample_features(features)

        cat_y12 = torch.cat((features[0], multi_features[1]), 1)
        cat_y23 = torch.cat((features[1], multi_features[2]), 1)
        cat_y34 = torch.cat((features[2], multi_features[3]), 1)
        cat_y45 = torch.cat((features[3], multi_features[4]), 1)


        # decoder
        o_seg_1 = self.out1(cat_y45)
        o_4 = self.up1(cat_y45, cat_y34)
        o_seg_2 = self.out2(o_4)
        o_3 = self.up2(o_4, cat_y23)
        o_seg_3 = self.out3(o_3)
        o_2 = self.up3(o_3, cat_y12)
        o_seg_4 = self.out4(o_2)
        o_1 = self.up4(o_2, multi_features[0])
        o_seg5 = self.out5(o_1)
        o_segs = [o_seg5, o_seg_4, o_seg_3, o_seg_2, o_seg_1]
        segs = []
        for i in range(len(o_segs)):
            if self.n_classes > 1:
                seg_now = F.softmax(o_seg5, dim=1)
            elif self.n_classes == 1:
                seg_now = torch.sigmoid(o_segs[i])
            segs.append(seg_now)
        return segs

    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()
        x = self.forward(x)
        return x


class MS_PVT_SegModel_Hierarchical_Fusing_Loss(torch.nn.Module):
    def __init__(self, n_classes=1, bilinear=False, isUpsample=False):
        super(MS_PVT_SegModel_Hierarchical_Fusing_Loss, self).__init__()
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.isUpsample = isUpsample

        # decoder
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.out1 = UpOutBoth(1024, n_classes, 16, 16)
        self.out2 = UpOutBoth(512, n_classes, 8, 8)
        self.out3 = UpOutBoth(256, n_classes, 4, 4)
        self.out4 = UpOutBoth(128, n_classes, 2, 2)
        self.out5 = OutConv(64, n_classes)

    def upsample_features(self, features):
        up_features = []
        for feature in features:
            feature = self.up1(feature)
            up_features.append(feature)

        return up_features

    def forward(self, x, cropped_imgs):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        # self.check_input_shape(x)
        features = self.encoder1(x)
        multi_features = self.encoder2(x, cropped_imgs)

        if self.isUpsample:
            features = self.upsample_features(features)

        cat_y12 = torch.cat((features[0], multi_features[1]), 1)
        cat_y23 = torch.cat((features[1], multi_features[2]), 1)
        cat_y34 = torch.cat((features[2], multi_features[3]), 1)
        cat_y45 = torch.cat((features[3], multi_features[4]), 1)

        # decoder
        o_seg_1_256, o_seg_1 = self.out1(cat_y45)
        o_4 = self.up1(cat_y45, cat_y34)
        o_seg_2_256, o_seg_2 = self.out2(o_4)
        o_3 = self.up2(o_4, cat_y23)
        o_seg_3_256, o_seg_3 = self.out3(o_3)
        o_2 = self.up3(o_3, cat_y12)
        o_seg_4_256, o_seg_4 = self.out4(o_2)
        o_1 = self.up4(o_2, multi_features[0])
        o_seg5 = self.out5(o_1)
        o_segs = [o_seg5, o_seg_4_256, o_seg_3_256, o_seg_2_256, o_seg_1_256, o_seg_4, o_seg_3, o_seg_2, o_seg_1]
        segs = []
        for i in range(len(o_segs)):
            if self.n_classes > 1:
                seg_now = F.softmax(o_seg5, dim=1)
            elif self.n_classes == 1:
                seg_now = torch.sigmoid(o_segs[i])
            segs.append(seg_now)
        return segs

    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()
        x = self.forward(x)
        return x
