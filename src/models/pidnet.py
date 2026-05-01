"""
PIDNet model definition
From the original implementation: github.com/XuJiacong/PIDNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.pidnet import pidnet_loss


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=0.1)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)


class SegmentHead(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(SegmentHead, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, momentum=0.1)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(interplanes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out, size=[height, width], mode="bilinear", align_corners=False)

        return out


class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        super(DAPPM, self).__init__()
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
            BatchNorm(inplanes, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
            BatchNorm(inplanes, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
            BatchNorm(inplanes, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            BatchNorm(inplanes, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale0 = nn.Sequential(
            BatchNorm(inplanes, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.process1 = nn.Sequential(
            BatchNorm(branch_planes, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process2 = nn.Sequential(
            BatchNorm(branch_planes, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process3 = nn.Sequential(
            BatchNorm(branch_planes, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process4 = nn.Sequential(
            BatchNorm(branch_planes, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.compression = nn.Sequential(
            BatchNorm(branch_planes * 5, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            BatchNorm(inplanes, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(
            self.process1(
                (
                    F.interpolate(
                        self.scale1(x),
                        size=[height, width],
                        mode="bilinear",
                        align_corners=False,
                    )
                    + x_list[0]
                )
            )
        )
        x_list.append(
            (
                self.process2(
                    (
                        F.interpolate(
                            self.scale2(x),
                            size=[height, width],
                            mode="bilinear",
                            align_corners=False,
                        )
                        + x_list[1]
                    )
                )
            )
        )
        x_list.append(
            self.process3(
                (
                    F.interpolate(
                        self.scale3(x),
                        size=[height, width],
                        mode="bilinear",
                        align_corners=False,
                    )
                    + x_list[2]
                )
            )
        )
        x_list.append(
            self.process4(
                (
                    F.interpolate(
                        self.scale4(x),
                        size=[height, width],
                        mode="bilinear",
                        align_corners=False,
                    )
                    + x_list[3]
                )
            )
        )

        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out


class PAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        super(PAPPM, self).__init__()
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
            BatchNorm(inplanes, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
            BatchNorm(inplanes, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
            BatchNorm(inplanes, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            BatchNorm(inplanes, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )

        self.scale0 = nn.Sequential(
            BatchNorm(inplanes, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )

        self.scale_process = nn.Sequential(
            BatchNorm(branch_planes * 4, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                branch_planes * 4,
                branch_planes * 4,
                kernel_size=3,
                padding=1,
                groups=4,
                bias=False,
            ),
        )

        self.compression = nn.Sequential(
            BatchNorm(branch_planes * 5, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
        )

        self.shortcut = nn.Sequential(
            BatchNorm(inplanes, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        scale_list = []

        x_ = self.scale0(x)
        scale_list.append(
            F.interpolate(
                self.scale1(x),
                size=[height, width],
                mode="bilinear",
                align_corners=False,
            )
            + x_
        )
        scale_list.append(
            F.interpolate(
                self.scale2(x),
                size=[height, width],
                mode="bilinear",
                align_corners=False,
            )
            + x_
        )
        scale_list.append(
            F.interpolate(
                self.scale3(x),
                size=[height, width],
                mode="bilinear",
                align_corners=False,
            )
            + x_
        )
        scale_list.append(
            F.interpolate(
                self.scale4(x),
                size=[height, width],
                mode="bilinear",
                align_corners=False,
            )
            + x_
        )

        scale_out = self.scale_process(torch.cat(scale_list, 1))

        out = self.compression(torch.cat([x_, scale_out], 1)) + self.shortcut(x)
        return out


class PagFM(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        after_relu=False,
        with_channel=False,
        BatchNorm=nn.BatchNorm2d,
    ):
        super(PagFM, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            BatchNorm(mid_channels),
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            BatchNorm(mid_channels),
        )
        if with_channel:
            self.up = nn.Sequential(
                nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
                BatchNorm(in_channels),
            )
        if after_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        input_size = x.size()
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)

        y_q = self.f_y(y)
        y_q = F.interpolate(
            y_q,
            size=[input_size[2], input_size[3]],
            mode="bilinear",
            align_corners=False,
        )
        x_k = self.f_x(x)

        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))

        y = F.interpolate(y, size=[input_size[2], input_size[3]], mode="bilinear", align_corners=False)
        x = (1 - sim_map) * x + sim_map * y

        return x


class LightBag(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(LightBag, self).__init__()
        self.conv_p = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            BatchNorm(out_channels),
        )
        self.conv_i = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            BatchNorm(out_channels),
        )

    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)

        p_add = self.conv_p((1 - edge_att) * i + p)
        i_add = self.conv_i(i + edge_att * p)

        return p_add + i_add


class DDFMv2(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(DDFMv2, self).__init__()
        self.conv_p = nn.Sequential(
            BatchNorm(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            BatchNorm(out_channels),
        )
        self.conv_i = nn.Sequential(
            BatchNorm(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            BatchNorm(out_channels),
        )

    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)

        p_add = self.conv_p((1 - edge_att) * i + p)
        i_add = self.conv_i(i + edge_att * p)

        return p_add + i_add


class Bag(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(Bag, self).__init__()

        self.conv = nn.Sequential(
            BatchNorm(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        return self.conv(edge_att * p + (1 - edge_att) * i)


class PIDNet(nn.Module):

    def __init__(
        self,
        m=2,
        n=3,
        num_classes=19,
        planes=64,
        ppm_planes=96,
        head_planes=128,
        augment=True,
    ):
        super(PIDNet, self).__init__()
        self.augment = augment

        # I Branch
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(planes, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(planes, momentum=0.1),
            nn.ReLU(inplace=True),
        )

        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, m, stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes * 2, planes * 4, n, stride=2)
        self.layer4 = self._make_layer(BasicBlock, planes * 4, planes * 8, n, stride=2)
        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 8, 2, stride=2)

        # P Branch
        self.compression3 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 2, momentum=0.1),
        )

        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 2, momentum=0.1),
        )
        self.pag3 = PagFM(planes * 2, planes)
        self.pag4 = PagFM(planes * 2, planes)

        self.layer3_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer4_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer5_ = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

        # D Branch
        if m == 2:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes)
            self.layer4_d = self._make_layer(Bottleneck, planes, planes, 1)
            self.diff3 = nn.Sequential(
                nn.Conv2d(planes * 4, planes, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(planes, momentum=0.1),
            )
            self.diff4 = nn.Sequential(
                nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(planes * 2, momentum=0.1),
            )
            self.spp = PAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = LightBag(planes * 4, planes * 4)
        else:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.layer4_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.diff3 = nn.Sequential(
                nn.Conv2d(planes * 4, planes * 2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(planes * 2, momentum=0.1),
            )
            self.diff4 = nn.Sequential(
                nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(planes * 2, momentum=0.1),
            )
            self.spp = DAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Bag(planes * 4, planes * 4)

        self.layer5_d = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

        # Prediction Head
        if self.augment:
            self.seghead_p = SegmentHead(planes * 2, head_planes, num_classes)
            self.seghead_d = SegmentHead(planes * 2, planes, 1)

        self.final_layer = SegmentHead(planes * 4, head_planes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.1),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)

    def _make_single_layer(self, block, inplanes, planes, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.1),
            )

        layer = block(inplanes, planes, stride, downsample, no_relu=True)

        return layer

    def forward(self, x):

        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.relu(self.layer2(self.relu(x)))
        x_ = self.layer3_(x)
        x_d = self.layer3_d(x)

        x = self.relu(self.layer3(x))
        x_ = self.pag3(x_, self.compression3(x))
        x_d = x_d + F.interpolate(
            self.diff3(x),
            size=[height_output, width_output],
            mode="bilinear",
            align_corners=False,
        )
        if self.augment:
            temp_p = x_

        x = self.relu(self.layer4(x))
        x_ = self.layer4_(self.relu(x_))
        x_d = self.layer4_d(self.relu(x_d))

        x_ = self.pag4(x_, self.compression4(x))
        x_d = x_d + F.interpolate(
            self.diff4(x),
            size=[height_output, width_output],
            mode="bilinear",
            align_corners=False,
        )
        if self.augment:
            temp_d = x_d

        x_ = self.layer5_(self.relu(x_))
        x_d = self.layer5_d(self.relu(x_d))
        x = F.interpolate(
            self.spp(self.layer5(x)),
            size=[height_output, width_output],
            mode="bilinear",
            align_corners=False,
        )

        x_ = self.final_layer(self.dfm(x_, x, x_d))

        if self.augment:
            x_extra_p = self.seghead_p(temp_p)
            x_extra_d = self.seghead_d(temp_d)
            return [x_extra_p, x_, x_extra_d]
        else:
            return x_


class FullPIDNetModel(nn.Module):

    def __init__(self, model, sem_loss, bd_loss, ignore_index=-1):
        super(FullPIDNetModel, self).__init__()
        self.model = model
        self.sem_loss = sem_loss
        self.bd_loss = bd_loss
        self.ignore_index = ignore_index

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label != self.ignore_index).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc

    def forward(self, inputs, labels, bd_gt, *args, **kwargs):
        outputs = self.model(inputs, *args, **kwargs)

        if labels is None:
            h, w = inputs.size(2), inputs.size(3)
        else:
            h, w = labels.size(1), labels.size(2)

        ph, pw = outputs[0].size(2), outputs[0].size(3)
        if ph != h or pw != w:
            for i in range(len(outputs)):
                outputs[i] = F.interpolate(
                    outputs[i], size=(h, w), mode="bilinear", align_corners=True
                )  # from original configs

        if bd_gt is None:
            return None, outputs, None, None

        acc = self.pixel_acc(outputs[-2], labels)

        loss_s, loss_b, loss_sb = pidnet_loss(outputs, labels, self.sem_loss, self.bd_loss, self.ignore_index, bd_gt)

        loss = loss_s + loss_b + loss_sb

        return torch.unsqueeze(loss, 0), outputs, acc, [loss_s, loss_b, loss_sb]
