import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torch.nn import functional as F
from torchvision.ops import FeaturePyramidNetwork

from .resnetv1b import resnet18_v1b, resnet34_v1b, resnet50_v1s, resnet101_v1s, resnet152_v1s

class FPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(
        self, in_channels_list, out_channels, conv_block, top_blocks=None):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed
            out_channels (int): number of channels of the FPN representation
            top_blocks (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list
        """
        super(FPN, self).__init__()
        self.inner_blocks = []
        self.layer_blocks = []
        for idx, in_channels in enumerate(in_channels_list, 1):
            if idx < 0:
                continue
            inner_block = "fpn_inner{}".format(idx)
            layer_block = "fpn_layer{}".format(idx)

            if in_channels == 0:
                continue
            inner_block_module = conv_block(in_channels, out_channels, 1)
            layer_block_module = conv_block(out_channels, out_channels, 3, 1)
            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        self.top_blocks = top_blocks


        self.to_discard = 0

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        results = []
        results.append(getattr(self, self.layer_blocks[-1])(last_inner))

        for feature, inner_block, layer_block in zip(
            x[self.to_discard:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            if not inner_block:
                continue
            inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="bilinear", align_corners=False)
            inner_lateral = getattr(self, inner_block)(feature)
            last_inner = inner_lateral + inner_top_down
            results.insert(0, getattr(self, layer_block)(last_inner))

        return tuple(results)


class ResNetBackbone(torch.nn.Module):
    def __init__(self, backbone='resnet50', pretrained_base=True, dilated=True, **kwargs):
        super(ResNetBackbone, self).__init__()

        if backbone == 'resnet18':
            pretrained = resnet18_v1b(pretrained=pretrained_base, dilated=dilated, **kwargs)
        elif backbone == 'resnet34':
            pretrained = resnet34_v1b(pretrained=pretrained_base, dilated=dilated, **kwargs)
        elif backbone == 'resnet50':
            pretrained = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        elif backbone == 'resnet101':
            pretrained = resnet101_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
        elif backbone == 'resnet152':
            pretrained = resnet152_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
        else:
            raise RuntimeError(f'unknown backbone: {backbone}')

        self.fuse = kwargs['fuse'] if 'fuse' in kwargs else True
        self.conv1 = pretrained.conv1
        self.bn1 = pretrained.bn1
        self.relu = pretrained.relu
        self.maxpool = pretrained.maxpool
        self.layer1 = pretrained.layer1
        self.layer2 = pretrained.layer2
        self.layer3 = pretrained.layer3
        self.layer4 = pretrained.layer4

        in_channels = [256, 512, 1024, 2048]
        self.fpn = FeaturePyramidNetwork(in_channels, out_channels=256)

        if self.fuse:
            self.refine = nn.ModuleList()
            for channel in in_channels:
                self.refine.append(nn.Conv2d(
                    256, 256, 3, stride=1, padding=1
                ))
        else:
            self.refine=None

        self.cls_head = nn.Sequential(
            nn.Conv2d(256, 256,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1,
                      kernel_size=1, stride=1, padding=1, bias=True)
        )

    def fuse_features(self, fpn_feats):
        if self.refine is None:
            return fpn_feats['f0']

        for i, k in enumerate(fpn_feats.keys()):
            if i == 0:
                x = self.refine[i](fpn_feats[k])
            else:
                x_l = self.refine[i](fpn_feats[k])

                target_h, target_w = x.shape[-2:]
                h, w = x_l.shape[-2:]
                assert target_h % h == 0
                assert target_w % w == 0
                factor_h, factor_w = target_h // h, target_w // w
                assert factor_h == factor_w
                x_l = F.interpolate(x_l, scale_factor=factor_h, mode='bilinear', align_corners=False)
                x = x + x_l

        return x

    def forward(self, x, additional_features=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if additional_features is not None:
            x = x + torch.nn.functional.pad(additional_features,
                                            [0, 0, 0, 0, 0, x.size(1) - additional_features.size(1)],
                                            mode='constant', value=0)
        x = self.maxpool(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        feat_dict = {'f0': c1, 'f1': c2, 'f3': c3, 'f4': c4}
        feats = self.fpn(feat_dict)
        feats = self.fuse_features(feats)

        return [self.cls_head(feats), None, feats]
