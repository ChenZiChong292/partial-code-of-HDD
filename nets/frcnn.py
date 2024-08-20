import torch.nn as nn

from nets.classifier import Resnet50RoIHead, VGG16RoIHead
from nets.resnet import resnet50, resnet101
from nets.rpn import RegionProposalNetwork
from nets.vgg16 import decom_vgg16


class FasterRCNN(nn.Module):
    def __init__(self, num_classes,
                 mode="training",
                 feat_stride=16,
                 anchor_scales=[8, 16, 32],
                 ratios=[0.5, 1, 2],
                 backbone='vgg',
                 pretrained=False):
        super(FasterRCNN, self).__init__()
        self.feat_stride = feat_stride
        if backbone == 'vgg':
            self.extractor, classifier = decom_vgg16(pretrained)
            self.rpn = RegionProposalNetwork(
                512, 512,
                ratios=ratios,
                anchor_scales=anchor_scales,
                feat_stride=self.feat_stride,
                mode=mode
            )
            self.head = VGG16RoIHead(
                n_class=num_classes + 1,
                roi_size=7,
                spatial_scale=1,
                classifier=classifier
            )
        elif backbone == 'resnet50' or backbone == 'resnet101':
            if backbone == 'resnet50':
                self.extractor, classifier = resnet50(pretrained)
            elif backbone == 'resnet101':
                self.extractor, classifier = resnet101(pretrained)
            self.rpn = RegionProposalNetwork(
                1024, 512,
                ratios=ratios,
                anchor_scales=anchor_scales,
                feat_stride=self.feat_stride,
                mode=mode
            )
            self.head = ResnetRoIHead(
                n_class=num_classes + 1,
                roi_size=14,
                spatial_scale=1,
                classifier=classifier
            )

    def forward(self, x, scale=1., mode="forward"):
        if mode == "forward":
            img_size = x.shape[2:]
            base_feature = self.extractor.forward(x)
            _, _, rois, roi_indices, _ = self.rpn.forward(base_feature, img_size, scale)
            roi_cls_locs, roi_scores = self.head.forward(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores, rois, roi_indices
        elif mode == "extractor":
            base_feature = self.extractor.forward(x)
            return base_feature
        elif mode == "rpn":
            base_feature, img_size = x
            rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn.forward(base_feature, img_size, scale)
            return rpn_locs, rpn_scores, rois, roi_indices, anchor
        elif mode == "head":
            base_feature, rois, roi_indices, img_size = x
            roi_cls_locs, roi_scores = self.head.forward(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
