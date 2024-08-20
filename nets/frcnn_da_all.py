import torch.nn as nn

from nets.classifier_da_all import ResnetRoIHead, VGG16RoIHead
from nets.resnet import resnet50, resnet101
from nets.rpn import RegionProposalNetwork
from nets.vgg16 import decom_vgg16
from mp.mp import SharedExtractor, FeatureForSVD, FeatureForTransformer, FeatureForDomainClassifier, \
    InstanceSharedExtractor, FeatureForInstanceTransformer, build_hdd_model


class FasterRCNN(nn.Module):
    def __init__(self, num_classes,
                 batch_size=1,
                 domain_text='',
                 instance_text='',
                 fp32=False,
                 token=[5, 10, 15, 20],
                 amp=False,
                 clip_backbone='',
                 mode="",
                 feat_stride=16,
                 anchor_scales=[8, 16, 32],
                 ratios=[0.5, 1, 2],
                 backbone='',
                 pretrained=False):
        super(FasterRCNN, self).__init__()
        N1, N2, N3, N4 = token
        self.clip_model = build_hdd_model(domain_classes=domain_text, instance_classes=instance_text, fp32=fp32,
                                          amp=amp, N1=N1, N2=N2, N3=N3, N4=N4, backbone=clip_backbone)
        self.batch_size = batch_size
        self.feat_stride = feat_stride
        self.shared_feature_extractor = SharedExtractor(in_channels=1024).cuda()
        self.instance_shared_feature_extractor = InstanceSharedExtractor().cuda()
        self.feature_for_svd = FeatureForSVD(p=24).cuda()
        self.global_classifier = FeatureForDomainClassifier(in_channels=1024).cuda()
        self.global_clip = FeatureForTransformer(in_channels=1024).cuda()
        self.instance_clip = FeatureForInstanceTransformer().cuda()
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
            self.head = Resnet50RoIHead(
                n_class=num_classes + 1,
                roi_size=14,
                spatial_scale=1,
                classifier=classifier
            )

    def forward(self, x, scale=1., mode="forward", stage='train'):
        if mode == "forward":
            img_size = x.shape[2:]
            base_feature = self.extractor.forward(x)

            _, _, rois, roi_indices, _ = self.rpn.forward(base_feature, img_size, scale)
            roi_cls_locs, roi_scores, fc7 = self.head.forward(base_feature, rois, roi_indices, img_size, stage='forward')
            return roi_cls_locs, roi_scores, rois, roi_indices, fc7
        elif mode == "extractor":
            base_feature = self.extractor.forward(x)
            if stage == 'train':
                shared_feature = self.shared_feature_extractor(base_feature)
                out_for_global_classifier = self.global_classifier(shared_feature)
                out_for_global_clip = self.global_clip(shared_feature)
                return base_feature, out_for_global_classifier, out_for_global_clip
            elif stage == 'val':
                return base_feature
            else:
                print('stage key error')
                raise ValueError
        elif mode == "rpn":
            base_feature, img_size = x
            rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn.forward(base_feature, img_size, scale)
            return rpn_locs, rpn_scores, rois, roi_indices, anchor
        elif mode == "head":
            base_feature, rois, roi_indices, img_size = x
            if stage == 'train':
                roi_cls_locs, roi_scores, instance_feature = self.head.forward(base_feature, rois, roi_indices,
                                                                               img_size, stage)
                instance_shared_feature = self.instance_shared_feature_extractor(instance_feature)
                out_for_instance_clip, out_for_instance_cos = self.instance_clip(instance_shared_feature, self.batch_size)
                out_for_svd = self.feature_for_svd(instance_shared_feature)

                return roi_cls_locs, roi_scores, out_for_instance_clip, out_for_instance_cos, out_for_svd
            elif stage == 'val':
                roi_cls_locs, roi_scores = self.head.forward(base_feature, rois, roi_indices,
                                                             img_size, stage)
                return roi_cls_locs, roi_scores
            else:
                print('stage key error')
                raise ValueError

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
