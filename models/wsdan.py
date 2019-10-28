"""
WS-DAN models

Hu et al.,
"See Better Before Looking Closer: Weakly Supervised Data Augmentation Network for Fine-Grained Visual Classification",
arXiv:1901.09891

Created: May 04,2019 - Yuchong Gu
Revised: May 07,2019 - Yuchong Gu
"""
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vgg import VGG
from models.resnet import ResNet
from models.inception import *

__all__ = ['WSDAN']


# Bilinear Attention Pooling
class BAP(nn.Module):
    def __init__(self, pool='GAP'):
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions):
        B = features.size(0)
        M = attentions.size(1)
        for i in range(M):
            AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, 1, -1)
            if i == 0:
                feature_matrix = AiF
            else:
                feature_matrix = torch.cat([feature_matrix, AiF], dim=1)
        feature_matrix = torch.mul(torch.sign(feature_matrix),torch.sqrt(torch.abs(feature_matrix)+1e-12))
        feature_matrix = nn.functional.normalize(feature_matrix, 2, [1,2])
        return feature_matrix


# WS-DAN: Weakly Supervised Data Augmentation Network for FGVC
class WSDAN(nn.Module):
    def __init__(self, num_classes, M=32, net=None):
        super(WSDAN, self).__init__()
        self.num_classes = num_classes
        self.M = M

        # Default Network
        self.baseline = 'inception'
        self.num_features = 768
        self.expansion = 1

        # Network Initialization
        if net is not None:
            self.features = net.get_features()

            if isinstance(net, ResNet):
                self.baseline = 'resnet'
                self.expansion = self.features[-1][-1].expansion
                self.num_features = 512
            elif isinstance(net, VGG):
                self.baseline = 'vgg'
                self.num_features = 512
        else:
            self.features = inception_v3(pretrained=True,transform_input=False).get_features()

        # Attention Maps
        self.attentions = nn.Conv2d(self.num_features * self.expansion, 192, kernel_size=1, bias=True)


        # Bilinear Attention Pooling
        self.bap = BAP(pool='GAP')


        # Classification Layer
        # self.fc = nn.Linear(self.M * self.num_features * self.expansion, self.num_classes)
        self.fc = nn.Conv2d(self.M * self.num_features, num_classes, kernel_size=1, bias=True)


        logging.info('WSDAN: using %s as feature extractor' % self.baseline)

    def forward(self, x):
        batch_size = x.size(0)

        # Feature Maps, Attention Maps and Feature Matrix
        feature_maps = self.features(x)
        attention_maps = F.relu(self.attentions(feature_maps), inplace=True)
        attention_maps = attention_maps[:, :self.M, :, :]
        feature_matrix = self.bap(feature_maps, attention_maps)

        # Classification

        p = self.fc(feature_matrix.reshape((-1,feature_maps.size(1) * attention_maps.size(1),1, 1))*100.0)
        p= torch.squeeze(p)

        # # Generate Attention Map
        # H, W = attention_maps.size(2), attention_maps.size(3)
        # if self.training:
        #     # Randomly choose one of attention maps Ak
        #     k_indices = np.random.randint(self.M, size=batch_size)
        #     attention_map = torch.zeros(batch_size, 1, H, W).to(torch.device("cuda"))  # (B, 1, H, W)
        #     for i in range(batch_size):
        #         attention_map[i] = attention_maps[i, k_indices[i]:k_indices[i] + 1, ...]
        # else:
        #     # Object Localization Am = mean(sum(Ak))
        #     attention_map = torch.mean(attention_maps, dim=1, keepdim=True)  # (B, 1, H, W)
        #
        # # Normalize Attention Map
        # attention_map = attention_map.view(batch_size, -1)  # (B, H * W)
        # attention_map_max, _ = attention_map.max(dim=1, keepdim=True)  # (B, 1)
        # attention_map_min, _ = attention_map.min(dim=1, keepdim=True)  # (B, 1)
        # attention_map = (attention_map - attention_map_min) / (attention_map_max - attention_map_min)  # (B, H * W)
        # attention_map = attention_map.view(batch_size, 1, H, W)  # (B, 1, H, W)

        return p, feature_matrix, attention_maps #p:分类结果，bap结果，随机选择的一个map且完成归一化

    def load_state_dict(self, state_dict, strict=True):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and model_dict[k].size() == v.size()}

        if len(pretrained_dict) == len(state_dict):
            logging.info('%s: All params loaded' % type(self).__name__)
        else:
            logging.info('%s: Some params were not loaded:' % type(self).__name__)
            not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
            logging.info(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))

        model_dict.update(pretrained_dict)
        super(WSDAN, self).load_state_dict(model_dict)


# if __name__ == '__main__':
#     net = WSDAN(num_classes=1000)
#     net.train()
#
#     for i in range(10):
#         input_test = torch.randn(10, 3, 512, 512)
#         p, feature_matrix, attention_map = net(input_test)
#
#     print(p.shape)
#     print(feature_matrix.shape)
#     print(attention_map.shape)
#
#     net.eval()
#     input_test = torch.randn(10, 3, 512, 352)
#     p, feature_matrix, attention_map = net(input_test)
#
#     print(p.shape)
#     print(feature_matrix.shape)
#     print(attention_map.shape)
