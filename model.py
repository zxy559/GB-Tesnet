import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


def resnet18_features(pretrained=True):
    net = resnet18(pretrained=pretrained)
    # drop avgpool + fc, keep conv feature map
    return nn.Sequential(*list(net.children())[:-2])


class GBTesNet(nn.Module):
    def __init__(self, num_classes, prototype_shape=(20, 512, 1, 1),
                 r_init_logit=-2.0, r_beta=10.0, pooling_k=3, use_ball=True,
                 pretrained=True):
        super(GBTesNet, self).__init__()
        self.num_classes = num_classes
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_prototypes_per_class = self.num_prototypes // num_classes
        self.r_beta = r_beta
        self.pooling_k = pooling_k
        self.use_ball = use_ball

        self.prototype_class_identity = torch.zeros(self.num_prototypes, num_classes)
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // self.num_prototypes_per_class] = 1

        self.features = resnet18_features(pretrained=pretrained)
        proto_ch = prototype_shape[1]
        self.add_on_layers = nn.Sequential(
            nn.Conv2d(512, proto_ch, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(proto_ch, proto_ch, kernel_size=1),
            nn.Sigmoid(),
        )
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)
        self.prototype_r_param = nn.Parameter(
            torch.full((self.num_prototypes,), float(r_init_logit)), requires_grad=True)
        self.last_layer = nn.Linear(self.num_prototypes, num_classes, bias=False)
        self._initialize_weights()

    def conv_features(self, x):
        return self.add_on_layers(self.features(x))

    def get_prototype_radii(self):
        return F.softplus(self.prototype_r_param)

    def _normalized_proto_weight(self):
        w = F.normalize(self.prototype_vectors.view(self.num_prototypes, -1), p=2, dim=1)
        return w.view(self.num_prototypes, self.prototype_shape[1], 1, 1)

    def _center_distances(self, x):
        # d = -cos, smaller is closer
        x_norm = F.normalize(x, p=2, dim=1)
        return -F.conv2d(x_norm, self._normalized_proto_weight())

    def _project_map(self, x):
        w = self.prototype_vectors.view(self.num_prototypes, self.prototype_shape[1], 1, 1)
        return F.conv2d(x, w)

    def _ball_distance_map(self, center_distances):
        if not self.use_ball:
            return center_distances
        r = self.get_prototype_radii().view(1, -1, 1, 1)
        residual = F.softplus(self.r_beta * (center_distances - r)) / self.r_beta
        return center_distances + residual

    def _pool_min(self, distances):
        values = distances.flatten(2)
        k = min(self.pooling_k, values.size(2))
        return torch.topk(values, k=k, dim=2, largest=False).values.mean(dim=2)

    def _pool_max(self, responses):
        values = responses.flatten(2)
        k = min(self.pooling_k, values.size(2))
        return torch.topk(values, k=k, dim=2, largest=True).values.mean(dim=2)

    def forward(self, x, return_aux=False):
        feat = self.conv_features(x)
        center = self._center_distances(feat)
        ball = self._ball_distance_map(center)
        project = self._project_map(feat)

        min_ball = self._pool_min(ball)
        max_proj = self._pool_max(project)
        activations = max_proj - min_ball if self.use_ball else max_proj
        logits = self.last_layer(activations)

        if return_aux:
            return logits, min_ball, {
                'feature_map': feat,
                'center_distances': center,
                'ball_distances': ball,
                'project_distances': project,
                'prototype_activations': activations,
                'radii': self.get_prototype_radii(),
            }
        return logits, min_ball

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        pos = torch.t(self.prototype_class_identity)
        neg = 1 - pos
        self.last_layer.weight.data.copy_(1 * pos + incorrect_strength * neg)

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)


def construct_GBTesNet(num_classes, prototypes_per_class=10, pretrained=True,
                       pooling_k=3, r_beta=10.0, use_ball=True):
    proto_shape = (num_classes * prototypes_per_class, 512, 1, 1)
    return GBTesNet(
        num_classes=num_classes,
        prototype_shape=proto_shape,
        r_beta=r_beta,
        pooling_k=pooling_k,
        use_ball=use_ball,
        pretrained=pretrained,
    )
