import torch
import torch.nn as nn

import torchvision.models as torch_models
import torch.nn.functional as F


def relu_fun():
    return nn.LeakyReLU(0.2)


class Encoder(nn.Module):
    def __init__(self, cfg, in_shape, features_shape=None):
        super().__init__()

        if cfg[0] == "vgg16":
            # directly use pre-trained vgg
            assert in_shape[0] == 3, "vgg only support color images"
            self.features, num_pool, channels = self._retrained_features(cfg[1])
        else:
            self.features, num_pool, channels = self._make_layers(cfg, in_shape[0])

        self.features_shape = features_shape
        if self.features_shape is None:
            # calculate fc_in_dim
            shrink = 2**num_pool
            self.features_shape = (channels, in_shape[1] // shrink, in_shape[2] // shrink)

        fc_in_dim = channels * self.features_shape[1] * self.features_shape[2]

        self.fc_layers = nn.Sequential(
            # nn.Linear(fc_in_dim, fc_hidden_dim),
            # relu_fun(),
            # nn.Linear(fc_hidden_dim, latent_dim),
            nn.Linear(fc_in_dim, fc_in_dim),
        )

    def _retrained_features(self, layer_num):
        vgg = torch_models.vgg16_bn(pretrained=True)

        features = vgg.features[: layer_num]

        num_pool = 0
        last_conv = None
        for layer in features:
            if isinstance(layer, nn.MaxPool2d):
                num_pool += 1

            if isinstance(layer, nn.Conv2d):
                last_conv = layer

        channels = last_conv.out_channels

        return features, num_pool, channels

    def _make_layers(self, cfg, channels):
        layers = []
        num_pool = 0
        for x in cfg:
            if x == "M":
                # layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                layers += [nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)]
                num_pool += 1
            else:
                layers += [
                    nn.Conv2d(channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    relu_fun(),
                ]
                channels = x
        return nn.Sequential(*layers), num_pool, channels

    def forward(self, x, return_features=False):
        features = self.features(x)

        out = features.view(features.size(0), -1)
        out = self.fc_layers(out)

        if return_features is True:
            return out, features

        return out


class Decoder(nn.Module):
    def __init__(self, cfg, feature_shape):
        super().__init__()

        self.feature_shape = feature_shape
        # self.fc_layers = nn.Sequential(
        #     nn.Linear(latent_dim, fc_hidden_dim),
        #     relu_fun(),
        #     nn.Linear(fc_hidden_dim, int(np.prod(feature_shape))),
        # )

        self.features = self._make_layers(cfg, feature_shape[0])

    def _make_layers(self, cfg, in_channels):
        layers = []
        for idx, x in enumerate(cfg):
            if x == "M":
                # layers += [torch.nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0)]
                layers += [torch.nn.Upsample(scale_factor=2)]

            else:
                if idx != len(cfg) - 1:
                    layers += [
                        nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                        nn.BatchNorm2d(x),
                        relu_fun(),
                    ]
                else:
                    # do not add batch norm and relu because we do not want to constrain output range too much
                    layers += [
                        nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    ]
                in_channels = x

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.fc_layers(x)
        x = x.view(x.size(0), *self.feature_shape)
        out = self.features(x)

        return out


def ae_denoise_loss(trainer, data, logits, target, cur_epoch, it, other_data):

    def to_classifier_feature(x):
        abs_val = (x[:, 0, ...]**2 + x[:, 1, ...]**2)
        abs_val = torch.sqrt(abs_val)

        feature = torch.log1p(abs_val)
        return feature

    denoised_feature = to_classifier_feature(data + logits)
    taget_feature = to_classifier_feature(target)

    loss = (denoised_feature - taget_feature).flatten(start_dim=1)

    loss = torch.sum(loss**2, dim=1)
    loss = torch.sqrt(loss)
    loss = torch.sum(loss) / logits.size(0)

    return loss


class AutoEncoder(nn.Module):
    def __init__(self, encoder_cfg, decoder_cfg, in_shape, features_shape=None):
        super().__init__()

        self.encoder = Encoder(encoder_cfg, in_shape, features_shape)
        self.decoder = Decoder(decoder_cfg, self.encoder.features_shape)

        self.latent = None

        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):

        # try to make latent robust to noise during training
        self.latent = self.encoder(x)

        output = self.decoder(self.latent)

        # remove extra data
        output = output[:, :, :x.shape[-2], :x.shape[-1]]
        assert output.shape == x.shape

        return output



