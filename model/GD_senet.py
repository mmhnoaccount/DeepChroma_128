import utils
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.se_resnet import BottleneckX, SEResNeXt
from model.options import DEFAULT_NET_OPT

class FeatureConv(nn.Module):
    def __init__(self, input_dim=512, output_dim=256, input_size=32, output_size=16):
        super(FeatureConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.output_size = output_size
        
        if input_size == output_size * 4:
            stride1, stride2 = 2, 2
        elif input_size == output_size * 2:
            stride1, stride2 = 2, 1
        else:
            stride1, stride2 = 1, 1
        
        self.network = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride1, padding=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=stride2, padding=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.network(x)

class Generator(nn.Module):
    def __init__(self, input_size, cv_class_num, iv_class_num, input_dim=1, output_dim=3, 
                 layers=[12, 8, 5, 5], net_opt=DEFAULT_NET_OPT):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.cv_class_num = cv_class_num
        self.iv_class_num = iv_class_num

        self.input_size = input_size
        self.layers = layers

        self.cardinality = 16

        self.bottom_h = self.input_size // 16
        self.Linear = nn.Linear(cv_class_num, self.bottom_h*self.bottom_h*32)

        self.color_fc_out = 64
        self.net_opt = net_opt

        self.conv1 = self._make_encoder_block_first(self.input_dim, 16)
        self.conv2 = self._make_encoder_block(16, 32)
        self.conv3 = self._make_encoder_block(32, 64)
        self.conv4 = self._make_encoder_block(64, 128)
        self.conv5 = self._make_encoder_block(128, 256)
        
        bottom_layer_len = 256 + 64 + (256 if net_opt['cit'] else 0)

        self.deconv1 = self._make_block(bottom_layer_len, 4*256, self.layers[0])
        self.deconv2 = self._make_block(256 + 128, 4*128, self.layers[1])
        self.deconv3 = self._make_block(128 + 64, 4*64, self.layers[2])
        self.deconv4 = self._make_block(64 + 32, 4*32, self.layers[3])
        self.deconv5 = nn.Sequential(
            nn.Conv2d(32 + 16, 32, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh(),
        )

        if net_opt['cit']:
            self.featureConv = FeatureConv()

        self.colorConv = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.Tanh(),
        )

        if net_opt['guide']:
            self.deconv_for_decoder = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), # output is 64 * 64
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), # output is 128 * 128
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # output is 256 * 256
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1, output_padding=0), # output is 256 * 256
                nn.Tanh(),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _make_block(self, inplanes, planes, block_num):
        return nn.Sequential(
            SEResNeXt._make_layer(self, BottleneckX, planes//4, block_num, inplanes=inplanes),
            nn.PixelShuffle(2),
        )
    
    def _make_encoder_block(self, inplanes, planes):
        return nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

    def _make_encoder_block_first(self, inplanes, planes):
        return nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )
        
    def forward(self, input, feature_tensor, c_tag_class):
        out1 = self.conv1(input)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)

        # ==============================
        # it's about color variant tag set
        # temporally, don't think about noise z
        c_tag_tensor = self.Linear(c_tag_class)
        c_tag_tensor = c_tag_tensor.view(-1, 32, self.bottom_h, self.bottom_h)
        c_tag_tensor = self.colorConv(c_tag_tensor)

        # ==============================
        # Convolution Layer for Feature Tensor

        if self.net_opt['cit']:
            feature_tensor = self.featureConv(feature_tensor)
            concat_tensor = torch.cat([out5, feature_tensor, c_tag_tensor], 1)
        else:
            concat_tensor = torch.cat([out5, c_tag_tensor], 1)

        out4_prime = self.deconv1(concat_tensor)

        # ==============================

        concat_tensor = torch.cat([out4_prime, out4], 1)
        out3_prime = self.deconv2(concat_tensor)

        concat_tensor = torch.cat([out3_prime, out3], 1)
        out2_prime = self.deconv3(concat_tensor)

        concat_tensor = torch.cat([out2_prime, out2], 1)
        out1_prime = self.deconv4(concat_tensor)

        concat_tensor = torch.cat([out1_prime, out1], 1)
        full_output = self.deconv5(concat_tensor)

        # ==============================
        # out4_prime should be input of Guide Decoder

        if self.net_opt['guide']:
            decoder_output = self.deconv_for_decoder(out4_prime)
        else:
            decoder_output = full_output

        return full_output, decoder_output
