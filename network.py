import utils
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.se_resnet import BottleneckX, SEResNeXt
from model.options import DEFAULT_NET_OPT

class MultiPrmSequential(nn.Sequential):
    def __init__(self, *args):
        super(MultiPrmSequential, self).__init__(*args)

    def forward(self, input, cat_feature):
        for module in self._modules.values():
            input = module(input, cat_feature)
        return input

def make_secat_layer(block, inplanes, planes, cat_planes, block_count, stride=1, no_bn=False):
    outplanes = planes * block.expansion
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        if no_bn:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False))
        else:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

    layers = []
    layers.append(block(inplanes, planes, cat_planes, 16, stride, downsample, no_bn=no_bn))
    for i in range(1, block_count):
        layers.append(block(outplanes, planes, cat_planes, 16, no_bn=no_bn))

    return MultiPrmSequential(*layers)

class SeCatLayer(nn.Module):
    def __init__(self, channel, cat_channel, reduction=16):
        super(SeCatLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(cat_channel + channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, cat_feature):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = torch.cat([y, cat_feature], 1)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SECatBottleneckX(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cat_channel, cardinality=16, stride=1, downsample=None, no_bn=False):
        super(SECatBottleneckX, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = None if no_bn else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, groups=cardinality, bias=False)
        self.bn2 = None if no_bn else nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = None if no_bn else nn.BatchNorm2d(planes * self.expansion)

        self.selayer = SeCatLayer(planes * self.expansion, cat_channel)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, cat_feature):
        residual = x
        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.bn2 is not None:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.bn3 is not None:
            out = self.bn3(out)

        out = self.selayer(out, cat_feature)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class FeatureConv(nn.Module):
    def __init__(self, input_dim=512, output_dim=256, input_size=32, output_size=16, net_opt=DEFAULT_NET_OPT):
        super(FeatureConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.output_size = output_size

        no_bn = not net_opt['bn']
        
        if input_size == output_size * 4:
            stride1, stride2 = 2, 2
        elif input_size == output_size * 2:
            stride1, stride2 = 2, 1
        else:
            stride1, stride2 = 1, 1
        
        seq = []
        seq.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride1, padding=1, bias=False))
        if not no_bn: seq.append(nn.BatchNorm2d(output_dim))
        seq.append(nn.ReLU(inplace=True))
        seq.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=stride2, padding=1, bias=False))
        if not no_bn: seq.append(nn.BatchNorm2d(output_dim))
        seq.append(nn.ReLU(inplace=True))
        seq.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False))
        seq.append(nn.ReLU(inplace=True))

        self.network = nn.Sequential(*seq)

    def forward(self, x):
        return self.network(x)

class DecoderBlock(nn.Module):
    def __init__(self, inplanes, planes, color_fc_out, block_num, no_bn):
        super(DecoderBlock, self).__init__()
        self.secat_layer = make_secat_layer(SECatBottleneckX, inplanes, planes//4, color_fc_out, block_num, no_bn=no_bn)
        self.ps = nn.PixelShuffle(2)

    def forward(self, x, cat_feature):
        out = self.secat_layer(x, cat_feature)
        return self.ps(out)
        

class Generator(nn.Module):
    def __init__(self, luma_size, chroma_size, luma_dim=1, output_dim=2,
                 layers=[6, 4, 3, 3], net_opt=DEFAULT_NET_OPT):
        super(Generator, self).__init__()
        self.luma_dim = luma_dim
        self.output_dim = output_dim
        self.chroma_size = chroma_size
        # self.ref_length = ref_length

        self.luma_size = luma_size
        self.layers = layers

        self.cardinality = 16

        #self.bottom_h = self.input_size // 16
        #self.Linear = nn.Linear(cv_class_num, self.bottom_h*self.bottom_h*32)

        self.color_fc_out = 64
        self.net_opt = net_opt

        no_bn = not net_opt['bn']
        '''
        if net_opt['relu']:
            self.colorFC = nn.Sequential(
                nn.Linear(self.ref_length, self.color_fc_out), nn.ReLU(inplace=True),
                nn.Linear(self.color_fc_out, self.color_fc_out), nn.ReLU(inplace=True),
                nn.Linear(self.color_fc_out, self.color_fc_out), nn.ReLU(inplace=True),
                nn.Linear(self.color_fc_out, self.color_fc_out)
            )
        else:
            self.colorFC = nn.Sequential(
                nn.Linear(self.ref_length, self.color_fc_out),
                nn.Linear(self.color_fc_out, self.color_fc_out),
                nn.Linear(self.color_fc_out, self.color_fc_out),
                nn.Linear(self.color_fc_out, self.color_fc_out)
            )
        '''
        self.conv1 = self._make_encoder_block_first(self.luma_dim, 32)
        self.conv2 = self._make_encoder_block_second(34, 32)
        self.down_conv = nn.Sequential(nn.Conv2d(32, 64, 3, 2, 1), nn.LeakyReLU(0.2),)
        self.conv3 = nn.Sequential(nn.Conv2d(65, 64, 3, 1, 1), nn.LeakyReLU(0.2),)
        self.conv4 = self._make_encoder_block(64, 128)
        self.conv5 = self._make_encoder_block(128, 256)
        
        #bottom_layer_len = 256 + 64 + (256 if net_opt['cit'] else 0)
        bottom_layer_len = 256 + 256

        '''
        self.deconv1 = DecoderBlock(bottom_layer_len, 4*256, self.color_fc_out, self.layers[0], no_bn=no_bn)
        self.deconv2 = DecoderBlock(256 + 128, 4*128, self.color_fc_out, self.layers[1], no_bn=no_bn)
        self.deconv3 = DecoderBlock(128 + 64, 4*64, self.color_fc_out, self.layers[2], no_bn=no_bn)
        self.deconv4 = DecoderBlock(64 + 32, 4*32, self.color_fc_out, self.layers[3], no_bn=no_bn)
        self.deconv5 = nn.Sequential(
            nn.Conv2d(32 + 16, 32, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, self.output_dim, 3, 1, 1),
            nn.Tanh(),
        )
        '''
        self.deconv1 = self._make_block_1(512, 512)
        self.deconv2 = self._make_block_1(256, 256)
        self.deconv3 = self._make_block_1(128, 128)
        self.deconv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, self.output_dim, 1, 1, 0),
            nn.Tanh(),
        )
        '''
        if net_opt['cit']:
            self.featureConv = FeatureConv(net_opt=net_opt)
        '''
        self.colorConv = nn.Sequential(
            nn.Conv2d(2, 32, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.Tanh()
        )

        if net_opt['guide']:
            self.deconv_for_decoder = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), # output is 64 * 64
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), # output is 128 * 128
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # output is 256 * 256
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(32, 2, 3, stride=1, padding=1, output_padding=0), # output is 256 * 256
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

    def _make_encoder_block(self, inplanes, planes):
        return nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

    def _make_encoder_block_first(self, inplanes, planes):
        return nn.Sequential(
            nn.Conv2d(inplanes, 16, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, planes, 3, 2, 1),
            nn.LeakyReLU(0.2),
        )

    def _make_encoder_block_second(self, inplanes, planes):
        return nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )


    def _make_block_1(self, inplanes, planes):
        return nn.Sequential(
            # SEResNeXt._make_layer(self, BottleneckX, planes//4, 2, inplanes=inplanes),
            # nn.Conv2d(planes, planes, 3, 1, 1),
            nn.Conv2d(inplanes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(inplanes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.PixelShuffle(2)
        )
        
    def forward(self, luma_in, chroma_in, QP_in):
        temp_out1 = self.conv1(luma_in)

        out1 = torch.cat([chroma_in, temp_out1], 1)
        out2 = self.conv2(out1)

        temp_out3 = self.down_conv(out2)
        temp_out3 = torch.cat([QP_in, temp_out3], 1)
        out3 = self.conv3(temp_out3)

        out4 = self.conv4(out3)
        out5 = self.conv5(out4)

        chroma_tensor = self.colorConv(chroma_in)

        concat_tensor = torch.cat([out5, chroma_tensor], 1)
        # print('concat_tensor:', concat_tensor.shape)
        out4_prime = self.deconv1(concat_tensor)
        # print('out4_prime:', out4_prime.shape)

        concat_tensor = torch.cat([out4, out4_prime], 1)
        # print('concat_tensor:', concat_tensor.shape)

        out3_prime = self.deconv2(concat_tensor)
        # print('out3_prime:', out3_prime.shape)

        concat_tensor = torch.cat([out3, out3_prime], 1)
        # print('concat_tensor:', concat_tensor.shape)

        out2_prime = self.deconv3(concat_tensor)

        concat_tensor = torch.cat([out2, out2_prime], 1)
        # print('concat_tensor:', concat_tensor.shape)

        out1_prime = self.deconv4(concat_tensor)
        # print('out1_prime:', out1_prime.shape)
        return out1_prime


class Discriminator(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, input_size=32, net_opt=DEFAULT_NET_OPT):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.input_size = input_size
        self.cardinality = 16

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_dim, 32, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, 2, 0),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = self._make_block_1(32, 64)
        self.conv3 = self._make_block_1(64, 128)
        self.conv4 = self._make_block_1(128, 256)
        self.conv5 = self._make_block_1(256, 512)
        '''
        self.conv6 = self._make_block_3(512, 512)
        self.conv7 = self._make_block_3(512, 512)
        self.conv8 = self._make_block_3(512, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        '''
        self.adv_judge = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
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

    def _make_block_1(self, inplanes, planes):
        return nn.Sequential(
            SEResNeXt._make_layer(self, BottleneckX, planes//4, 2, inplanes=inplanes),
            nn.Conv2d(planes, planes, 3, 2, 1),
            nn.LeakyReLU(0.2),
        )
    '''
    def _make_block_2(self, inplanes, planes):
        return nn.Sequential(
            SEResNeXt._make_layer(self, BottleneckX, planes//4, 2, inplanes=inplanes),
        )

    def _make_block_3(self, inplanes, planes):
        return nn.Sequential(
            SEResNeXt._make_layer(self, BottleneckX, planes//4, 1, inplanes=inplanes),
        )
    '''
    def forward(self, input):
        out = self.conv1(input)
        #print(out.shape)
        out = self.conv2(out)
        #print(out.shape)
        out = self.conv3(out)
        #print(out.shape)
        out = self.conv4(out)
        #print(out.shape)
        out = self.conv5(out)
        #print(out.shape)
        '''
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.avgpool(out)
        '''
        out = out.view(out.size(0), -1)

        adv_judge = self.adv_judge(out)

        return adv_judge

if __name__ == '__main__':
    temp1 = torch.zeros(1, 2, 32, 32)
    D = Discriminator(input_dim=2, input_size=32)
    out = D(temp1)
    print(out.shape)

    temp2 = torch.zeros(1, 1, 128, 128)
    temp3 = torch.zeros(1, 2, 64, 64)
    temp4 = torch.zeros(1, 1, 32, 32)
    G = Generator(luma_size=128, chroma_size=32, luma_dim=1, output_dim=2)
    out2 = G(temp2, temp3, temp4)
    print(out2[0].shape)


