"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import utils
import functools
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
from torch.nn import init

##################################################################################
# Discriminator
##################################################################################
# Defines the PatchGAN discriminator with the specified arguments.
class PatchDis(nn.Module):
    def __init__(self, input_dim, params):
        super(PatchDis, self).__init__()

        self.n_layer = params['patch_n_layer']
        self.dim = params['dim']
        self.norm = params['norm']
        self.activ = params['activ']
        self.num_scales = params['num_scales']
        self.gan_type = params['gan_type']
        self.pad_type = params['pad_type']
        self.use_sigmoid = not (self.gan_type =='lsgan')
        self.input_dim = input_dim
        self.cnns = nn.ModuleList()

        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim = self.dim
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(self.input_dim, self.dim, kernel_size=kw, stride=2, padding=padw),
                    nn.LeakyReLU(0.2, True)]
        for i in range(self.n_layer - 1):
            sequence += [
                Conv2dBlock(dim, dim * 2, kw, 2, padw, norm=self.norm, activation=self.activ, pad_type='zero')]
            dim *= 2
        sequence +=[
                Conv2dBlock(dim, dim * 2, kw, 1, padw, norm=self.norm, activation=self.activ, pad_type='zero')]

        sequence += [nn.Conv2d(dim * 2, 1, kernel_size=kw, stride=1, padding=padw)]
        if self.use_sigmoid:
            sequence += [nn.Sigmoid()]
        sequence = nn.Sequential(*sequence)
        return sequence

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            # x = self.downsample(x)
        return outputs

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1)**2) # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

class MsImageDis(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, input_dim, params):
        super(MsImageDis, self).__init__()
        self.n_layer = params['n_layer']
        self.gan_type = params['gan_type']
        self.dim = params['dim']
        self.norm = params['norm']
        self.activ = params['activ']
        self.num_scales = params['num_scales']
        self.pad_type = params['pad_type']
        self.input_dim = input_dim
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1)**2) # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

##################################################################################
# Generator
##################################################################################

class AdaINGen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, params):
        super(AdaINGen, self).__init__()
        dim = params['dim']
        style_dim = params['style_dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']
        mlp_dim = params['mlp_dim']
        mpl_n_blk = params['mlp_n_blk']
        # style encoder
        self.enc_style = StyleEncoder(4, input_dim, dim, style_dim, norm='none', activ=activ, pad_type=pad_type)

        # content encoder
        self.enc_content = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim, input_dim, res_norm='adain', activ=activ, pad_type=pad_type)

        # MLP to generate AdaIN parameters
        self.mlp = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, mpl_n_blk, norm='none', activ=activ)

    def forward(self, images):
        # reconstruct an image
        content, style_fake = self.encode(images)
        images_recon = self.decode(content, style_fake)
        return images_recon

    def encode(self, images):
        # encode an image to its content and style codes
        style_fake = self.enc_style(images)
        content = self.enc_content(images)
        return content, style_fake

    def decode(self, content, style):
        # decode content and style codes to an image
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params

class AdaINGanilla(nn.Module):
    # ------------  ADAIN WITH GANILLA GENRATOR  ------------ #
    # AdaIN Ganilla Generator auto-encoder architecture
    # Generator architector taken from:
    # https://github.com/giddyyupp/ganilla
    # ------------------------------------------------------- #

    def __init__(self, input_dim, params):
        super(AdaINGanilla, self).__init__()
        dim               = params['dim']
        style_dim         = params['style_dim']
        n_downsample      = params['n_downsample']
        n_res             = params['n_res']
        activ             = params['activ']
        pad_type          = params['pad_type']
        mlp_dim           = params['mlp_dim']
        ganilla_ngf       = params['ganilla_ngf']
        ganilla_block_nf  = params['ganilla_block_nf']
        ganilla_layer_nb  = params['ganilla_layer_nb']
        use_dropout       = params['use_dropout']
        output_dim        = params['output_dim']
        # Ganilla Style Encoder
        # self.enc_style = GanillaStyleEncoder(input_dim, style_dim, ganilla_ngf, ganilla_block_nf, ganilla_layer_nb,
        #                                      use_dropout, norm = 'none', pad_type =pad_type)
        # style encoder
        self.enc_style = StyleEncoder(4, input_dim, dim, style_dim, norm='none', activ=activ, pad_type=pad_type)

        # Ganilla Content Encoder
        self.enc_content = GanillaContentEncoder(input_dim, ganilla_ngf, ganilla_block_nf, ganilla_layer_nb,
                                                 use_dropout, norm = 'in', pad_type =pad_type)

        sk_sizes = [self.enc_content.layer1[ganilla_layer_nb[0] - 1].conv2.out_channels,
                    self.enc_content.layer2[ganilla_layer_nb[1] - 1].conv2.out_channels,
                    self.enc_content.layer3[ganilla_layer_nb[2] - 1].conv2.out_channels,
                    self.enc_content.layer4[ganilla_layer_nb[3] - 1].conv2.out_channels]
        self.dec = GanillaDecoder(output_dim, *sk_sizes, res_norm='adain', activ=activ, pad_type=pad_type)
        #self.dec = GanillaDecoder2(n_res,output_dim, *sk_sizes, res_norm='adain', activ=activ, pad_type=pad_type)

        # MLP to generate AdaIN parameters
        self.mlp = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)
                    # input_dim, output_dim, dim, n_blk, norm = 'none', activ = 'relu'

    def forward(self, images):
        # reconstruct an image
        content, style_fake = self.encode(images)
        images_recon = self.decode(content, style_fake)
        return images_recon

    def encode(self, images):
        # encode an image to its content and style codes
        style_fake = self.enc_style(images)
        content = self.enc_content(images)
        return content, style_fake

    def decode(self, content, style):
        # decode content and style codes to an image
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params

class VAEGen(nn.Module):
    # VAE architecture
    def __init__(self, input_dim, params):
        super(VAEGen, self).__init__()
        dim = params['dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']

        # content encoder
        self.enc = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.dec = Decoder(n_downsample, n_res, self.enc.output_dim, input_dim, res_norm='in', activ=activ, pad_type=pad_type)

    def forward(self, images):
        # This is a reduced VAE implementation where we assume the outputs are multivariate Gaussian distribution with mean = hiddens and std_dev = all ones.
        hiddens = self.encode(images)
        if self.training == True:
            noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
            images_recon = self.decode(hiddens + noise)
        else:
            images_recon = self.decode(hiddens)
        return images_recon, hiddens

    def encode(self, images):
        hiddens = self.enc(images)
        noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
        return hiddens, noise

    def decode(self, hiddens):
        images = self.dec(hiddens)
        return images


##################################################################################
# Encoder and Decoders
##################################################################################

class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(StyleEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)] # global average pooling
        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

class GanillaStyleEncoder(nn.Module):
    def __init__(self, input_dim, style_dim, ganilla_ngf, ganilla_block_nf, ganilla_layer_nb, use_dropout, norm, pad_type):
        super(GanillaStyleEncoder, self).__init__()

        self.layer0 = FirstBlock_Ganilla(input_dim, ganilla_ngf, norm=norm, pad_type=pad_type)
        # residuals
        self.layer1 = self._make_layer_ganilla(BasicBlock_Ganilla, ganilla_ngf, ganilla_block_nf[0], ganilla_layer_nb[0],
                                               use_dropout, norm, stride=1)
        self.layer2 = self._make_layer_ganilla(BasicBlock_Ganilla, ganilla_block_nf[0], ganilla_block_nf[1],
                                               ganilla_layer_nb[1], use_dropout, norm, stride=2)
        self.layer3 = self._make_layer_ganilla(BasicBlock_Ganilla, ganilla_block_nf[1], ganilla_block_nf[2],
                                               ganilla_layer_nb[2], use_dropout, norm, stride=2)
        self.layer4 = self._make_layer_ganilla(BasicBlock_Ganilla, ganilla_block_nf[2], ganilla_block_nf[3],
                                               ganilla_layer_nb[3], use_dropout, norm, stride=2)

        self.pool_layer = nn.AdaptiveAvgPool2d(1) # global average pooling
        self.fc_style   = nn.Conv2d(ganilla_block_nf[3], style_dim, 1, 1, 0)


    def _make_layer_ganilla(self, block, inplanes, planes, blocks, use_dropout, norm, stride=1):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(inplanes, planes, use_dropout, stride=stride, norm = norm, pad_type = 'reflect'))
            inplanes = planes # * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Ganilla Encoder
        x = self.layer0(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # Global Pooling & FC -> Style Code
        x = self.pool_layer(x4)
        x = self.fc_style(x)
        out = x
        return out

class GanillaContentEncoder(nn.Module):
    def __init__(self, input_dim, ganilla_ngf, ganilla_block_nf, ganilla_layer_nb, use_dropout, norm, pad_type):
        super(GanillaContentEncoder, self).__init__()

        self.layer0 = FirstBlock_Ganilla(input_dim, ganilla_ngf, norm=norm, pad_type=pad_type)
        # residuals
        self.layer1 = self._make_layer_ganilla(BasicBlock_Ganilla, ganilla_ngf, ganilla_block_nf[0], ganilla_layer_nb[0],
                                               use_dropout, norm, stride=1)
        self.layer2 = self._make_layer_ganilla(BasicBlock_Ganilla, ganilla_block_nf[0], ganilla_block_nf[1],
                                               ganilla_layer_nb[1], use_dropout, norm, stride=2)
        self.layer3 = self._make_layer_ganilla(BasicBlock_Ganilla, ganilla_block_nf[1], ganilla_block_nf[2],
                                               ganilla_layer_nb[2], use_dropout, norm, stride=2)
        self.layer4 = self._make_layer_ganilla(BasicBlock_Ganilla, ganilla_block_nf[2], ganilla_block_nf[3],
                                               ganilla_layer_nb[3], use_dropout, norm, stride=2)

    def _make_layer_ganilla(self, block, inplanes, planes, blocks, use_dropout, norm, stride=1):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(inplanes, planes, use_dropout, stride=stride, norm = norm, pad_type = 'reflect'))
            inplanes = planes # * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Ganilla Encoder
        x = self.layer0(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        out = x4
        return [x1, x2, x3, out]

class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        # Conv2dBlock parameters- input_dim, output_dim, kernel_size, stride, padding = 0, norm = 'none', activation = 'relu', pad_type = 'zero')
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

class GanillaDecoder(nn.Module):
    def __init__(self, output_dim, C2_size, C3_size, C4_size, C5_size, res_norm='none', activ='lrelu', pad_type='reflect', feature_size=128):
        super(GanillaDecoder, self).__init__()
        # upsample C5 to get P5 from the FPN paper
        kw_adain = 3
        pdw_adain=1
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = Conv2dBlock(feature_size, feature_size, kw_adain, 1, pdw_adain, norm=res_norm, activation=activ,
                                pad_type=pad_type)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = Conv2dBlock(feature_size, feature_size, kw_adain, 1, pdw_adain, norm=res_norm, activation=activ,
                                pad_type=pad_type)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = Conv2dBlock(feature_size, feature_size, kw_adain, 1, pdw_adain, norm=res_norm, activation=activ,
                                pad_type=pad_type)

        self.P2_1 = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P2_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.rp4 = nn.ReflectionPad2d(1)
        self.P2_2 = nn.Conv2d(int(feature_size), int(feature_size / 2), kernel_size=3, stride=1, padding=0)

        self.final= Conv2dBlock(int(feature_size / 2), output_dim, 7, 1, padding=output_dim, norm='none', activation='tanh',
                                pad_type=pad_type)
    def forward(self, inputs):

        C2, C3, C4, C5 = inputs

        i = 0
        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_adain_x = self.P5_2(P5_upsampled_x)

        i += 1
        P4_x = self.P4_1(C4)
        P4_x = P5_adain_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_adain_x = self.P4_2(P4_upsampled_x)

        i += 1
        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_adain_x
        P3_upsampled_x = self.P3_upsampled(P3_x)
        P3_adain_x = self.P3_2(P3_upsampled_x)

        i += 1
        P2_x = self.P2_1(C2)
        P2_x = P2_x + P3_adain_x
        P2_upsampled_x = self.P2_upsampled(P2_x)
        P2_x = self.rp4(P2_upsampled_x)
        P2_x = self.P2_2(P2_x)

        out = self.final(P2_x)
        return out

class GanillaDecoder2(nn.Module):
    def __init__(self, n_res, output_dim, C2_size, C3_size, C4_size, C5_size, res_norm='none', activ='lrelu', pad_type='reflect', feature_size=128):
        super(GanillaDecoder2, self).__init__()
        # upsample C5 to get P5 from the FPN paper
        kw_adain = 3
        pdw_adain=1

        self.res_block_model = []
        # AdaIN residual blocks
        self.res_block_model += [ResBlocks(n_res, C5_size, res_norm, activ, pad_type=pad_type)]
        self.res_block_model = nn.Sequential(*self.res_block_model)

        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = Conv2dBlock(feature_size, feature_size, kw_adain, 1, pdw_adain, norm='ln', activation=activ,
                                pad_type=pad_type)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = Conv2dBlock(feature_size, feature_size, kw_adain, 1, pdw_adain, norm='ln', activation=activ,
                                pad_type=pad_type)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = Conv2dBlock(feature_size, feature_size, kw_adain, 1, pdw_adain, norm='ln', activation=activ,
                                pad_type=pad_type)

        self.P2_1 = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P2_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.rp4 = nn.ReflectionPad2d(1)
        self.P2_2 = nn.Conv2d(int(feature_size), int(feature_size / 2), kernel_size=3, stride=1, padding=0)

        self.final= Conv2dBlock(int(feature_size / 2), output_dim, 7, 1, padding=output_dim, norm='none', activation='tanh',
                                pad_type=pad_type)
    def forward(self, inputs):

        C2, C3, C4, C5 = inputs

        res_x = self.res_block_model(C5)

        i = 0
        P5_x = self.P5_1(res_x)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        #P5_adain_x = self.P5_2(P5_upsampled_x)

        i += 1
        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        #P4_adain_x = self.P4_2(P4_upsampled_x)

        i += 1
        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_upsampled_x = self.P3_upsampled(P3_x)
        #P3_adain_x = self.P3_2(P3_upsampled_x)

        i += 1
        P2_x = self.P2_1(C2)
        P2_x = P2_x + P3_upsampled_x
        P2_upsampled_x = self.P2_upsampled(P2_x)
        P2_x = self.rp4(P2_upsampled_x)
        P2_x = self.P2_2(P2_x)

        out = self.final(P2_x)
        return out


class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero'):
        super(Decoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

##################################################################################
# Sequential Models
##################################################################################
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class BasicBlock_Ganilla(nn.Module):

    # inputs should be input_dim, output_dim, kernel_size, stride, padding = 0, norm = 'none', activation = 'relu', pad_type = 'zero'
    def __init__(self, input_dim, output_dim, use_dropout, kernel_size=3, stride=1, padding = 1, norm = 'none', pad_type='reflect'):
        super(BasicBlock_Ganilla, self).__init__()
        self.expansion = 1

        # initialize padding
        if pad_type == 'reflect':
            # Pads the input tensor using the reflection of the input boundary
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        self.rp1    = self.pad
        self.conv1  = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=0, bias=False)
        self.bn1    = self.norm
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(0.5)
        self.rp2    = self.pad
        self.conv2  = nn.Conv2d(output_dim, output_dim, kernel_size=kernel_size, stride=1, padding=0, bias=False)
        self.bn2    = self.norm
        self.out_planes = output_dim

        self.shortcut = nn.Sequential()
        if stride != 1 or input_dim != output_dim:
            # self.shortcut = nn.Sequential(
            #     nn.Conv2d(input_dim,  output_dim, kernel_size=1, stride=stride, bias=False),
            #     self.norm
            # )
            shortcut_layers = [nn.Conv2d(input_dim,  output_dim, kernel_size=1, stride=stride, bias=False)]
            if self.norm:
                shortcut_layers += [self.norm]
            self.shortcut = nn.Sequential(*shortcut_layers)
            # self.final_conv = nn.Sequential(
            #     self.pad,
            #     nn.Conv2d(self.expansion * output_dim * 2, self.expansion * output_dim, kernel_size=3, stride=1,
            #               padding=0, bias=False),
            #     self.norm
            # )
            final_conv_layers = [self.pad]
            final_conv_layers += [nn.Conv2d(self.expansion * output_dim * 2, self.expansion * output_dim, kernel_size=3, stride=1,
                                  padding=0, bias=False)]
            if self.norm:
                final_conv_layers += [self.norm]
            self.final_conv = nn.Sequential(*final_conv_layers)
        else:
            # self.final_conv = nn.Sequential(
            #     self.pad,
            #     nn.Conv2d(output_dim * 2, output_dim, kernel_size=3, stride=1, padding=0, bias=False),
            #     self.norm
            # )
            final_conv_layers = [self.pad]
            final_conv_layers += [nn.Conv2d(output_dim * 2, output_dim, kernel_size=3, stride=1, padding=0, bias=False)]
            if self.norm:
                final_conv_layers += [self.norm]
            self.final_conv = nn.Sequential(*final_conv_layers)

    def forward(self, x):
        out = self.conv1(self.rp1(x))
        if self.norm:
            out = self.norm(out)
        out = F.relu(out)
        if self.use_dropout:
            out = self.dropout(out)
        out = self.conv2(self.rp2(out))
        if self.norm:
            out = self.norm(out)
        inputt = self.shortcut(x)
        catted = torch.cat((out, inputt), 1)
        out = self.final_conv(catted)
        out = F.relu(out)
        return out


class FirstBlock_Ganilla(nn.Module):

    # input_dim = input_nc, output_dim = ngf (number generator filters in the first layer)
    def __init__(self, input_dim, output_dim, padding=1, norm='none', pad_type='reflect'):
        super(FirstBlock_Ganilla, self).__init__()
        self.expansion = 1

        # initialize padding
        if pad_type == 'reflect':
            # Pads the input tensor using the reflection of the input boundary
            self.pad = nn.ReflectionPad2d
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        self.pad1    = self.pad(input_dim)
        self.conv1   = nn.Conv2d(input_dim, output_dim, kernel_size=7, stride=1, padding=0, bias=True)
        # self.norm    = nn.InstanceNorm2d(output_dim)
        self.relu    = nn.ReLU(inplace=True)
        self.pad2    = self.pad(padding)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        x = self.pad1(x)
        x = self.conv1(x)
        if self.norm:
            x = self.norm(x)
        x = self.relu(x)
        x = self.pad2(x)
        out = self.maxpool(x)
        return out

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

class BigGUnetDecoder(nn.Module):
    def __init__(self, D_init='ortho', output_dim=1, D_activation=nn.ReLU(inplace=False), D_wide=True, SN_eps=1e-12):

        # Network Architecture
        ch = 64
        self.resolution   = [128, 64, 32, 16, 8, 4]
        self.out_channels = [item * ch for item in [1, 2, 4, 8, 16, 16]]
        self.in_channels  = [3] + [ch*item for item in [1, 2, 4, 8, 16]]

        self.which_conv = functools.partial(SNConv2d,
                          kernel_size=3, padding=1,
                          num_svs=1, num_itrs=1,
                          eps=self.SN_eps)
        self.which_linear = functools.partial(SNLinear,
                                              num_svs=1, num_itrs=1,
                                              eps=self.SN_eps)
        self.which_embedding = functools.partial(SNEmbedding,
                                                 num_svs=1, num_itrs=1,
                                                 eps=self.SN_eps)
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # Use Wide D as in BigGAN
        self.D_wide = D_wide
        # Activation
        self.activation = D_activation

        self.blocks = []
        for index in range(len(self.out_channels)):
            self.blocks += [[BigGDBlock(in_channels=self.in_channels[index],
                                           out_channels=self.out_channels[index],
                                           which_conv=self.which_conv,
                                           wide=self.D_wide,
                                           activation=self.activation,
                                           preactivation=(index > 0),
                                           downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]

        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        # larger if we're e.g. turning this into a VAE with an inference output
        self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)
        # Embedding for projection discrimination
        # self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1])
        self.init_weights()

        # self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
        #                         betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)

        # Initialize
        def init_weights(self):
            self.param_count = 0
            for module in self.modules():
                if (isinstance(module, nn.Conv2d)
                        or isinstance(module, nn.Linear)
                        or isinstance(module, nn.Embedding)):
                    if self.init == 'ortho':
                        init.orthogonal_(module.weight)
                    elif self.init == 'N02':
                        init.normal_(module.weight, 0, 0.02)
                    elif self.init in ['glorot', 'xavier']:
                        init.xavier_uniform_(module.weight)
                    else:
                        print('Init style not recognized...')
                    self.param_count += sum([p.data.nelement() for p in module.parameters()])
            print('Param count for G''s initialized parameters: %d' % self.param_count)

        def forward(self, x, y=None):
            # Stick x into h for cleaner for loops without flow control
            h = x
            # Loop over blocks
            for index, blocklist in enumerate(self.blocks):
                for block in blocklist:
                    h = block(h)
            # Apply global sum pooling as in SN-GAN
            h = torch.sum(self.activation(h), [2, 3])
            # Get initial class-unconditional output
            out = self.linear(h)
            # Get projection of final featureset onto class vectors and add to evidence
            out = out + torch.sum(self.embed(y) * h, 1, keepdim=True)
            return out

# Residual block for the discriminator
class BigGDBlock(nn.Module):
    def __init__(self, in_channels, out_channels, wide=True,
                 preactivation=False, activation=None, downsample=None, ):
        super(BigGDBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
        self.hidden_channels = self.out_channels if wide else self.in_channels
        self.which_conv = SNConv2d
        self.preactivation = preactivation
        self.activation = activation
        self.downsample = downsample

        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.hidden_channels)
        self.conv2 = self.which_conv(self.hidden_channels, self.out_channels)
        self.learnable_sc = True if (in_channels != out_channels) or downsample else False
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                           kernel_size=1, padding=0)

    def shortcut(self, x):
        if self.preactivation:
            if self.learnable_sc:
                x = self.conv_sc(x)
            if self.downsample:
                x = self.downsample(x)
        else:
            if self.downsample:
                x = self.downsample(x)
            if self.learnable_sc:
                x = self.conv_sc(x)
        return x

    def forward(self, x):
        if self.preactivation:
            # h = self.activation(x) # NOT TODAY SATAN
            # Andy's note: This line *must* be an out-of-place ReLU or it
            #              will negatively affect the shortcut connection.
            h = F.relu(x)
        else:
            h = x
        h = self.conv1(h)
        h = self.conv2(self.activation(h))
        if self.downsample:
            h = self.downsample(h)

        return h + self.shortcut(x)

##################################################################################
# VGG network definition
##################################################################################
class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        # relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        # relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        # relu3_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        h = F.relu(self.conv4_2(h), inplace=True)
        h = F.relu(self.conv4_3(h), inplace=True)
        # relu4_3 = h

        h = F.relu(self.conv5_1(h), inplace=True)
        h = F.relu(self.conv5_2(h), inplace=True)
        h = F.relu(self.conv5_3(h), inplace=True)
        relu5_3 = h

        return relu5_3
        # return [relu1_2, relu2_2, relu3_3, relu4_3]

##################################################################################
# Normalization layers
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

# Spectral normalization base class based on BigGan implementation
class SN(object):
    def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
        # Number of power iterations per step
        self.num_itrs = num_itrs
        # Number of singular values
        self.num_svs = num_svs
        # Transposed?
        self.transpose = transpose
        # Epsilon value for avoiding divide-by-0
        self.eps = eps
        # Register a singular vector for each sv
        for i in range(self.num_svs):
            self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
            self.register_buffer('sv%d' % i, torch.ones(1))

    # Singular vectors (u side)
    @property
    def u(self):
        return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

    # Singular values;
    # note that these buffers are just for logging and are not used in training.
    @property
    def sv(self):
        return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]

    # Compute the spectrally-normalized weight
    def W_(self):
        W_mat = self.weight.view(self.weight.size(0), -1)
        if self.transpose:
            W_mat = W_mat.t()
        # Apply num_itrs power iterations
        for _ in range(self.num_itrs):
            svs, us, vs = utils.power_iteration(W_mat, self.u, update=self.training, eps=self.eps)
            # Update the svs
        if self.training:
            with torch.no_grad():  # Make sure to do this in a no_grad() context or you'll get memory leaks!
                for i, sv in enumerate(svs):
                    self.sv[i][:] = sv
        return self.weight / svs[0]

##################################################################################
# Convolution layers
##################################################################################

# 2D Conv layer with spectral norm
class SNConv2d(nn.Conv2d, SN):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
             padding=0, dilation=1, groups=1, bias=True,
             num_svs=1, num_itrs=1, eps=1e-12):
    nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride,
                     padding, dilation, groups, bias)
    SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)
  def forward(self, x):
    return F.conv2d(x, self.W_(), self.bias, self.stride,
                    self.padding, self.dilation, self.groups)