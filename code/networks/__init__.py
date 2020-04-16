import functools

import torch
import torch.nn as nn
from torch.nn import init

####################
# initialize
####################

def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        if classname != "MeanShift":
            # print('initializing [%s] ...' % classname)
            init.normal_(m.weight.data, 0.0, std)
            if m.bias is not None:
                m.bias.data.zero_()
    elif isinstance(m, (nn.Linear)):
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d)):
        init.normal_(m.weight.data, 1.0, std)
        init.constant_(m.bias.data, 0.0)

def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        if classname != "MeanShift":
            # print('initializing [%s] ...' % classname)
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()
    elif isinstance(m, (nn.Linear)):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d)):
        init.constant_(m.weight.data, 1.0)
        m.weight.data *= scale
        init.constant_(m.bias.data, 0.0)

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        if classname != "MeanShift":
            # print('initializing [%s] ...' % classname)
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
    elif isinstance(m, (nn.Linear)):
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d)):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

####################
# define network
####################

def create_model(opt):
    if opt['mode'].startswith('sr'):
        net = define_net(opt['networks'], opt['use_gpu'])
        return net
    else:
        raise NotImplementedError("The mode [%s] of networks is not recognized." % opt['mode'])

def define_D(opt):
    net_opt = opt['net_D']
    which_model = net_opt['which_model_D']
    if which_model == 'LightCNN':
        from .modules.architecture import LigntCNN as m
        net = m()
    elif which_model == 'UNetF':
        # UNet feature critic
        from .modules.architecture import UNetFeatureDiscriminator as m
        net = m(net_opt['feature_only'])
    else:
        raise NotImplementedError("The discriminator network [%s] is not implemented" % which_model)

    if opt['use_gpu']:
        net = nn.DataParallel(net).cuda()
    return net

def define_F(opt):
    net_opt = opt['net_F']
    which_model = net_opt['which_model_F']
    if opt['use_gpu']:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if which_model == 'SENet':
        from .modules.architecture import SENet as m
        net = m(device=device)
    elif which_model == 'VGG16':
        from .modules.architecture import VGGFeatureExtractor as m
        net = m(use_bn=net_opt['use_bn'], use_input_norm=net_opt['use_input_norm'], device=device)
    elif which_model == 'LightCNN':
        from .modules.architecture import LightCNNFeatureExtractor as m
        net = m()
    else:
        raise NotImplementedError("The feature extractor network [%s] is not implemented" % which_model)

    if opt['use_gpu']:
        net = nn.DataParallel(net).cuda()
    return net

# choose one network
def define_net(opt, use_gpu):
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    which_model = opt['which_model'].upper()
    print('===> Building network [%s]...'%which_model)

    if which_model == 'DIC':
        from .dic_arch import DIC
        net = DIC(opt, device)
    else:
        raise NotImplementedError("Network [%s] is not recognized." % which_model)

    if use_gpu:
        net = nn.DataParallel(net).cuda()

    return net
