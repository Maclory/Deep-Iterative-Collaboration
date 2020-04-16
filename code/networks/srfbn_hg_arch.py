import torch
import torch.nn as nn
from .blocks import ConvBlock, DeconvBlock, FeatureHeatmapFusingBlock
from .modules.architecture import StackedHourGlass
from .srfbn_arch import FeedbackBlock

def merge_heatmap_5(heatmap_in, detach):
    '''
    merge 68 heatmap to 5
    heatmap: B*N*32*32
    '''
    # landmark[36:42], landmark[42:48], landmark[27:36], landmark[48:68]
    heatmap = heatmap_in.clone()
    max_heat = heatmap.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    max_heat = torch.max(max_heat, torch.ones_like(max_heat) * 0.05)
    heatmap /= max_heat
    if heatmap.size(1) == 5:
        return heatmap.detach() if detach else heatmap
    elif heatmap.size(1) == 68:
        new_heatmap = torch.zeros_like(heatmap[:, :5])
        new_heatmap[:, 0] = heatmap[:, 36:42].sum(1) # left eye
        new_heatmap[:, 1] = heatmap[:, 42:48].sum(1) # right eye
        new_heatmap[:, 2] = heatmap[:, 27:36].sum(1) # nose
        new_heatmap[:, 3] = heatmap[:, 48:68].sum(1) # mouse
        new_heatmap[:, 4] = heatmap[:, :27].sum(1) # face silhouette
        return new_heatmap.detach() if detach else new_heatmap
    elif heatmap.size(1) == 194: # Helen
        new_heatmap = torch.zeros_like(heatmap[:, :5])
        tmp_id = torch.cat((torch.arange(134, 153), torch.arange(174, 193)))
        new_heatmap[:, 0] = heatmap[:, tmp_id].sum(1) # left eye
        tmp_id = torch.cat((torch.arange(114, 133), torch.arange(154, 173)))
        new_heatmap[:, 1] = heatmap[:, tmp_id].sum(1) # right eye
        tmp_id = torch.arange(41, 57)
        new_heatmap[:, 2] = heatmap[:, tmp_id].sum(1) # nose
        tmp_id = torch.arange(58, 113)
        new_heatmap[:, 3] = heatmap[:, tmp_id].sum(1) # mouse
        tmp_id = torch.arange(0, 40)
        new_heatmap[:, 4] = heatmap[:, tmp_id].sum(1) # face silhouette
        return new_heatmap.detach() if detach else new_heatmap
    else:
        raise NotImplementedError('Fusion for face landmark number %d not implemented!' % heatmap.size(1))
        
        
class FeedbackBlockHeatmapAttention(FeedbackBlock):
    def __init__(self,
                 num_features,
                 num_groups,
                 upscale_factor,
                 act_type,
                 norm_type,
                 num_heatmap,
                 num_fusion_block,
                 device=torch.device('cuda')):
        super().__init__(num_features,
                         num_groups,
                         upscale_factor,
                         act_type,
                         norm_type,
                         device)
        self.fusion_block = FeatureHeatmapFusingBlock(num_features,
                                                      num_heatmap,
                                                      num_fusion_block)
        
    def forward(self, x, heatmap):
        if self.should_reset:
            self.last_hidden = torch.zeros(x.size()).to(self.device)
            self.last_hidden.copy_(x)
            self.should_reset = False

        x = torch.cat((x, self.last_hidden), dim=1)
        x = self.compress_in(x)
        
        # fusion
        x = self.fusion_block(x, heatmap)

        lr_features = []
        hr_features = []
        lr_features.append(x)

        for idx in range(self.num_groups):
            LD_L = torch.cat(tuple(lr_features), 1)    # when idx == 0, lr_features == [x]
            if idx > 0:
                LD_L = self.uptranBlocks[idx-1](LD_L)
            LD_H = self.upBlocks[idx](LD_L)

            hr_features.append(LD_H)

            LD_H = torch.cat(tuple(hr_features), 1)
            if idx > 0:
                LD_H = self.downtranBlocks[idx-1](LD_H)
            LD_L = self.downBlocks[idx](LD_H)

            lr_features.append(LD_L)

        del hr_features
        output = torch.cat(tuple(lr_features[1:]), 1)   # leave out input x, i.e. lr_features[0]
        output = self.compress_out(output)

        self.last_hidden = output

        return output
    
    
class FeedbackBlockCustom(FeedbackBlock):
    def __init__(self, num_features, num_groups, upscale_factor, act_type,
                 norm_type, num_features_in):
        super(FeedbackBlockCustom, self).__init__(
            num_features, num_groups, upscale_factor, act_type, norm_type)
        self.compress_in = ConvBlock(num_features_in, num_features,
                                     kernel_size=1,
                                     act_type=act_type, norm_type=norm_type)
    
    def forward(self, x):
        x = self.compress_in(x)

        lr_features = []
        hr_features = []
        lr_features.append(x)

        for idx in range(self.num_groups):
            LD_L = torch.cat(tuple(lr_features), 1)    # when idx == 0, lr_features == [x]
            if idx > 0:
                LD_L = self.uptranBlocks[idx-1](LD_L)
            LD_H = self.upBlocks[idx](LD_L)

            hr_features.append(LD_H)

            LD_H = torch.cat(tuple(hr_features), 1)
            if idx > 0:
                LD_H = self.downtranBlocks[idx-1](LD_H)
            LD_L = self.downBlocks[idx](LD_H)

            lr_features.append(LD_L)

        del hr_features
        output = torch.cat(tuple(lr_features[1:]), 1)   # leave out input x, i.e. lr_features[0]
        output = self.compress_out(output)

        return output

        
class SRFBN_HG(nn.Module):
    def __init__(self, opt):
        super().__init__()
        in_channels = opt['in_channels']
        out_channels = opt['out_channels']
        num_groups = opt['num_groups']
        hg_num_feature = opt['hg_num_feature']
        hg_num_stack = opt['hg_num_stack']
        hg_num_keypoints = opt['hg_num_keypoints']
        hg_connect_type = opt['hg_connect_type']
        act_type = 'prelu'
        norm_type = None

        self.num_steps = opt['num_steps']
        num_features = opt['num_features']
        self.upscale_factor = opt['scale']

        if self.upscale_factor == 4:
            stride = 4
            padding = 2
            kernel_size = 8
        elif self.upscale_factor == 8:
            stride = 8
            padding = 2
            kernel_size = 12
        else:
            raise NotImplementedError("Upscale factor %d not implemented!" % self.upscale_factor)

        # LR feature extraction block
        self.conv_in = ConvBlock(
            in_channels,
            4 * num_features,
            kernel_size=3,
            act_type=act_type,
            norm_type=norm_type)
        self.feat_in = ConvBlock(
            4 * num_features,
            num_features,
            kernel_size=1,
            act_type=act_type,
            norm_type=norm_type)

        # basic block
        # first block takes only original LR feature as input, coarse SR
        self.first_block = FeedbackBlockCustom(num_features, num_groups, self.upscale_factor,
                                   act_type, norm_type, num_features)
        # second block takes LR feature, last FB output and heatmap as input
        self.block = FeedbackBlockCustom(num_features, num_groups, self.upscale_factor,
                                   act_type, norm_type, 2 * num_features + hg_num_keypoints)

        # reconstruction block
        # self.upsample = nn.Upsample(scale_factor=upscale_factor, mode='bilinear')

        self.out = DeconvBlock(
            num_features,
            num_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            act_type='prelu',
            norm_type=norm_type)
        self.conv_out = ConvBlock(
            num_features,
            out_channels,
            kernel_size=3,
            act_type=None,
            norm_type=norm_type)

        self.HG = StackedHourGlass(hg_num_feature, hg_num_stack, hg_num_keypoints, hg_connect_type)
        
        if self.upscale_factor == 4:
            self.HG_out = None
        elif self.upscale_factor == 8:
            self.HG_out = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        inter_res = nn.functional.interpolate(
            x,
            scale_factor=self.upscale_factor,
            mode='bilinear',
            align_corners=False)

        x = self.conv_in(x)
        x = self.feat_in(x)
        sr_outs = []
        heatmap_outs = []
        hg_last_hidden = None
        
        f_in = x
        for step in range(self.num_steps):
            if step == 0:
                # use first FB to do coarse SR
                FB_out = self.first_block(f_in)
            else:
                FB_out = self.block(f_in)
                
            h = torch.add(inter_res, self.conv_out(self.out(FB_out)))
            
            # detach, stop heatmap loss propogate to SR
            heatmap, hg_last_hidden = self.HG(h, hg_last_hidden) 
            if self.HG_out:
                heatmap_out = self.HG_out(heatmap) 
            else:
                heatmap_out = factor * heatmap
            f_in = torch.cat((x, FB_out, heatmap_out), dim=1)
            
            sr_outs.append(h)
            heatmap_outs.append(heatmap)

        return sr_outs, heatmap_outs  # return output of every timesteps
