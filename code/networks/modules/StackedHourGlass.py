import torch
import torch.nn as nn


################################
# Hourglass landmark detector
################################

class StackedHourGlass(nn.Module):
    def __init__(self, num_feature, num_stack, num_keypoints, connect_type):
        super(StackedHourGlass, self).__init__()
        self.num_feature = num_feature
        self.num_stack = num_stack
        self.num_keypoints = num_keypoints
        self.connect_type = connect_type
        self.pre_conv_block = nn.Sequential(
            nn.Conv2d(3, self.num_feature // 4, 7, 2, 3),
            nn.BatchNorm2d(self.num_feature // 4),
            nn.ReLU(inplace=True),
            ResidualBlock(self.num_feature // 4, self.num_feature // 2),
            nn.MaxPool2d(2, 2),
            ResidualBlock(self.num_feature // 2, self.num_feature // 2),
            ResidualBlock(self.num_feature // 2, self.num_feature),
        )
        self._init_stacked_hourglass()

    def _init_stacked_hourglass(self):
        for i in range(self.num_stack):
            setattr(self, 'hg' + str(i), HourGlass(4, self.num_feature))
            setattr(self, 'hg' + str(i) + '_res1',
                    ResidualBlock(self.num_feature, self.num_feature))
            setattr(self, 'hg' + str(i) + '_lin1',
                    Lin(self.num_feature, self.num_feature))
            setattr(self, 'hg' + str(i) + '_conv_pred',
                    nn.Conv2d(self.num_feature, self.num_keypoints, 1))
            
            if i < self.num_stack - 1:
                if self.connect_type == 'cat':
                    setattr(self, 'compress_step' + str(i), 
                            nn.Conv2d(2 * self.num_feature, self.num_feature, 1))
                setattr(self, 'hg' + str(i) + '_conv1',
                        nn.Conv2d(self.num_feature, self.num_feature, 1))
                setattr(self, 'hg' + str(i) + '_conv2',
                        nn.Conv2d(self.num_keypoints, self.num_feature, 1))
                
    def forward(self, x, last_hidden=None):
        return_hidden = []
        x = self.pre_conv_block(x) #(n,128,32,32)
        inter = x

        for i in range(self.num_stack):
            hg = eval('self.hg'+str(i))(inter)
            # Residual layers at output resolution
            ll = hg
            ll = eval('self.hg'+str(i)+'_res1')(ll)
            # Linear layer to produce first set of predictions
            ll = eval('self.hg'+str(i)+'_lin1')(ll)
            # Predicted heatmaps
            out = eval('self.hg'+str(i)+'_conv_pred')(ll)
            # Add predictions back
            if i < self.num_stack - 1:
                ll_ = eval('self.hg'+str(i)+'_conv1')(ll)
                tmpOut_ = eval('self.hg'+str(i)+'_conv2')(out)
                # cross connections
                if last_hidden is None:
                    inter = inter + ll_ + tmpOut_
                else:
                    if self.connect_type == 'mean':
                        inter = inter + ll_ + (tmpOut_ + last_hidden[i]) / 2
                    elif self.connect_type == 'cat':
                        compressedOut_ = eval('self.compress_step' + str(i))(torch.cat((tmpOut_, last_hidden[i]), 1))
                        inter = inter + ll_ + compressedOut_
                return_hidden.append(tmpOut_)
        return out, return_hidden # return final heatmap only
    

class FeedbackHourGlass(nn.Module):
    def __init__(self, num_feature, num_keypoints):
        super().__init__()
        self.num_feature = num_feature
        self.num_keypoints = num_keypoints
        
        self.pre_conv_block = nn.Sequential(
            nn.Conv2d(3, self.num_feature // 4, 7, 2, 3),
            nn.ReLU(inplace=True),
            ResidualBlock(self.num_feature // 4, self.num_feature // 2, False),
            nn.MaxPool2d(2, 2),
            ResidualBlock(self.num_feature // 2, self.num_feature // 2, False),
            ResidualBlock(self.num_feature // 2, self.num_feature, False),
        )
        self.compress_in = nn.Conv2d(2 * self.num_feature, 2 * self.num_feature, 1)
        
        self.hg = HourGlass(4, 2 * self.num_feature, False)
        self.hg_conv_out = nn.Sequential(
            ResidualBlock(self.num_feature, self.num_feature, False),
            Lin(self.num_feature, self.num_feature, False),
            nn.Conv2d(self.num_feature, self.num_keypoints, 1))

    def forward(self, x, last_hidden=None):
        feature = self.pre_conv_block(x)
        if last_hidden is None:
            feature = self.compress_in(torch.cat((feature, feature), dim=1))
        else:
            feature = self.compress_in(torch.cat((feature, last_hidden), dim=1))
        feature = self.hg(feature)
        heatmap = self.hg_conv_out(feature[:, :self.num_feature]) # first half
        return heatmap, feature[:, self.num_feature:] # second half

    
class FeedbackHourGlassWithCoarse(nn.Module):
    def __init__(self, num_feature, num_keypoints):
        super().__init__()
        self.num_feature = num_feature
        self.num_keypoints = num_keypoints
        
        self.pre_conv_block = nn.Sequential(
            nn.Conv2d(3, self.num_feature // 4, 7, 2, 3),
            nn.ReLU(inplace=True),
            ResidualBlock(self.num_feature // 4, self.num_feature // 2, False),
            nn.MaxPool2d(2, 2),
            ResidualBlock(self.num_feature // 2, self.num_feature // 2, False),
            ResidualBlock(self.num_feature // 2, self.num_feature, False),
        )
        self.compress_in = nn.Conv2d(2 * self.num_feature, 2 * self.num_feature, 1)
        
        self.first_hg = HourGlass(4, self.num_feature, False)
        
        self.hg = HourGlass(4, 2 * self.num_feature, False)
        self.hg_conv_out = nn.Sequential(
            ResidualBlock(self.num_feature, self.num_feature, False),
            Lin(self.num_feature, self.num_feature, False),
            nn.Conv2d(self.num_feature, self.num_keypoints, 1))

    def forward(self, x, last_hidden):
        feature = self.pre_conv_block(x)
        
        if last_hidden is None:
            feature = self.first_hg(feature)
            heatmap = self.hg_conv_out(feature)
            return heatmap, feature
        else:  
            feature = self.compress_in(torch.cat((feature, last_hidden), dim=1))
            feature = self.hg(feature)
            heatmap = self.hg_conv_out(feature[:, :self.num_feature]) # first half
            return heatmap, feature[:, self.num_feature:] # second half

    
class HourGlass(nn.Module):
    def __init__(self, num_layer, num_feature, need_bn=True):
        super(HourGlass, self).__init__()
        self._n = num_layer
        self._f = num_feature
        self.need_bn = need_bn
        self._init_layers(self._n, self._f)

    def _init_layers(self, n, f):
        setattr(self, 'res' + str(n) + '_1', ResidualBlock(f, f, self.need_bn))
        setattr(self, 'pool' + str(n) + '_1', nn.MaxPool2d(2, 2))
        setattr(self, 'res' + str(n) + '_2', ResidualBlock(f, f, self.need_bn))
        if n > 1:
            self._init_layers(n - 1, f)
        else:
            self.res_center = ResidualBlock(f, f, self.need_bn)
        setattr(self, 'res' + str(n) + '_3', ResidualBlock(f, f, self.need_bn))

    def _forward(self, x, n, f):
        up1 = eval('self.res' + str(n) + '_1')(x)

        low1 = eval('self.pool' + str(n) + '_1')(x)
        low1 = eval('self.res' + str(n) + '_2')(low1)
        if n > 1:
            low2 = self._forward(low1, n - 1, f)
        else:
            low2 = self.res_center(low1)
        low3 = low2
        low3 = eval('self.' + 'res' + str(n) + '_3')(low3)
        up2 = nn.functional.interpolate(low3, scale_factor=2, mode='bilinear', align_corners=True)

        return up1 + up2

    def forward(self, x):
        return self._forward(x, self._n, self._f)
    

class Lin(nn.Module):
    def __init__(self,numIn,numout,need_bn=True):
        super(Lin,self).__init__()
        if need_bn:
            self.conv_block = nn.Sequential(
                nn.Conv2d(numIn,numout,1), 
                nn.BatchNorm2d(numout),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv_block = nn.Sequential(
                nn.Conv2d(numIn,numout,1), 
                nn.ReLU(inplace=True)
            )
    def forward(self,x):
        return self.conv_block(x)
    
    
class ResidualBlock(nn.Module):
    def __init__(self, num_in, num_out, need_bn=True):
        super(ResidualBlock, self).__init__()
        if need_bn:
            self.conv_block = nn.Sequential(
                nn.Conv2d(num_in, num_out // 2, 1), 
                nn.BatchNorm2d(num_out // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_out // 2, num_out // 2, 3, stride=1, padding=1),
                nn.BatchNorm2d(num_out // 2), nn.ReLU(inplace=True),
                nn.Conv2d(num_out // 2, num_out, 1), nn.BatchNorm2d(num_out))
            self.skip_layer = None if num_in == num_out else nn.Sequential(
                nn.Conv2d(num_in, num_out, 1), nn.BatchNorm2d(num_out))
        else:
            self.conv_block = nn.Sequential(
                nn.Conv2d(num_in, num_out // 2, 1), 
                nn.ReLU(inplace=True),
                nn.Conv2d(num_out // 2, num_out // 2, 3, stride=1, padding=1),
                nn.Conv2d(num_out // 2, num_out, 1))
            self.skip_layer = None if num_in == num_out else nn.Conv2d(num_in, num_out, 1)

    
    def forward(self, x):
        residual = self.conv_block(x)
        if self.skip_layer:
            x = self.skip_layer(x)
        return x + residual
