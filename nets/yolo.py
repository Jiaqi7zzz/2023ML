from collections import OrderedDict

import torch
import torch.nn as nn

from nets.darknet import darknet53

def conv2d(filter_in, filter_out, kernel_size):
    pad = 1 if kernel_size == 3 else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

def conv2d_noBN(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

def make_last_layers(filters_list, in_filters, out_filter):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True)
    )
    return m

def layers_noBN(filters_list, in_filters, out_filter):
    m = nn.Sequential(
        conv2d_noBN(in_filters, filters_list[0], 1),
        conv2d_noBN(filters_list[0], filters_list[1], 3),
        conv2d_noBN(filters_list[1], filters_list[0], 1),
        conv2d_noBN(filters_list[0], filters_list[1], 3),
        conv2d_noBN(filters_list[1], filters_list[0], 1),
        conv2d_noBN(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True)
    )
    return m

class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, pretrained = False , flag = 0):
        super(YoloBody, self).__init__()
        #   52,52,256
        #   26,26,512
        #   13,13,1024
        self.backbone = darknet53()
        self.l2_lambda = 0.001 if flag else 0
        #   out_filters : [64, 128, 256, 512, 1024]

        out_filters = self.backbone.layers_out_filters

        if not flag:
            self.last_layer0 = make_last_layers([512, 1024], out_filters[-1], len(anchors_mask[0]) * (num_classes + 5))

            self.last_layer1_conv = conv2d(512, 256, 1)
            self.last_layer1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.last_layer1 = make_last_layers([256, 512], out_filters[-2] + 256, len(anchors_mask[1]) * (num_classes + 5))

            self.last_layer2_conv = conv2d(256, 128, 1)
            self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.last_layer2 = make_last_layers([128, 256], out_filters[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))

        else:

            self.last_layer0            = layers_noBN([512, 1024], out_filters[-1], len(anchors_mask[0]) * (num_classes + 5))

            self.last_layer1_conv       = conv2d_noBN(512, 256, 1)
            self.last_layer1_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
            self.last_layer1            = layers_noBN([256, 512], out_filters[-2] + 256, len(anchors_mask[1]) * (num_classes + 5))

            self.last_layer2_conv       = conv2d_noBN(256, 128, 1)
            self.last_layer2_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
            self.last_layer2            = layers_noBN([128, 256], out_filters[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))

    # def l2_regularization(self):

    #     l2_reg = torch.tensor(0.).cuda()
    #     for param in self.parameters():
    #         l2_reg += torch.norm(param , p = 2)

    #     return self.l2_lambda * 0.5 * l2_reg
            
    def l1_regularization(self):

        l1_reg = torch.tensor(0.).cuda()
        for param in self.parameters():
            l1_reg += torch.norm(param , p = 1)

        return self.l2_lambda/600 * l1_reg
    
    def forward(self, x):
        x2, x1, x0 = self.backbone(x)

        out0_branch = self.last_layer0[:5](x0)
        out0 = self.last_layer0[5:](out0_branch)

        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)

        x1_in = torch.cat([x1_in, x1], 1)

        out1_branch = self.last_layer1[:5](x1_in)
        out1 = self.last_layer1[5:](out1_branch)

        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)

        x2_in = torch.cat([x2_in, x2], 1)
        
        out2 = self.last_layer2(x2_in)
        return out0, out1, out2
    

class YoloBody_noFPN(nn.Module):
    '''
    仅保留13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
    输出为out0 = (batch_size,255,13,13)
    '''
    def __init__(self, anchors_mask, num_classes, pretrained = False):
        super(YoloBody_noFPN, self).__init__()
        self.l2_lambda = 0
        self.backbone = darknet53()
        self.backbone.load_state_dict(torch.load("model_data/darknet53_backbone_weights.pth"))
        out_filters = self.backbone.layers_out_filters
        self.last_layer0            = make_last_layers([512, 1024], out_filters[-1], len(anchors_mask[0]) * (num_classes + 5))

    def forward(self, x):
        _, _, x0 = self.backbone(x)
        out0_branch = self.last_layer0[:5](x0)
        out0        = self.last_layer0[5:](out0_branch)
        return out0, out0, out0