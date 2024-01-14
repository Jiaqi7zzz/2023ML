'''
机器学习大作业--yolobody部分复现。
'''
import math
from functools import partial
import numpy as np
import torch
import torch.nn as nn

'''
  13x13的特征层对应的anchor是[116,90],[156,198],[373,326]
  26x26的特征层对应的anchor是[30,61],[62,45],[59,119]
  52x52的特征层对应的anchor是[10,13],[16,30],[33,23]
'''

class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, input_shape, cuda, anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]):
        super(YOLOLoss, self).__init__()
        self.anchors        = anchors
        self.num_classes    = num_classes
        self.bbox_attrs     = 5 + num_classes
        self.input_shape    = input_shape
        self.anchors_mask   = anchors_mask

        self.giou           = True
        self.balance        = [0.4, 1.0, 4]
        self.box_ratio      = 0.05
        self.obj_ratio      = 5 * (input_shape[0] * input_shape[1]) / (416 ** 2)
        self.cls_ratio      = 1 * (num_classes / 80)

        self.ignore_threshold = 0.5
        self.cuda           = cuda

    def clip_by_tensor(self, t):
        t_min = 1e-7
        t_max = 1 - 1e-7
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result
    
    '''
    MSELoss
    '''
    def MSELoss(self, pred, target):
        return torch.pow(pred - target, 2)
    '''
    二元交叉熵损失
    '''
    def BCELoss(self, pred, target):
        pred = self.clip_by_tensor(pred)
        output = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output

    def box_giou(self, b1, b2):
        b1_xy = b1[..., :2]
        b1_wh = b1[..., 2:4]
        b1_wh_half = b1_wh/2.
        b1_mins = b1_xy - b1_wh_half
        b1_maxes = b1_xy + b1_wh_half

        b2_xy = b2[..., :2]
        b2_wh = b2[..., 2:4]
        b2_wh_half = b2_wh/2.
        b2_mins = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half

        intersect_mins = torch.max(b1_mins, b2_mins)
        intersect_maxes = torch.min(b1_maxes, b2_maxes)
        intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(b1_mins))
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1] # width multiply height
        b1_S = b1_wh[..., 0] * b1_wh[..., 1]
        b2_S = b2_wh[..., 0] * b2_wh[..., 1]
        union_area = b1_S+ b2_S- intersect_area
        iou = intersect_area / union_area # 交并比

        enclose_mins = torch.min(b1_mins, b2_mins)
        enclose_maxes  = torch.max(b1_maxes, b2_maxes)
        enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]

        giou = iou - (enclose_area - union_area) / enclose_area
        
        return giou
    
    def get_target(self, l, targets, anchors, in_h, in_w):
        '''
        提取网络的输出值，获得其与真实值的差别
        网络输出的是bs 3 13 13 5+
        target是真实框
        in_h in_w是特征图的尺寸 13 26等等
        '''
        bs = len(targets) # batch_size / bs 3 13 13 5+
        '''
        无目标的tensor，哪些点上不含目标，因此是bs 3 13 13大小的
        每个图像的loss，大目标的loss小，小目标的loss大
        真实目标框，与bs 3 13 13 5+相同
        '''
        noobj_mask = torch.ones(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad = False)
        box_loss_scale = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad = False)
        y_true = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, self.bbox_attrs, requires_grad = False)

        for b in range(bs):            
            if len(targets[b]) == 0:
                continue
            batch_target = torch.zeros_like(targets[b]) # 3 13 13 5+
            '''
            转到特征层上面去
            in_w = 13
            '''
            batch_target[:, [0,2]] = targets[b][:, [0,2]] * in_w
            batch_target[:, [1,3]] = targets[b][:, [1,3]] * in_h
            batch_target[:, 4] = targets[b][:, 4]
            batch_target = batch_target.cpu()
            
            gt_box = torch.FloatTensor(torch.cat((torch.zeros((batch_target.size(0), 2)), batch_target[:, 2:4]), 1)) # num_true_box（真实的框的数目）, 4
            anchor_shapes = torch.FloatTensor(torch.cat((torch.zeros((len(anchors), 2)), torch.FloatTensor(anchors)), 1)) #9（len(anchors)）, 4（高宽 高宽）这是先验框
            '''
            计算交并比
            self.calculate_iou(gt_box, anchor_shapes) = [num_true_box, 9]每一个真实框和9个先验框的重合情况
            best_ns:
            [每个真实框最大的重合度max_iou, 每一个真实框最重合的先验框的序号]
            '''
            best_ns = torch.argmax(self.calculate_iou(gt_box, anchor_shapes), dim=-1)

            for t, best_n in enumerate(best_ns):
                '''
                过滤掉不属于当前特征层的先验框！！！（每次处理三个，但是要先建好九个）
                '''
                if best_n not in self.anchors_mask[l]:
                    continue
                k = self.anchors_mask[l].index(best_n)
                # 属于哪个特征图
                '''
                计算真实框中心点的坐标在特征图上所属的网格索引
                '''
                i = torch.floor(batch_target[t, 0]).long()
                j = torch.floor(batch_target[t, 1]).long()

                c = batch_target[t, 4].long()
                noobj_mask[b, k, j, i] = 0

                y_true[b, k, j, i, 0] = batch_target[t, 0]
                y_true[b, k, j, i, 1] = batch_target[t, 1]
                y_true[b, k, j, i, 2] = batch_target[t, 2]
                y_true[b, k, j, i, 3] = batch_target[t, 3]
                y_true[b, k, j, i, 4] = 1
                y_true[b, k, j, i, c + 5] = 1

                box_loss_scale[b, k, j, i] = batch_target[t, 2] * batch_target[t, 3] / in_w / in_h

        return y_true, noobj_mask, box_loss_scale

        
    def forward(self, l, input, targets=None):
        #   l代表的是，当前输入进来的有效特征层，是第几个有效特征层
        #   input的shape为 bs, 3*(5+num_classes), 13, 13
        #                   bs, 3*(5+num_classes), 26, 26
        #                   bs, 3*(5+num_classes), 52, 52
        #   targets代表的是真实框。
        bs      = input.size(0)
        in_h    = input.size(2)
        in_w    = input.size(3)
        '''
        每一层特征的一点对应原输入（416）的像素点的大小
        '''
        stride_h = self.input_shape[0] / in_h
        stride_w = self.input_shape[1] / in_w

        '''
        scale后的anchor（把原图上的转化到现在的特征图上）
        stride_w 和 h 取值为52 26 13
        '''
        scaled_anchors  = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]
        '''
        把
        bs, 3*(5+num_classes), 13, 13
        转成
        bs 3 13 13 5+

        prediction是包含框的信息的
        '''
        prediction = input.view(bs, len(self.anchors_mask[l]), self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()
        
        x = torch.sigmoid(prediction[..., 0]) # Sigmoid归一化
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        conf = torch.sigmoid(prediction[..., 4]) # 是否有物体
        pred_cls = torch.sigmoid(prediction[..., 5:]) # cls置信度

        y_true, noobj_mask, box_loss_scale = self.get_target(l, targets, scaled_anchors, in_h, in_w)


        noobj_mask, pred_boxes = self.get_ignore(l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask)

        y_true = y_true.type_as(x)
        noobj_mask = noobj_mask.type_as(x)
        box_loss_scale = box_loss_scale.type_as(x)

        # box_loss_scale是真实框宽高的乘积
        box_loss_scale = 2 - box_loss_scale
            
        loss = 0
        obj_mask = y_true[..., 4] == 1
        # 存在物体时（y_true[...,4] == 1 时才为True）
        n = torch.sum(obj_mask)
        if n != 0:
            giou = self.box_giou(pred_boxes, y_true[..., :4]).type_as(x)
            loss_loc = torch.mean((1 - giou)[obj_mask])
                # 定位损失
            loss_cls = torch.mean(self.BCELoss(pred_cls[obj_mask], y_true[..., 5:][obj_mask]))
            loss += loss_loc * self.box_ratio + loss_cls * self.cls_ratio

        loss_conf = torch.mean(self.BCELoss(conf, obj_mask.type_as(conf))[noobj_mask.bool() | obj_mask])
        loss += loss_conf * self.balance[l] * self.obj_ratio
        return loss
    
    def calculate_iou(self, _box_a, _box_b):
        b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
        b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
        b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
        b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2

        '''
        0 1 2 3 分别对应： 左上x左上y 右下x右下y
        '''
        box_a = torch.zeros_like(_box_a)
        box_b = torch.zeros_like(_box_b)
        box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
        box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2

        A , B = box_a.size(0) , box_b.size(0)

        max_xy  = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy  = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter   = torch.clamp((max_xy - min_xy), min=0) # 防止产生负的边
        inter   = inter[:, :, 0] * inter[:, :, 1]

        area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # expand as [A,B]
        area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter) 

        union = area_a + area_b - inter
        return inter / union  # [A,B]
    
    def get_ignore(self, l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask):
        '''
        写不明白了 这个函数是源代码里面的
        '''
        bs = len(targets)

        #-----------------------------------------------------#
        #   生成网格，先验框中心，网格左上角
        #-----------------------------------------------------#
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(x.shape).type_as(x)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(y.shape).type_as(x)

        # 生成先验框的宽高
        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[l]]
        anchor_w = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([0])).type_as(x)
        anchor_h = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([1])).type_as(x)
        
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        #-------------------------------------------------------#
        #   计算调整后的先验框中心与宽高
        #-------------------------------------------------------#
        pred_boxes_x    = torch.unsqueeze(x + grid_x, -1)
        pred_boxes_y    = torch.unsqueeze(y + grid_y, -1)
        pred_boxes_w    = torch.unsqueeze(torch.exp(w) * anchor_w, -1)
        pred_boxes_h    = torch.unsqueeze(torch.exp(h) * anchor_h, -1)
        pred_boxes      = torch.cat([pred_boxes_x, pred_boxes_y, pred_boxes_w, pred_boxes_h], dim = -1)
        
        for b in range(bs):           
            #-------------------------------------------------------#
            #   将预测结果转换一个形式
            #   pred_boxes_for_ignore      num_anchors, 4
            #-------------------------------------------------------#
            pred_boxes_for_ignore = pred_boxes[b].view(-1, 4)
            #-------------------------------------------------------#
            #   计算真实框，并把真实框转换成相对于特征层的大小
            #   gt_box      num_true_box, 4
            #-------------------------------------------------------#
            if len(targets[b]) > 0:
                batch_target = torch.zeros_like(targets[b])
                #-------------------------------------------------------#
                #   计算出正样本在特征层上的中心点
                #-------------------------------------------------------#
                batch_target[:, [0,2]] = targets[b][:, [0,2]] * in_w
                batch_target[:, [1,3]] = targets[b][:, [1,3]] * in_h
                batch_target = batch_target[:, :4].type_as(x)
                #-------------------------------------------------------#
                #   计算交并比
                #   anch_ious       num_true_box, num_anchors
                #-------------------------------------------------------#
                anch_ious = self.calculate_iou(batch_target, pred_boxes_for_ignore)
                #-------------------------------------------------------#
                #   每个先验框对应真实框的最大重合度
                #   anch_ious_max   num_anchors
                #-------------------------------------------------------#
                anch_ious_max, _    = torch.max(anch_ious, dim = 0)
                anch_ious_max       = anch_ious_max.view(pred_boxes[b].size()[:3])
                noobj_mask[b][anch_ious_max > self.ignore_threshold] = 0
        return noobj_mask, pred_boxes
    
   

def weights_init(net, init_type='normal', init_gain = 0.02):
    pass

    
def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def step_lr(lr, decay_rate, step_size, iters):
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
    step_size   = total_iters / step_num
    func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

