# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torchvision
import cv2
                # B - G - R
RED             = (0, 0, 255)
GREEN           = (0, 255, 0)
DARK_GREEN      = (115, 181, 34)
BLUE            = (255, 0, 0)
CYAN            = (255, 128, 0)
YELLOW          = (0, 255, 255)
ORANGE          = (0, 165, 255)
PURPLE          = (255, 0, 255)
PINK            = (180, 105, 255)

Black           = (0,0,0)
LightCoral      = (128,240,128)
Chartreuse      = (0,127,255)
Magenta         = (255,255,0)

# SBC_colors = [ORANGE, RED, CYAN, DARK_GREEN, GREEN, BLUE, YELLOW, PURPLE, PINK]
SBC_colors = [ORANGE, ORANGE, ORANGE, RED, RED, RED, CYAN, CYAN, CYAN]

KPS_colors = [DARK_GREEN, DARK_GREEN, YELLOW, YELLOW, PINK]

lane_class_colors = {
    1 : Black        ,#  'fishline'                # 鱼骨线
    2 : Black        ,#  'variable_line'           # 可变车道
    3 : Black        ,#  'parking'                 # 停车位线
    4 : RED          ,#  'SingleSolid'             # 单实线
    5 : GREEN        ,#  'SingleDotted'            # 单虚线
    6 : RED          ,#  'ForkSolid'               # 分歧线
    7 : BLUE         ,#  'Roadline'                # 路边
    8 : YELLOW       ,#  'DoubleSolid'             # 双实线
    9 : PURPLE       ,#  'DoubleDotted'            # 双虚线
    10: DARK_GREEN   ,#  'SolidDotted'             # 左实右虚
    11: CYAN         ,#  'DottedSolid'             # 左虚右实
    12: ORANGE       ,#  'Fence'                   # 护栏
    13: LightCoral   ,#  'DoubleSingleSolid'       # 双单实线
    14: Chartreuse   ,#  'DoubleSingleDotted'      # 双单虚线
    15: Magenta      ,#  'ShortDotted'             # 短虚线
    16: Black        ,#  'Ignore'                  # 忽略的线
}

# lane_class_colors = {
#     1 : Black        ,#  'fishline'                # 鱼骨线
#     2 : Black        ,#  'variable_line'           # 可变车道
#     3 : Black        ,#  'parking'                 # 停车位线
#     4 : RED          ,#  'SingleSolid'             # 单实线
#     5 : GREEN        ,#  'SingleDotted'            # 单虚线
#     6 : Black          ,#  'ForkSolid'               # 分歧线
#     7 : Black         ,#  'Roadline'                # 路边
#     8 : Black       ,#  'DoubleSolid'             # 双实线
#     9 : Black       ,#  'DoubleDotted'            # 双虚线
#     10: Black   ,#  'SolidDotted'             # 左实右虚
#     11: Black         ,#  'DottedSolid'             # 左虚右实
#     12: Black       ,#  'Fence'                   # 护栏
#     13: Black   ,#  'DoubleSingleSolid'       # 双单实线
#     14: Black   ,#  'DoubleSingleDotted'      # 双单虚线
#     15: Black      ,#  'ShortDotted'             # 短虚线
#     16: Black        ,#  'Ignore'                  # 忽略的线
# }
def save_batch_image_with_curves(batch_image,
                                batch_curves,
                                batch_labels,
                                file_name,
                                nrow=2,
                                padding=2,
                                **kwargs):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    # print(file_name)
    B, C, H, W = batch_image.size()
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0

    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            boxes = batch_curves[k]
            labels = batch_labels[k]
            # print(labels)
            num_box = boxes.shape[0]
            i = 0
            for n in range(num_box):
                lane = boxes[:, 3:][n]
                xs = lane[:len(lane) // 2]
                ys = lane[len(lane) // 2:]
                ys = ys[xs >= 0] * H
                xs = xs[xs >= 0] * W
                cls = int(labels[n].cpu())
                # if (cls > 0 and cls < 10):
                if cls > 0:
                    for jj, xcoord, ycoord in zip(range(xs.shape[0]), xs, ys):
                        j_x = x * width + padding + xcoord
                        j_y = y * height + padding + ycoord
                        cv2.circle(ndarr, (int(j_x), int(j_y)), 2, lane_class_colors[cls], 2)
                    i += 1
            k = k + 1
    cv2.imwrite(file_name, ndarr)

def save_batch_image_with_dbs(batch_image,
                              batch_curves,
                              batch_labels,
                              file_name,
                              nrow=2,
                              padding=2,
                              **kwargs):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    # print(file_name)
    B, C, H, W = batch_image.size()
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            pred = batch_curves[k].cpu().numpy()  # 10 7
            labels = batch_labels[k].cpu().numpy()  # 10
            # pred = pred[labels >= 1] # only draw lanes
            num_pred = pred.shape[0]
            if num_pred > 0:
                for n, lane in enumerate(pred):
                    cls = int(labels[n])
                    if cls == 0 :
                        continue
                    lane = lane[1:]
                    lower, upper = lane[0], lane[1]
                    lane = lane[2:]
                    ys = np.linspace(lower, upper, num=100)
                    
                    points = np.zeros((len(ys), 2), dtype=np.int32)
                    points[:, 1] = (ys * H).astype(int)
                    # Calculate the predicted xs
                    # points[:, 0] = ((lane[0]*ys**5 + lane[1]*ys**4 + lane[2]*ys**3 + lane[3]*ys**2 + lane[4]*ys + lane[5])* W).astype(int)
                    
                    points[:, 0] = ((lane[0] / (ys - lane[1]) ** 2 + lane[2] / (ys - lane[1]) + lane[3] + lane[4] * ys - lane[1]*lane[4])* W).astype(int)
                    points = points[(points[:, 0] > 0) & (points[:, 0] < W)]
                    points[:, 0] += x * width + padding
                    points[:, 1] += y * height + padding
                    for current_point, next_point in zip(points[:-1], points[1:]):
                        cv2.line(ndarr, tuple(current_point), tuple(next_point), color=lane_class_colors[cls], thickness=2)
            k = k + 1
    cv2.imwrite(file_name, ndarr)

def save_debug_images_boxes(input, tgt_curves, tgt_labels,pred_curves, pred_labels, prefix=None,**kwargs):
    save_batch_image_with_curves(input, tgt_curves, tgt_labels,'{}_gt.jpg'.format(prefix),**kwargs)
    save_batch_image_with_dbs(input, pred_curves, pred_labels,'{}_pred.jpg'.format(prefix),**kwargs)

