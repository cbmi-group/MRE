import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score
from scipy.spatial.distance import directed_hausdorff
from sklearn.utils.multiclass import type_of_target

import torch
import os

'''
metrics: input type ( CPU Tensor [num_imgs, c, h ,w] ) 
input 'seg' means final segmentation tensor, 'pred' means prediction tensor.
'''


def hausdorff_distance(seg, label):
    segmentation = seg.squeeze(1)
    mask = label.squeeze(1)

    n_imgs = segmentation.size()[0]
    total_dist = 0

    for i in range(n_imgs):
        non_zero_seg = np.transpose(np.nonzero(segmentation[i].cpu().numpy()))
        non_zero_mask = np.transpose(np.nonzero(mask[i].cpu().numpy()))
        h_dist = max(directed_hausdorff(non_zero_seg, non_zero_mask)[0],
                     directed_hausdorff(non_zero_mask, non_zero_seg)[0])
        total_dist += h_dist

    mean_dist = total_dist / n_imgs

    return mean_dist


def acc(seg, label):
    now_num = seg.shape[0]
    # seg, label = np.array(seg), np.array(label)
    seg_one = seg.reshape(-1)
    label_one = label.reshape(-1)

    label_T = label_one > 0
    corrects = torch.eq(seg_one, label_T).sum()
    all_num = seg_one.numel()

    # corrects = (seg.int() == label.int())
    acc = corrects / all_num
    return acc * now_num


def roc(pred, label):
    pred, label = np.array(pred), np.array(label)
    preds_roc = np.reshape(pred, -1)
    labels_roc = np.reshape(label, -1)
    fpr, tpr, thresholds = roc_curve(labels_roc, preds_roc)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def dice_cof(pred, label, reduce=False):
    matrix_sum = pred.int() + label.int()
    i = torch.sum(matrix_sum == 2, dim=(1, 2, 3))
    x1 = torch.sum(pred == 1, dim=(1, 2, 3))
    x2 = torch.sum(label == 1, dim=(1, 2, 3))
    dice_score = 2. * i.float() / (x1.float() + x2.float())
    if reduce:
        return torch.mean(dice_score)
    else:
        return torch.sum(dice_score)


def IoU(preds, labels, reduce=False):
    matrix_sum = preds.int() + labels.int()
    i = torch.sum(matrix_sum == 2, dim=(1, 2, 3))
    u = torch.sum(matrix_sum == 1, dim=(1, 2, 3))
    iou = i.float() / (i.float() + u.float() + 1e-9)
    if reduce:
        iou = torch.mean(iou)
    else:
        iou = torch.sum(iou)
    return iou


def acc_list(seg, label):
    total_acc = 0.0
    img_num = len(seg)
    for auc_index in range(img_num):
        now_pred = seg[auc_index]
        now_labels = label[auc_index]
        val_acc = acc(now_pred[0], now_labels[0])
        total_acc += val_acc

    return total_acc


def dIoU(preds, labels, reduce=False):
    matrix_sum = preds.int() + labels.int()
    i = torch.sum(matrix_sum == 2)
    u = torch.sum(matrix_sum == 1)
    iou = i.float() / (i.float() + u.float() + 1e-9)
    if reduce:
        iou = torch.mean(iou)
    else:
        iou = iou
    return iou


def mIoU(preds, labels, reduce=False):
    matrix_sum = preds.int() + labels.int()
    f_i = torch.sum(matrix_sum == 2, dim=(1, 2, 3))
    u = torch.sum(matrix_sum == 1, dim=(1, 2, 3))
    b_i = torch.sum(matrix_sum == 0, dim=(1, 2, 3))
    f_iou = f_i.float() / (f_i.float() + u.float() + 1e-9)
    b_iou = b_i.float() / (b_i.float() + u.float() + 1e-9)
    miou = 0.5 * (f_iou + b_iou)
    if reduce:
        miou = torch.mean(miou)
    else:
        miou = torch.sum(miou)
    return miou


def dmIoU(preds, labels, reduce=False):
    matrix_sum = preds.int() + labels.int()
    f_i = torch.sum(matrix_sum == 2)
    u = torch.sum(matrix_sum == 1)
    b_i = torch.sum(matrix_sum == 0)
    f_iou = f_i.float() / (f_i.float() + u.float() + 1e-9)
    b_iou = b_i.float() / (b_i.float() + u.float() + 1e-9)
    miou = 0.5 * (f_iou + b_iou)
    if reduce:
        miou = torch.mean(miou)
    else:
        miou = miou
    return miou


def F1_score(pred, label, reduce=False):
    pred, label = pred.int(), label.int()
    p = torch.sum((label == 1).int(), dim=(1, 2, 3))
    tp = torch.sum((pred == 1).int() & (label == 1).int(), dim=(1, 2, 3))
    fp = torch.sum((pred == 1).int() & (label == 0).int(), dim=(1, 2, 3))
    recall = tp.float() / (p.float() + 1e-9)
    precision = tp.float() / (tp.float() + fp.float() + 1e-9)
    f1 = (2 * recall * precision) / (recall + precision + 1e-9)
    if reduce:
        f1 = torch.mean(f1)
    else:
        f1 = torch.sum(f1)
    return f1


def dF1_score(pred, label, reduce=False):
    pred, label = pred.int(), label.int()
    p = torch.sum((label == 1).int())
    tp = torch.sum((pred == 1).int() & (label == 1).int())
    fp = torch.sum((pred == 1).int() & (label == 0).int())
    recall = tp.float() / (p.float() + 1e-9)
    precision = tp.float() / (tp.float() + fp.float() + 1e-9)
    f1 = (2 * recall * precision) / (recall + precision + 1e-9)
    if reduce:
        f1 = torch.mean(f1)
    else:
        f1 = f1
    return f1
