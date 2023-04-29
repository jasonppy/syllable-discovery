import math
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from .trainer_utils import AverageMeter


def calc_recalls_from_S_one_to_many_coarse(S, row_img_id, column_img_id):
    # image is row, audio is colum
    row = S.size(0)
    column = S.size(1)
    I2A_scores, I2A_ind = S.topk(100, 1)
    A2I_scores, A2I_ind = S.topk(100, 0)
    A_r1 = AverageMeter()
    A_r5 = AverageMeter()
    A_r10 = AverageMeter()
    A_r100 = AverageMeter()
    I_r1 = AverageMeter()
    I_r5 = AverageMeter()
    I_r10 = AverageMeter()
    I_r100 = AverageMeter()
    for i in range(row):
        A_foundind = -1
        for ind in range(100):
            if row_img_id[i] == column_img_id[I2A_ind[i, ind]]:
                A_foundind = ind
                break
        # do r1s
        if A_foundind == 0:
            A_r1.update(1)
        else:
            A_r1.update(0)
        # do r5s
        if A_foundind >= 0 and A_foundind < 5:
            A_r5.update(1)
        else:
            A_r5.update(0)
        # do r10s
        if A_foundind >= 0 and A_foundind < 10:
            A_r10.update(1)
        else:
            A_r10.update(0)
        # do r10s
        if A_foundind >= 0 and A_foundind < 100:
            A_r100.update(1)
        else:
            A_r100.update(0)

    for i in range(column):
        I_foundind = -1
        for ind in range(100):
            if column_img_id[i] == row_img_id[A2I_ind[ind, i]]:
                I_foundind = ind
                break
        # do r1s
        if I_foundind == 0:
            I_r1.update(1)
        else:
            I_r1.update(0)
        # do r5s
        if I_foundind >= 0 and I_foundind < 5:
            I_r5.update(1)
        else:
            I_r5.update(0)
        # do r10s
        if I_foundind >= 0 and I_foundind < 10:
            I_r10.update(1)
        else:
            I_r10.update(0)
        # do r100s
        if I_foundind >= 0 and I_foundind < 100:
            I_r100.update(1)
        else:
            I_r100.update(0)

    recalls = {'A_r1':A_r1.avg, 'A_r5':A_r5.avg, 'A_r10':A_r10.avg, 'A_r100':A_r100.avg,
                'I_r1':I_r1.avg, 'I_r5':I_r5.avg, 'I_r10':I_r10.avg, 'I_r100':I_r100.avg}
                #'A_meanR':A_meanR.avg, 'I_meanR':I_meanR.avg}

    return recalls   