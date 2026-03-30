import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

def calc_iou(bbox1, bbox2):
    """
    Assuming bbox in xmin, ymin, xmax, ymax format
    """
    ixmin = max(bbox1[0], bbox2[0])
    ixmax = min(bbox1[2], bbox2[2])
    iymin = max(bbox1[1], bbox2[1])
    iymax = min(bbox1[3], bbox2[3])

    iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
    ih = np.maximum(iymax - iymin + 1.0, 0.0)
    intersection = iw * ih
    
    union = ((bbox1[2] - bbox1[0] + 1.0) * (bbox1[3] - bbox1[1] + 1.0) + 
             (bbox2[2] - bbox2[0] + 1.0) * (bbox2[3] - bbox2[1] + 1.0) - intersection)
    iou = intersection / union
    return iou

def calc_box_feats(bbox1, bbox2):
    """
    Assuming bbox1, bbox2 are in the xmin, ymin, xmax, ymax format
    """
    xmin_1, ymax_1 = bbox1[0], bbox1[3]
    xmin_2, ymax_2 = bbox2[0], bbox2[3]
    width_1, height_1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
    width_2, height_2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]

    y_rel_dist = 2 * (ymax_1 - ymax_2) / (height_1 + height_2)
    x_rel_dist = 2 * (xmin_1 - xmin_2) / (height_1 + height_2)
    y_rel_size = np.log((height_1 + 1e-7) / height_2)
    x_rel_size = np.log((width_1 + 1e-7) / width_2)
    return [y_rel_dist, x_rel_dist, y_rel_size, x_rel_size]
    
def calc_geom_feats(src_bboxes, dst_bboxes):
    edge_attr = np.zeros((src_bboxes.shape[0] * dst_bboxes.shape[0], 5), dtype = np.float32)
    ind = 0
    for i in range(src_bboxes.shape[0]):
        for j in range(dst_bboxes.shape[0]):
            bbox1 = src_bboxes[i].copy()
            bbox2 = dst_bboxes[j].copy()
            pos_enc = calc_box_feats(bbox1, bbox2)
            iou = calc_iou(bbox1, bbox2)
            edge_attr[ind, :4] = pos_enc
            edge_attr[ind, 4] = iou
            ind += 1
    edge_attr = torch.from_numpy(edge_attr)
    return edge_attr

def calc_edge_attr(obs_batch):
    seq_len, k = obs_batch.shape[0], obs_batch.shape[1]
    edge_attr = np.zeros(((seq_len - 1), k * k, 5), dtype = np.float32)
    for frame in range(edge_attr.shape[0]):
        ind = 0
        for i in range(obs_batch[frame].shape[0]):
            for j in range(obs_batch[frame + 1].shape[0]):
                bbox1 = obs_batch[frame][i].copy()
                bbox2 = obs_batch[frame + 1][j].copy()
                bbox1[2:4] += bbox1[0:2]
                bbox2[2:4] += bbox2[0:2]
                pos_enc = calc_box_feats(bbox1, bbox2)
                iou = calc_iou(bbox1, bbox2)
                edge_attr[frame][ind, :4] = pos_enc
                edge_attr[frame][ind, 4] = iou
                ind += 1
    edge_attr = torch.from_numpy(edge_attr)
    edge_dim = edge_attr.shape[-1]
    edge_attr = edge_attr.view(-1, edge_dim)
    return edge_attr

def calc_perm(As):
    P = As[-1]
    for i in range(len(As) - 2, -1, -1):
        P = torch.matmul(P, As[i])
    return P

def get_projection(P, k, d):
    H = torch.zeros(d*k, d*k*2).cuda()
    H[0:k, 0:k] = P[0:k, :]
    H[k:, k:2*k] = P[0:k, :]
    return H

def sinkhorn(log_alpha, n_iters = 20):
    k = log_alpha.shape[1]
    log_alpha = log_alpha.view(-1, k, k)
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim = 2, keepdim = True).view(-1, k, 1)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim = 1, keepdim = True).view(-1, 1, k)
    return log_alpha.exp()

# def sinkhorn_inference(matrix):
#     num_rows, num_cols = matrix.shape
#     desired_row_sums = torch.ones(num_rows, 1)
#     desired_col_sums = torch.ones(1, num_cols)
#     desired_row_sums[-1] = num_cols - 1
#     desired_col_sums[0, -1] = num_rows - 1
#     for _ in range(20):
#         matrix = torch.log(desired_row_sums) + matrix - torch.logsumexp(matrix, 1, keepdims=True)
#         matrix = torch.log(desired_col_sums) + matrix - torch.logsumexp(matrix, 0, keepdims=True)
#     return matrix

def vis(save_dir, seq, tracks, meas_batch, state_pred, state, A_soft, A_hard, P_hard, P_gt):

    count, d = 0, 2
    k = meas_batch.shape[1] // d
    start_frame, end_frame = tracks[:, 0].min(), tracks[:, 0].max() + 1

    for frame in range(start_frame, end_frame):
        img = cv2.imread('../MOT15/train/{}/img1/{:06d}.jpg'.format(seq, frame))
        img_height, img_width, _ = img.shape
        cv2.putText(img, '{:04d}'.format(frame), (20, 50), 0, 1.5, (255, 0, 255), thickness=2)

        bboxes = tracks[tracks[:, 0] == frame, 2:6]
        for i in range(bboxes.shape[0]):
            x, y, w, h = bboxes[i]
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), thickness=2)
            
        for ID in range(k):
            y_pred, x_pred = state_pred[count][:k*d][ID].item(), state_pred[count][:k*d][ID + k].item()
            y_pred, x_pred = int(y_pred), int(x_pred)
            cv2.line(img, (x_pred, y_pred+10), (x_pred, y_pred-10), (0, 255, 0), thickness=2)
            cv2.line(img, (x_pred-10, y_pred), (x_pred+10, y_pred), (0, 255, 0), thickness=2)

            #White color
            y, x = state[count][:k*d][ID].item(), state[count][:k*d][ID + k].item()
            y, x = int(y), int(x)
            cv2.circle(img, (x, y), 10, (255, 255, 255), thickness=2)   
            cv2.putText(img, str(ID), (x-10, y-20), 0, 1, (255, 255, 255), thickness=2)

            y_meas, x_meas = meas_batch[count][ID].item(), meas_batch[count][ID + k].item()
            y_meas, x_meas = int(y_meas), int(x_meas)
            cv2.line(img, (x_meas, y_meas+10), (x_meas, y_meas-10), (0, 0, 255), thickness=2)
            cv2.line(img, (x_meas-10, y_meas), (x_meas+10, y_meas), (0, 0, 255), thickness=2)
            cv2.putText(img, str(ID), (x_meas-10, y_meas+40), 0, 1, (0, 0, 255), thickness=2)
            
        length = int(np.floor(img_height / 2))
        if count == 0:
            A_img_soft = F.interpolate(torch.eye(k)[None, None], (length, length)).squeeze()
            A_img_soft = torch.stack([A_img_soft for _ in range(3)]).permute(1, 2, 0)
            A_img_soft = (255 * A_img_soft).numpy().astype(np.uint8)

            A_img_hard = A_img_soft
            P_img_hard = A_img_soft
            P_img_gt = A_img_soft
        else:
            A = A_soft[count - 1]
            A_img_soft = F.interpolate(A[None, None], (length, length)).squeeze()
            A_img_soft = torch.stack([A_img_soft for _ in range(3)]).permute(1, 2, 0)
            A_img_soft = (255 * A_img_soft).numpy().astype(np.uint8)

            A_img_hard = A_hard[count - 1].float()
            A_img_hard = F.interpolate(A_img_hard[None, None], (length, length)).squeeze()
            A_img_hard = torch.stack([A_img_hard for _ in range(3)]).permute(1, 2, 0)
            A_img_hard = (255 * A_img_hard).numpy().astype(np.uint8)

            P_img_hard = P_hard[count - 1].float()
            P_img_hard = F.interpolate(P_img_hard[None, None], (length, length)).squeeze()
            P_img_hard = torch.stack([P_img_hard for _ in range(3)]).permute(1, 2, 0)
            P_img_hard = (255 * P_img_hard).numpy().astype(np.uint8)

            P_img_gt = P_gt[count - 1]
            P_img_gt = F.interpolate(P_img_gt[None, None], (length, length)).squeeze()
            P_img_gt = torch.stack([P_img_gt for _ in range(3)]).permute(1, 2, 0)
            P_img_gt = (255 * P_img_gt).numpy().astype(np.uint8)
            
        vertical_sep = 128 * torch.stack([torch.ones(length, 1) for _ in range(3)]).permute(1, 2, 0)
        vertical_sep = (255 * vertical_sep).numpy().astype(np.uint8)
        
        img1 = np.concatenate((A_img_soft, vertical_sep, A_img_hard), axis=1)
        img2 = np.concatenate((P_img_hard, vertical_sep, P_img_gt), axis=1)

        horizontal_sep = 128 * torch.stack([torch.ones(1, img1.shape[1]) for _ in range(3)]).permute(1, 2, 0)
        horizontal_sep = (255 * horizontal_sep).numpy().astype(np.uint8)
        img3 = np.concatenate((img1, horizontal_sep, img2), axis=0)
        
        if img3.shape[0] - img.shape[0] == 1:
            img = np.concatenate((img, img3[:-1]), axis=1)
        elif img3.shape[0] == img.shape[0]:
            img = np.concatenate((img, img3), axis=1)
            
        cv2.imwrite(save_dir + '{:04d}.png'.format(frame), img)
        count += 1