import os
import cv2
import time
import pickle, joblib
import numpy as np
np.set_printoptions(suppress=True)

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_printoptions(sci_mode=False)
torch.backends.cuda.matmul.allow_tf32 = False

import sys
sys.path.insert(0, '/home/lishuai/Experiment/MOT/LPT/lib/deep-person-reid')
from torchreid.models.osnet import osnet_x0_5

from matplotlib import pyplot as plt
from utils import getIoU, computeBoxFeatures
from kalman import forward, backward, ll

def computeEdgeAttr(obs_batch):
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
                pos_enc = computeBoxFeatures(bbox1, bbox2)
                iou = getIoU(bbox1, bbox2)
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

def sinkhorn(log_alpha, n_iters = 20):
    k = log_alpha.shape[1]
    log_alpha = log_alpha.view(-1, k, k)
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim = 2, keepdim = True).view(-1, k, 1)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim = 1, keepdim = True).view(-1, 1, k)
    return log_alpha.exp()

def get_projection(P, k, d):
    H = torch.zeros(d*k, d*k*2).cuda()
    H[0:k, 0:k] = P[0:k, :]
    H[k:, k:2*k] = P[0:k, :]
    return H

def extract_feats(reid_net_, tensors_):
    with torch.no_grad():
        feats = reid_net_.featuremaps(tensors_)
        feats = reid_net_.global_avgpool(feats)
        feats = feats.view(feats.size(0), -1)
        feats = reid_net_.fc(feats)
        feats = feats.view(10, -1, 512)
        feats = F.normalize(feats, dim = -1)
    return feats

if __name__ == '__main__':

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    net_kf = nn.Sequential(nn.Linear(5, 5), nn.ReLU(), 
                           nn.Linear(5, 1))
    net_kf = net_kf.cuda()
    optimizer_kf = torch.optim.Adam(net_kf.parameters(), lr=1e-2)
    #optimizer_kf = torch.optim.Adam(net_kf.parameters(), lr=1e-3)

    net = nn.Sequential(nn.Linear(5, 5), nn.ReLU(), 
                        nn.Linear(5, 1))
    net = net.cuda()
    net.load_state_dict(net_kf.state_dict())
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    #optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    d = 2
    logSigQ = 5 * torch.Tensor([1]).cuda()
    sigQ, sigR = logSigQ.exp(), 5 #0.1

    #logSigQ = 5 * torch.Tensor([1])
    #sigQ, sigR = logSigQ.exp().item(), 5

    epoch_loss_list_kf, loss_list_kf = [], []
    epoch_loss_list, loss_list = [], []
    batch_size = 4 
    num_epochs = 10

    with open('data/FRCNN.pkl', 'rb') as f:
        train_list = pickle.load(f)

    start_time = time.time()
    num_videos = train_list.__len__()
    for epoch in range(num_epochs):
        epoch_loss_kf, batch_loss_kf = 0, 0
        epoch_loss, batch_loss = 0, 0
        
        for ind in range(num_videos):
            obs_batch, meas_batch, tensors, folder = train_list[ind]
            #import ipdb;ipdb.set_trace()
            meas_batch = torch.from_numpy(meas_batch).float().cuda()
            k = obs_batch.shape[1]
            
            edge_attr = computeEdgeAttr(obs_batch)
            edge_attr = edge_attr.cuda()
            
            state_init = torch.zeros(k*d*2, 1).cuda()
            state_init[:k*d, 0] = meas_batch[0]           # Initialise using first measurment
            cov_init = 300 * torch.eye(k*2*2).cuda()      # Initialise with high uncertainty

            #########################################Kalman Filtering#####################################
            logits_kf = net_kf(edge_attr)
            logits_kf = logits_kf.view(-1, k, k).permute(0, 2, 1)
            
            As_kf = sinkhorn(logits_kf)
            As_kf = torch.cat((torch.eye(k).unsqueeze(0).cuda(), As_kf))
            
            Ps_kf = [calc_perm(As_kf[:_]) for _ in range(1, obs_batch.shape[0] + 1)]
            Ps_kf = torch.stack(Ps_kf)
            Ht_list_kf = [get_projection(Ps_kf[_], k=k, d=2) for _ in range(Ps_kf.shape[0])]

            #import ipdb;ipdb.set_trace()
            state_kf, cov_kf, state_pred_kf, cov_pred_kf = forward(meas_batch.clone(), Ht_list_kf, state_init, cov_init,
                                                                   sigQ, sigR, k, d)

            cov_z_kf = [Ht_list_kf[i].matmul(cov_kf[i]).matmul(Ht_list_kf[i].T) for i in range(len(Ht_list_kf))]
            cov_z_kf = [(cov_z_kf[i] + cov_z_kf[i].T) / 2 for i in range(len(cov_z_kf))]
            z_x_kf = [Ht_list_kf[i].matmul(state_kf[i]) for i in range(len(Ht_list_kf))]

            if ind == 999:
                import ipdb;ipdb.set_trace()
            
            lls_kf = [ll(z_x_kf[i], cov_z_kf[i], meas_batch[i], sigR) for i in range(len(Ht_list_kf))]
            loss_kf = -torch.stack(lls_kf).mean()
            
            #import ipdb;ipdb.set_trace()
            batch_loss_kf += loss_kf
            epoch_loss_kf += loss_kf
            ###############################################################################################
            
            #########################################Kalman Smoothing######################################
            logits = net(edge_attr)
            logits = logits.view(-1, k, k).permute(0, 2, 1)
            
            As = sinkhorn(logits)
            As = torch.cat((torch.eye(k).unsqueeze(0).cuda(), As))
            
            Ps = [calc_perm(As[:_]) for _ in range(1, obs_batch.shape[0] + 1)]
            Ps = torch.stack(Ps)
            Ht_list = [get_projection(Ps[_], k=k, d=2) for _ in range(Ps.shape[0])]
            
            state, cov, state_pred, cov_pred = forward(meas_batch.clone(), Ht_list, state_init, cov_init, 
                                                       sigQ, sigR, k, d)
            smoothing = True
            if smoothing:
                state, cov = backward(state, cov, state_pred, cov_pred, k, d)
                
            cov_z = [Ht_list[i].matmul(cov[i]).matmul(Ht_list[i].T) for i in range(len(Ht_list))]
            cov_z = [(cov_z[i] + cov_z[i].T) / 2 for i in range(len(cov_z))]

            z_x = [Ht_list[i].matmul(state[i]) for i in range(len(Ht_list))]
            lls = [ll(z_x[i], cov_z[i], meas_batch[i], sigR) for i in range(len(Ht_list))]
            loss = -torch.stack(lls).mean()

            batch_loss += loss
            epoch_loss += loss
            ###############################################################################################
            
            if (ind > 0 and (ind + 1) % batch_size == 0) or (ind == num_videos - 1):
                
                if (ind + 1) % batch_size == 0:
                    batch_loss = batch_loss / batch_size
                    batch_loss_kf = batch_loss_kf / batch_size
                else:
                    batch_loss = batch_loss / (num_videos % batch_size)
                    batch_loss_kf = batch_loss_kf / (num_videos % batch_size)
                    
                loss_list_kf.append(batch_loss_kf.detach().cpu())
                loss_list.append(batch_loss.detach().cpu())
                print('Epoch {}, iteration [{}/{}] batch loss kf {:.2f} vs {:.2f}'.format(
                    epoch, ind, num_videos, batch_loss_kf, batch_loss))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss = 0
                
                optimizer_kf.zero_grad()
                loss_kf.backward()
                optimizer_kf.step()
                batch_loss_kf = 0

        #import ipdb;ipdb.set_trace()
        epoch_loss_list.append(epoch_loss.detach().cpu() / num_videos)
        epoch_loss_list_kf.append(epoch_loss_kf.detach().cpu() / num_videos)

        torch.save(net.state_dict(), 'ckpts/kskf/epoch-{:02d}.pth'.format(epoch))
        torch.save(net_kf.state_dict(), 'ckpts/kskf/kf-epoch-{:02d}.pth'.format(epoch))

    print('Elapsed {:.2f} seconds'.format(time.time() - start_time))
    import ipdb;ipdb.set_trace()

    # plt.figure(figsize=(6, 2))

    # plt.plot(torch.stack(epoch_loss_list_kf)[1:], color = 'r', label = 'Kalman filtering')
    # plt.plot(torch.stack(epoch_loss_list)[1:], label = 'Kalman smoothing')

    # #plt.plot(torch.stack(epoch_loss_list_kf)[1:] * 1e-3, color = 'r', label = 'Kalman filtering')
    # #plt.plot(torch.stack(epoch_loss_list)[1:] * 1e-3, label = 'Kalman smoothing')

    # plt.xticks(np.arange(len(epoch_loss_list_kf[1:])), np.arange(2, len(epoch_loss_list_kf[1:])+2))
    # plt.xlabel('# of epochs')
    # plt.ylabel('Loss')
    # plt.grid()
    # plt.legend()
    # plt.savefig('loss_curve.pdf', bbox_inches = 'tight')
    # plt.show()

    ###########################################Finetuning OSNet#################################################
    #anet = ANet()
    #anet = anet.cuda()

    #optimizer = torch.optim.Adam(anet.parameters(), lr=5e-2) 
    #optimizer = torch.optim.Adam(anet.parameters(), lr=1e-2) 
    #optimizer = torch.optim.Adam(anet.parameters(), lr=1e-3)

    reid_net_ = osnet_x0_5(pretrained=True)
    del reid_net_.classifier
    reid_net_ = reid_net_.cuda()
    optimizer = torch.optim.Adam(reid_net_.parameters(), lr=1e-4)

    temp = 0.1
    num_epochs = 3
    num_videos = train_list.__len__()
    epoch_loss_list, loss_list = [], []

    for epoch in range(num_epochs):
        epoch_loss, batch_loss = 0, 0
        
        for ind in range(num_videos):
            obs_batch, meas_batch, tensors, folder = train_list[ind]
            k = obs_batch.shape[1]
            
            with torch.no_grad():
                edge_attr = getGeomFeats(obs_batch)
                edge_attr = edge_attr.cuda()
                logits = net(edge_attr)
                logits = logits.view(-1, k, k).permute(0, 2, 1)
                probs = F.softmax(logits, dim = -1)

            As = sinkhorn(logits)
            As = torch.cat((torch.eye(k).unsqueeze(0).cuda(), As))

            Ps = [calc_perm(As[:_]) for _ in range(1, obs_batch.shape[0] + 1)]
            Ps = torch.stack(Ps)

            tensors = tensors.cuda()
            tensors_ = tensors.view(-1, 3, 128, 64)
            
            feats = reid_net_.featuremaps(tensors_)
            feats = reid_net_.global_avgpool(feats)
            feats = feats.view(feats.size(0), -1)
            feats = reid_net_.fc(feats)
            feats = feats.view(10, -1, 512)
            feats = F.normalize(feats, dim = -1)
        
            cos_sim_tensor = torch.zeros(9, k, k).cuda()
            for frame in range(1, 10):
                cos_sim_tensor[frame - 1] = torch.matmul(feats[frame], feats[0].T) / temp 
            cos_sim_tensor = F.log_softmax(cos_sim_tensor, dim = -1)
            loss = nn.KLDivLoss(reduction='batchmean')(cos_sim_tensor, target = Ps[1:]) / k
            
            plt.clf()
            plt.figure(figsize=(10, 2))
            plt.subplot(1, 3, 1)
            plt.grid()
            plt.title('Loss {:.2f}'.format(loss))
            plt.plot(loss_list)

            plt.subplot(1, 3, 2)
            plt.imshow(Ps[-1].detach().cpu(), cmap='gray')
            plt.title('Teacher network')
            plt.colorbar();

            plt.subplot(1, 3, 3)
            plt.imshow(cos_sim_tensor[-1].exp().detach().cpu(), cmap='gray')
            plt.title('Student network')
            plt.colorbar()
            
            plt.savefig('ckpts/imgs/' + '{}_{:03d}.jpg'.format(epoch, ind), bbox_inches = 'tight')
            display.clear_output(wait=True)
            display.display(plt.gcf())
            
            epoch_loss += loss        
            loss_list.append(loss.detach().cpu())            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            batch_loss = 0
            
        epoch_loss_list.append(epoch_loss.detach().cpu() / num_videos)
        torch.save(reid_net_.state_dict(), 'ckpts/epoch-{:02d}-app.pth'.format(epoch))