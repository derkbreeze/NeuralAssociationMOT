import torch
import numpy as np

def forward(Z, Ht_list, state_init, cov_init, sigQ=0.1, sigR=0.1, k=10, d=2, dt=0.01):
    state = [state_init.clone()]
    cov = [cov_init.clone()]
    state_pred = [state_init.clone()]
    cov_pred = [cov_init.clone()]
    #import ipdb;ipdb.set_trace()
    Q = torch.zeros(2*d*k, 2*d*k).cuda()
    Q[d*k:, d*k:] = sigQ * torch.eye(d*k).cuda()
    
    F = torch.eye(2*d*k).cuda()
    for j in range(d*k):
        F[j, d*k+j] = dt
        
    for i in range(1, Z.shape[0]):
        state_pred.append(torch.matmul(F, state[i - 1]))            #Prediction mean
        cov_pred.append(torch.matmul(F, cov[i - 1]).matmul(F.T) + Q)#Prediction covariance

        H = Ht_list[i]                                              #Observation matrix
        R = sigR * torch.eye(H.shape[0]).cuda()                     #Observation covariance 
        S = torch.matmul(H, cov_pred[i]).matmul(H.T) + R            #Innovatioin matrix
        K = torch.matmul(cov_pred[i], H.T).matmul(torch.inverse(S)) #Kalman gain
        residual = Z[i].reshape(-1, 1) - torch.matmul(H, state_pred[i])
        state.append(state_pred[i] + torch.matmul(K, residual))

        I_KH = torch.eye(2*d*k).cuda() - torch.matmul(K, H)
        cov_post = torch.matmul(I_KH, cov_pred[i])
        cov.append(cov_post)
    return state, cov, state_pred, cov_pred

def backward(state, cov, state_pred, cov_pred, k=10, d=2, dt=0.01):
    F = torch.eye(2*d*k).cuda()
    for j in range(d*k):
        F[j, d*k+j] = dt
        
    for j in range(len(state) - 1, 0, -1):
        C = torch.matmul(cov[j], F.T).matmul(torch.inverse(cov_pred[j]))
        state[j] = state[j] + torch.matmul(C, (state[j] - state_pred[j]))
        cov[j] = cov[j] + torch.matmul(C, (cov[j] - cov_pred[j])).matmul(C.T)
    return state, cov


def ll(z, cov, meas, sigR):
    R = sigR * torch.eye(cov.shape[0]).cuda()
    #import ipdb;ipdb.set_trace()
    return torch.distributions.MultivariateNormal(meas, cov+R).log_prob(z.squeeze())

# def ll_(z, cov, meas, sigR):
#     R = sigR * torch.eye(cov.shape[0]).cuda()
#     cov_ = cov + R 
#     const = -0.5 * cov_.shape[0] * torch.Tensor([2 * torch.pi]).log().cuda() - 0.5 * cov_.det().log()

#     #loss = const - 0.5 * torch.linalg.multi_dot(( (z-meas[:, None]).T, cov_.inverse(), z-meas[:, None]))
#     loss = const - 0.5 * torch.linalg.multi_dot((z.squeeze()-meas, cov_.inverse(), z.squeeze()-meas))
#     return loss