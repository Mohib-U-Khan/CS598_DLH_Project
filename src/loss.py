import torch
import torch.nn as nn


def shy_loss(pred, label, original_h, reconstruction, TPs, alphas, visit_lens, obj_r, device):
  # Loss 1: prediction loss
  criterion = nn.BCELoss()
  loss_1 = criterion(pred, label)
  # Loss 2: fidelity loss
  loss_2 = 0.0
  for k in range(len(TPs)):
    recon = torch.flatten(reconstruction[k])
    target = torch.flatten(original_h[k][:, 0:int(visit_lens[k])])
    loss_2 += criterion(recon, target)
  loss_2 = loss_2 / len(TPs)
  # Loss 3: distinctness loss
  if len(TPs[0].shape) > 2:
    loss_3 = 0.0
    Q = torch.eye(TPs[0].shape[0]).to(device)
    for j in range(len(TPs)):
      swap_tp = torch.swapaxes(TPs[j], 0, -1)
      loss_temp = 0.0
      for jj in range(len(swap_tp)):
        one_visit = swap_tp[jj]
        loss_temp += torch.norm(Q - torch.matmul(one_visit.t(), one_visit), p=2)
      loss_3 += loss_temp / len(swap_tp)
    loss_3 = loss_3 / len(TPs)
    # Loss 5: alpha loss
    loss_4 = torch.mean(torch.sqrt(torch.var(alphas, dim=1)) - torch.norm(alphas, p=2, dim=1))
    loss = obj_r[0] * loss_1 + obj_r[1] * loss_2 + obj_r[2] * loss_3 - obj_r[3] * loss_4
    loss_name_list = ['Prediction', 'Fidelity', 'Distinctness', 'Alpha']
    return loss, [loss_1, loss_2, loss_3, loss_4], loss_name_list
  else:
    loss = obj_r[0] * loss_1 + obj_r[1] * loss_2
    loss_name_list = ['Prediction', 'Fidelity']
    return loss, [loss_1, loss_2], loss_name_list