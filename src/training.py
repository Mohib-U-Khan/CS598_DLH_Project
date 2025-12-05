import torch
from loss import *
from evaluation import evaluate_model


def train(model, lrate, num_epoch, train_loader, test_loader, model_directory, early_stop_range, objective_rates, device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate)
    test_loss_per_epoch = []; train_average_loss_per_epoch = []; prediction_loss_per_epoch = []
    r2_list, r4_list, n2_list, n4_list = [], [], [], []
    for epoch in range(num_epoch):
        one_epoch_train_loss = []
        for i, (patients, labels, pids, visit_lens) in enumerate(train_loader):
            patients = patients.to(device); labels = labels.to(device)
            pred, tp_list, recon_h_list, alphas = model(patients, visit_lens)
            loss, _, _ = shy_loss(pred, labels.to(torch.float32), patients, recon_h_list, tp_list, alphas, visit_lens.tolist(), objective_rates, device)
            one_epoch_train_loss.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_average_loss_per_epoch.append((sum(one_epoch_train_loss) / len(one_epoch_train_loss)).item())
        print('Epoch: [{}/{}], Training Loss: {:.9f}'.format(epoch+1, num_epoch, train_average_loss_per_epoch[-1]))
        model.eval()
        pred_list = []; label_list = []; original_patient_list = []; test_tp_list = []; test_recon_list = []; visit_lens_list = []; alpha_list = []
        for (test_patients, test_labels, test_pids, test_visit_lens) in test_loader:
            test_patients = test_patients.to(device); test_labels = test_labels.to(device)
            with torch.no_grad():
              pred, tp_list, recon_h_list, alphas = model(test_patients, test_visit_lens)
            pred_list.append(pred); label_list.append(test_labels); original_patient_list.append(test_patients); alpha_list.append(alphas)
            test_tp_list += tp_list; test_recon_list += recon_h_list; visit_lens_list += test_visit_lens.tolist()
        pred = torch.vstack(pred_list); test_labels = torch.vstack(label_list); test_patients = torch.vstack(original_patient_list); test_alphas = torch.vstack(alpha_list)
        test_loss, loss_list, name_list = shy_loss(pred, test_labels.to(torch.float32), test_patients, test_recon_list, test_tp_list, test_alphas, visit_lens_list, objective_rates, device)
        test_loss_per_epoch.append(test_loss.item())
        prediction_loss_per_epoch.append(loss_list[0].item())
        _, _, _, _, metric_r2, metric_n2, _, _, _, _, metric_r4, metric_n4, _, _, _, _, _, _, = evaluate_model(pred, test_labels, 5, 10, 15, 20, 25, 30)
        r2_list.append(metric_r2); r4_list.append(metric_r4); n2_list.append(metric_n2); n4_list.append(metric_n4)
        print('Test Epoch {}: {:.9f} (recall@10); {:.9f} (recall@20); {:.9f} (ndcg@10); {:.9f} (ndcg@20)'.format(epoch+1, metric_r2, metric_r4, metric_n2, metric_n4))
        if len(test_tp_list[0].shape) > 2:
            print('{}: {:.9f}; {}: {:.9f}; {}: {:.9f}; {}: {:.9f}'.format(name_list[0], loss_list[0].item(), name_list[1], loss_list[1], name_list[2], loss_list[2].item(), name_list[3], loss_list[3].item()))
        else:
            print('{}: {:.9f}; {}: {:.9f}'.format(name_list[0], loss_list[0].item(), name_list[1], loss_list[1]))
        if epoch >= 30 and prediction_loss_per_epoch[-1] < min(prediction_loss_per_epoch[0:-1]):
           torch.save(model.state_dict(), f'../saved_models/{model_directory}/shy_epoch_{epoch+1}.pth')
        early_stop = (-1) * early_stop_range
        last_loss = prediction_loss_per_epoch[early_stop:]
        if epoch >= 30 and sorted(last_loss) == last_loss:
           break
        model.train()
    return r2_list, r4_list, n2_list, n4_list, test_loss_per_epoch, train_average_loss_per_epoch, prediction_loss_per_epoch