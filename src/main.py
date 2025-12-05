import os
import torch
import warnings
import argparse
import numpy as np
import pickle as pickle
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import SHy
from training import train
from dataset import *
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    # Settle down all hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_idx', type=int, default=0, help="GPU index")
    parser.add_argument('--seed', type=int, default=3407, help="random seed")
    parser.add_argument('--dataset_name', type=str, default='MIMIC_III', help="experiment dataset")
    parser.add_argument('--single_dim', type=int, default=32, help="embedding dimension of one ICD-9 level")
    parser.add_argument('--HGNN_dim', type=int, default=256, help="hidden dim in HGNN")
    parser.add_argument('--after_HGNN_dim', type=int, default=128, help="hidden dim after HGNN")
    parser.add_argument('--HGNN_layer_num', type=int, default=2, help="number of HGNN layers")
    parser.add_argument('--nhead', type=int, default=4, help="number of heads in HGNN")
    parser.add_argument('--num_TP', type=int, default=5, help="number of temporal phenotypes")
    parser.add_argument('--n_c', type=int, default=10, help="number of cosine weight vectors")
    parser.add_argument('--hid_state_dim', type=int, default=128, help="temporal phenotype embedding dim")
    parser.add_argument('--key_dim', type=int, default=256,  help="key dim for self attention")
    parser.add_argument('--SA_head', type=int, default=8,  help="number of heads for self-attention")
    parser.add_argument('--dropout', type=float, default=0.001,  help="dropout ratio")
    parser.add_argument('--batch_size', type=int, default=128, help="batch size")
    parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
    parser.add_argument('--num_epoch', type=int, default=100, help="number of epochs")
    parser.add_argument('--early_stop_range', type=int, default=10, help="early stop threshold for training process")
    parser.add_argument('--HGNN_model', type=str, default='UniGINConv', help="which hypergraph nn to use")
    parser.add_argument('--temperature', type=float, nargs='+')
    parser.add_argument('--add_ratio', type=float, nargs='+')
    parser.add_argument('--loss_weight', type=float, nargs='+')
    args = parser.parse_args()
    objective_rates = args.loss_weight
    device = torch.device(f"cuda:{args.device_idx}" if torch.cuda.is_available() else "cpu")

    # Set random seed and directory information; create directories for saving training results
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    currentDateAndTime = datetime.now()
    model_directory = f'{currentDateAndTime.strftime("%m_%d_%YM%H_%M_%S")}__{args.seed}__{args.dataset_name}'
    model_path = os.path.join('../saved_models/', model_directory)
    os.mkdir(model_path)
    log_path = os.path.join('../training_logs/', model_directory)
    os.mkdir(log_path)

    # Load all data and further preprocess them to get the desirable format
    if args.dataset_name == 'MIMIC_III':
        with open(f'../data/{args.dataset_name}/binary_train_codes_x.pkl', 'rb') as f0:
            binary_train_codes_x = pickle.load(f0)

        with open(f'../data/{args.dataset_name}/binary_test_codes_x.pkl', 'rb') as f1:
            binary_test_codes_x = pickle.load(f1)

        train_codes_y = np.load(f'../data/{args.dataset_name}/train_codes_y.npy')
        train_visit_lens = np.load(f'../data/{args.dataset_name}/train_visit_lens.npy')
        test_codes_y = np.load(f'../data/{args.dataset_name}/test_codes_y.npy')
        test_visit_lens = np.load(f'../data/{args.dataset_name}/test_visit_lens.npy')
        code_levels = np.load(f'../data/{args.dataset_name}/code_levels.npy')
        train_pids = np.load(f'../data/{args.dataset_name}/train_pids.npy')
        test_pids = np.load(f'../data/{args.dataset_name}/test_pids.npy')
        padded_X_train = torch.transpose(transform_and_pad_input(binary_train_codes_x), 1, 2)
        padded_X_test = torch.transpose(transform_and_pad_input(binary_test_codes_x), 1, 2)
    else:
        data_dir = "../data/MIMIC_IV"
        train_codes_y = np.load(f'{data_dir}/train_codes_y.npy')
        train_visit_lens = np.load(f'{data_dir}/train_visit_lens.npy')
        train_pids = np.load(f'{data_dir}/train_pids.npy')
        test_codes_y = np.load(f'{data_dir}/test_codes_y.npy')
        test_visit_lens = np.load(f'{data_dir}/test_visit_lens.npy')
        test_pids = np.load(f'{data_dir}/test_pids.npy')
        code_levels = np.load(f'{data_dir}/code_levels.npy')

    trans_y_train = torch.tensor(train_codes_y)
    trans_y_test = torch.tensor(test_codes_y)
    class_num = train_codes_y.shape[1]

    # Initialize model and data loaders
    model = SHy(code_levels, args.single_dim, args.HGNN_dim, args.after_HGNN_dim, args.HGNN_layer_num-1, args.nhead, args.num_TP, args.temperature,
                args.add_ratio, args.n_c, args.hid_state_dim, args.dropout, args.key_dim, args.SA_head, args.HGNN_model, device).to(device)
    print(f'Number of parameters of this model: {sum(param.numel() for param in model.parameters())}')
    if args.dataset_name == 'MIMIC_III':
        training_data = MIMICiiiDataset(padded_X_train, trans_y_train, train_pids, train_visit_lens)
        train_loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
        test_data = MIMICiiiDataset(padded_X_test, trans_y_test, test_pids, test_visit_lens)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    else:
        training_data = MIMICivDataset('binary_train_x_slices/binary_train_codes_x', train_visit_lens, trans_y_train, train_pids, 'Train')
        train_loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
        test_data = MIMICivDataset('binary_test_x_slices/binary_test_codes_x', test_visit_lens, trans_y_test, test_pids, 'Test')
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)

    # Start training
    r2_list, r4_list, n2_list, n4_list, test_loss_per_epoch, train_average_loss_per_epoch, prediction_loss_per_epoch = train(
        model, args.lr, args.num_epoch, train_loader, test_loader, model_directory, args.early_stop_range, objective_rates, device)

    # Save all results
    with open(f'../training_logs/{model_directory}/r2_list.pkl', 'wb') as f101:
        pickle.dump(r2_list, f101)

    with open(f'../training_logs/{model_directory}/r4_list.pkl', 'wb') as f103:
        pickle.dump(r4_list, f103)

    with open(f'../training_logs/{model_directory}/n2_list.pkl', 'wb') as f107:
        pickle.dump(n2_list, f107)

    with open(f'../training_logs/{model_directory}/n4_list.pkl', 'wb') as f109:
        pickle.dump(n4_list, f109)

    with open(f'../training_logs/{model_directory}/train_average_loss_per_epoch.pkl', 'wb') as f112:
        pickle.dump(train_average_loss_per_epoch, f112)

    with open(f'../training_logs/{model_directory}/test_loss_per_epoch.pkl', 'wb') as f113:
        pickle.dump(test_loss_per_epoch, f113)

    with open(f'../training_logs/{model_directory}/prediction_loss_per_epoch.pkl', 'wb') as f114:
        pickle.dump(prediction_loss_per_epoch, f114)

    plt.plot(train_average_loss_per_epoch, 'r', label="Train")
    plt.plot(test_loss_per_epoch, 'b', label="Test")
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig(f"../training_logs/{model_directory}/total_loss_plot.svg")
    plt.clf()

    plt.plot(prediction_loss_per_epoch, 'r', label="Test")
    plt.ylabel('Prediction Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig(f"../training_logs/{model_directory}/prediction_loss_plot.svg")
    plt.clf()

    plt.plot(r4_list, 'r', label="k=20")
    plt.ylabel('Recall')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig(f"../training_logs/{model_directory}/recall_plot.svg")
    plt.clf()

    plt.plot(n4_list, 'r', label="k=20")
    plt.ylabel('nDCG')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig(f"../training_logs/{model_directory}/ndcg_plot.svg")
    plt.clf()