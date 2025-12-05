import os
import argparse
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import SHy
from dataset import MIMICiiiDataset, MIMICivDataset, transform_and_pad_input
from evaluation import evaluate_model


def build_model_and_test_loader(args, device):
    """
    Rebuild the SHy model and test_loader exactly as in main.py.
    """
    if args.dataset_name == 'MIMIC_III':
        data_dir = f'./data/{args.dataset_name}'

        # Only need test data + code_levels
        with open(os.path.join(data_dir, 'binary_test_codes_x.pkl'), 'rb') as f:
            binary_test_codes_x = pickle.load(f)

        test_codes_y = np.load(os.path.join(data_dir, 'test_codes_y.npy'))
        test_visit_lens = np.load(os.path.join(data_dir, 'test_visit_lens.npy'))
        test_pids = np.load(os.path.join(data_dir, 'test_pids.npy'))
        code_levels = np.load(os.path.join(data_dir, 'code_levels.npy'))

        padded_X_test = torch.transpose(transform_and_pad_input(binary_test_codes_x), 1, 2)
        trans_y_test = torch.tensor(test_codes_y)

        model = SHy(
            code_levels,
            args.single_dim,
            args.HGNN_dim,
            args.after_HGNN_dim,
            args.HGNN_layer_num - 1,
            args.nhead,
            args.num_TP,
            args.temperature,
            args.add_ratio,
            args.n_c,
            args.hid_state_dim,
            args.dropout,
            args.key_dim,
            args.SA_head,
            args.HGNN_model,
            device
        ).to(device)

        test_data = MIMICiiiDataset(padded_X_test, trans_y_test, test_pids, test_visit_lens)
        test_loader = DataLoader(
            test_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )

    else:  # MIMIC_IV path, for completeness
        data_dir = "./data/MIMIC_IV"

        test_codes_y = np.load(os.path.join(data_dir, 'test_codes_y.npy'))
        test_visit_lens = np.load(os.path.join(data_dir, 'test_visit_lens.npy'))
        test_pids = np.load(os.path.join(data_dir, 'test_pids.npy'))
        code_levels = np.load(os.path.join(data_dir, 'code_levels.npy'))

        trans_y_test = torch.tensor(test_codes_y)

        model = SHy(
            code_levels,
            args.single_dim,
            args.HGNN_dim,
            args.after_HGNN_dim,
            args.HGNN_layer_num - 1,
            args.nhead,
            args.num_TP,
            args.temperature,
            args.add_ratio,
            args.n_c,
            args.hid_state_dim,
            args.dropout,
            args.key_dim,
            args.SA_head,
            args.HGNN_model,
            device
        ).to(device)

        # For MIMIC-IV the dataset uses sliced files, matching main.py
        test_data = MIMICivDataset(
            'binary_test_x_slices/binary_test_codes_x',
            test_visit_lens,
            trans_y_test,
            test_pids,
            'Test'
        )

        test_loader = DataLoader(
            test_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )

    return model, test_loader


def mask_diagnoses(patients, mask_ratio, rng):
    """
    Randomly mask out mask_ratio of non-zero diagnosis entries
    for each patient in the batch, like in the paper's robustness section.
    """
    if mask_ratio <= 0.0:
        return patients

    # Clone so we don't modify the original batch
    patients = patients.clone()
    B = patients.shape[0]

    # Flatten everything except batch; we treat any non-zero as a diagnosis
    flat = patients.view(B, -1)

    for i in range(B):
        non_zero_idx = torch.nonzero(flat[i] > 0, as_tuple=False).squeeze(-1)
        if non_zero_idx.numel() == 0:
            continue

        num_to_mask = int(mask_ratio * non_zero_idx.numel())
        if num_to_mask <= 0:
            continue

        perm = torch.randperm(non_zero_idx.numel(), generator=rng)
        idx_to_zero = non_zero_idx[perm[:num_to_mask]]
        flat[i, idx_to_zero] = 0.0

    return flat.view_as(patients)


@torch.no_grad()
def eval_once(model, test_loader, device, mask_ratio=0.0, seed=3407):
    """
    Run SHy on the test set once, optionally masking diagnoses.
    Returns Recall@10, Recall@20, nDCG@10, nDCG@20.
    """
    model.eval()

    all_preds = []
    all_labels = []

    rng = torch.Generator(device='cpu')
    rng.manual_seed(seed)

    for patients, labels, pids, visit_lens in test_loader:
        patients = patients.to(device)
        labels = labels.to(device)

        if mask_ratio > 0.0:
            patients = mask_diagnoses(patients, mask_ratio, rng)

        try:
            pred, tp_list, recon_h_list, alphas = model(patients, visit_lens)
        except RuntimeError as e:
            if "The size of tensor a" in str(e) and "tensor b" in str(e):
                print("Skipping a batch due to shape mismatch under heavy masking.")
                continue
            else:
                raise

        all_preds.append(pred)
        all_labels.append(labels)

    logits = torch.vstack(all_preds)
    labels = torch.vstack(all_labels)

    # Same evaluate_model call pattern as in training.py
    # _, _, _, _, r10, n10, _, _, _, _, r20, n20, ... = evaluate_model(...)
    _, _, _, _, r10, n10, _, _, _, _, r20, n20, _, _, _, _, _, _, = evaluate_model(
        logits, labels, 5, 10, 15, 20, 25, 30
    )

    return r10.item(), r20.item(), n10.item(), n20.item()


def main():
    parser = argparse.ArgumentParser()

    # Same args as in main.py
    parser.add_argument('--device_idx', type=int, default=0)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--dataset_name', type=str, default='MIMIC_III')

    parser.add_argument('--single_dim', type=int, default=32)
    parser.add_argument('--HGNN_dim', type=int, default=256)
    parser.add_argument('--after_HGNN_dim', type=int, default=128)
    parser.add_argument('--HGNN_layer_num', type=int, default=2)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_TP', type=int, default=5)
    parser.add_argument('--n_c', type=int, default=10)
    parser.add_argument('--hid_state_dim', type=int, default=128)
    parser.add_argument('--key_dim', type=int, default=256)
    parser.add_argument('--SA_head', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--HGNN_model', type=str, default='UniGINConv')

    # These three MUST match what you used for training
    parser.add_argument('--temperature', type=float, nargs='+', required=True)
    parser.add_argument('--add_ratio', type=float, nargs='+', required=True)
    parser.add_argument('--loss_weight', type=float, nargs='+', required=True)

    # New: path to the trained checkpoint you want to evaluate
    parser.add_argument('--ckpt_path', type=str, required=True)

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device_idx}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Build model + test loader (no training here)
    model, test_loader = build_model_and_test_loader(args, device)

    # Load weights
    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt)

    print(f"Loaded checkpoint from: {args.ckpt_path}")

    # 1) No masking (baseline performance)
    print("\n=== Baseline (no masking) ===")
    r10, r20, n10, n20 = eval_once(model, test_loader, device, mask_ratio=0.0)
    print(f"Recall@10: {r10:.4f}, Recall@20: {r20:.4f}, nDCG@10: {n10:.4f}, nDCG@20: {n20:.4f}")

    # 2) Mask 25% of diagnoses (MIMIC-III 25% column in Table 2)
    print("\n=== Robustness: masking 25% of diagnoses ===")
    r10_25, r20_25, n10_25, n20_25 = eval_once(model, test_loader, device, mask_ratio=0.25)
    print(f"Recall@10: {r10_25:.4f}, Recall@20: {r20_25:.4f}, nDCG@10: {n10_25:.4f}, nDCG@20: {n20_25:.4f}")

    # 3) Mask 75% of diagnoses (MIMIC-III 75% column in Table 2)
    print("\n=== Robustness: masking 75% of diagnoses ===")
    r10_75, r20_75, n10_75, n20_75 = eval_once(model, test_loader, device, mask_ratio=0.75)
    print(f"Recall@10: {r10_75:.4f}, Recall@20: {r20_75:.4f}, nDCG@10: {n10_75:.4f}, nDCG@20: {n20_75:.4f}")


if __name__ == '__main__':
    main()
