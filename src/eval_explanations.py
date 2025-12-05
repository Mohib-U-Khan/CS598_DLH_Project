import os
import argparse
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import SHy
from dataset import MIMICiiiDataset, MIMICivDataset
import pickle as pickle


def build_model_and_test_loader(args):
    """
    Very similar to main.py, but:
    - we ONLY build the model + test loader (no training).
    """
    device = torch.device(f"cuda:{args.device_idx}" if torch.cuda.is_available() else "cpu")

    # ---------- Load data ----------
    if args.dataset_name == 'MIMIC_III':
        data_dir = f'./data/{args.dataset_name}'

        with open(os.path.join(data_dir, 'binary_train_codes_x.pkl'), 'rb') as f0:
            binary_train_codes_x = pickle.load(f0)
        with open(os.path.join(data_dir, 'binary_test_codes_x.pkl'), 'rb') as f1:
            binary_test_codes_x = pickle.load(f1)

        train_codes_y = np.load(os.path.join(data_dir, 'train_codes_y.npy'))
        train_visit_lens = np.load(os.path.join(data_dir, 'train_visit_lens.npy'))
        test_codes_y = np.load(os.path.join(data_dir, 'test_codes_y.npy'))
        test_visit_lens = np.load(os.path.join(data_dir, 'test_visit_lens.npy'))
        code_levels = np.load(os.path.join(data_dir, 'code_levels.npy'))
        train_pids = np.load(os.path.join(data_dir, 'train_pids.npy'))
        test_pids = np.load(os.path.join(data_dir, 'test_pids.npy'))

        # transform_and_pad_input is from dataset.py
        from dataset import transform_and_pad_input
        padded_X_train = torch.transpose(transform_and_pad_input(binary_train_codes_x), 1, 2)
        padded_X_test = torch.transpose(transform_and_pad_input(binary_test_codes_x), 1, 2)

        trans_y_train = torch.tensor(train_codes_y)
        trans_y_test = torch.tensor(test_codes_y)

        # Build datasets / loaders
        training_data = MIMICiiiDataset(padded_X_train, trans_y_train, train_pids, train_visit_lens)
        test_data = MIMICiiiDataset(padded_X_test, trans_y_test, test_pids, test_visit_lens)

    else:  # MIMIC_IV
        data_dir = "./data/MIMIC_IV"
        train_codes_y = np.load(os.path.join(data_dir, 'train_codes_y.npy'))
        train_visit_lens = np.load(os.path.join(data_dir, 'train_visit_lens.npy'))
        train_pids = np.load(os.path.join(data_dir, 'train_pids.npy'))

        test_codes_y = np.load(os.path.join(data_dir, 'test_codes_y.npy'))
        test_visit_lens = np.load(os.path.join(data_dir, 'test_visit_lens.npy'))
        test_pids = np.load(os.path.join(data_dir, 'test_pids.npy'))

        code_levels = np.load(os.path.join(data_dir, 'code_levels.npy'))

        trans_y_train = torch.tensor(train_codes_y)
        trans_y_test = torch.tensor(test_codes_y)

        training_data = MIMICivDataset(
            'binary_train_x_slices/binary_train_codes_x',
            train_visit_lens,
            trans_y_train,
            train_pids,
            'Train'
        )
        from dataset import MIMICivDataset as _MIMICivDataset  # just to silence lints
        test_data = MIMICivDataset(
            'binary_test_x_slices/binary_test_codes_x',
            test_visit_lens,
            trans_y_test,
            test_pids,
            'Test'
        )

    train_loader = DataLoader(
        training_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    # ---------- Build model ----------
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

    print(f'Number of parameters of this model: {sum(param.numel() for param in model.parameters())}')

    return model, test_loader, device, code_levels


def pearson_corr(a, b):
    """
    a, b: 1D numpy arrays
    Returns Pearson correlation; returns 0 if variance is ~0.
    """
    if len(a) < 2:
        return 0.0
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.0
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denom = (np.linalg.norm(a_centered) * np.linalg.norm(b_centered))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a_centered, b_centered) / denom)


def compute_explanation_metrics(model, test_loader, device):
    """
    Computes:
      - Faithfulness: Pearson corr between phenotype weights and change in prediction
        when we "remove" each phenotype (by zeroing its latent embedding).:contentReference[oaicite:1]{index=1}
      - Complexity: average number of diagnoses included in phenotypes.:contentReference[oaicite:2]{index=2}
      - Distinctness: 1 - average Jaccard overlap between phenotypes.:contentReference[oaicite:3]{index=3}
    """
    model.eval()
    faithfulness_scores = []
    complexity_scores = []
    distinctness_scores = []

    with torch.no_grad():
        # Hierarchical embeddings are shared across patients
        X = model.hier_embed_layer()  # shape: [num_codes, embed_dim]
        X = X.to(device)

        for batch in test_loader:
            patients, labels, pids, visit_lens = batch
            patients = patients.to(device)
            visit_lens = visit_lens.to(device)

            batch_size = patients.size(0)

            for b in range(batch_size):
                visit_len = int(visit_lens[b].item())
                if visit_len <= 0:
                    continue

                # H_i: [num_codes, visit_len]
                H_i = patients[b, :, :visit_len]

                # Encoder → phenotypes and latent phenotype embeddings
                TPs, latent_TPs, _ = model.encoder(X, H_i)

                # TPs: [T, num_codes, visit_len]  OR [num_codes, visit_len] if only 1 TP
                if TPs.dim() == 2:
                    # Single phenotype: add phenotype dim
                    TPs = TPs.unsqueeze(0)

                num_TP = TPs.size(0)

                # latent_TPs is typically [T, hidden_dim]; ensure it has a batch dim for fclf:
                if latent_TPs.dim() == 2:
                    # [T, H] -> [1, T, H]
                    latent_TPs = latent_TPs.unsqueeze(0)
                elif latent_TPs.dim() == 3 and latent_TPs.size(0) != 1:
                    # If for some reason it has shape [B, T, H] with B>1, just take the current patient
                    # (we are iterating per-patient anyway).
                    latent_TPs = latent_TPs[0:1]

                # Classifier → predictions + attention weights α
                # latent_TPs: [1, T, H]
                pred_full, alpha = model.fclf(latent_TPs)
                # pred_full: [1, num_codes], alpha: [1, T]
                pred_full = pred_full[0]   # -> [num_codes]
                alpha = alpha[0] 

                # ----- Complexity -----
                # binarize masks at 0.5 threshold
                TPs_bin = (TPs > 0.5).float()
                complexity_i = float(TPs_bin.sum().item())
                complexity_scores.append(complexity_i)

                # ----- Distinctness -----
                # For each visit and each phenotype pair, compute IoU and average.
                pair_ious = []
                for t in range(visit_len):
                    # slice per-visit masks: [T, num_codes]
                    per_visit = TPs_bin[:, :, t]  # (T, C)
                    for i in range(num_TP):
                        for j in range(i + 1, num_TP):
                            mask_i = per_visit[i]
                            mask_j = per_visit[j]
                            inter = torch.logical_and(mask_i > 0.5, mask_j > 0.5).sum().item()
                            union = torch.logical_or(mask_i > 0.5, mask_j > 0.5).sum().item()
                            if union > 0:
                                pair_ious.append(inter / union)
                if len(pair_ious) > 0:
                    avg_iou = float(np.mean(pair_ious))
                    distinctness_i = 1.0 - avg_iou
                    distinctness_scores.append(distinctness_i)

                # ----- Faithfulness -----
                alpha_np = alpha.detach().cpu().numpy()
                if alpha_np.ndim > 1:
                    alpha_np = alpha_np.reshape(-1)

                pred_probs = pred_full.detach().cpu().numpy()
                top_label = int(pred_probs.argmax())
                base_score = float(pred_probs[top_label])

                delta_scores = []
                for k in range(num_TP):
                    # Zero out phenotype k in the latent space: latent_TPs is [1, T, H]
                    latent_masked = latent_TPs.clone()
                    latent_masked[0, k, :] = 0.0

                    pred_masked, _ = model.fclf(latent_masked)
                    pred_masked_np = pred_masked[0].detach().cpu().numpy()  # strip batch dim
                    score_masked = float(pred_masked_np[top_label])
                    delta_scores.append(base_score - score_masked)

                delta_scores = np.array(delta_scores, dtype=float)
                faith = pearson_corr(alpha_np, delta_scores)
                faithfulness_scores.append(faith)

    # Aggregate
    faithfulness_mean = float(np.mean(faithfulness_scores)) if len(faithfulness_scores) > 0 else 0.0
    complexity_mean = float(np.mean(complexity_scores)) if len(complexity_scores) > 0 else 0.0
    distinctness_mean = float(np.mean(distinctness_scores)) if len(distinctness_scores) > 0 else 0.0

    return {
        "Faithfulness": faithfulness_mean,
        "Complexity": complexity_mean,
        "Distinctness": distinctness_mean,
        "num_patients": len(faithfulness_scores)
    }


def main():
    parser = argparse.ArgumentParser()

    # Same model hyperparameters as main.py
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
    parser.add_argument('--key_dim', type=int, default=256, help="key dim for self attention")
    parser.add_argument('--SA_head', type=int, default=8, help="number of heads for self-attention")
    parser.add_argument('--dropout', type=float, default=0.001, help="dropout ratio")
    parser.add_argument('--batch_size', type=int, default=128, help="batch size")
    parser.add_argument('--HGNN_model', type=str, default='UniGINConv', help="which hypergraph nn to use")

    # These three you already used for training/robustness:
    parser.add_argument('--temperature', type=float, nargs='+', required=True)
    parser.add_argument('--add_ratio', type=float, nargs='+', required=True)
    parser.add_argument('--loss_weight', type=float, nargs='+', required=True)

    # Path to trained checkpoint
    parser.add_argument('--ckpt_path', type=str, required=True, help="path to shy_epoch_*.pth")

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Build model + loader
    model, test_loader, device, code_levels = build_model_and_test_loader(args)

    # Load checkpoint
    print(f"Loading checkpoint from: {args.ckpt_path}")
    state_dict = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Compute metrics
    results = compute_explanation_metrics(model, test_loader, device)

    print("\n=== Explanation Metrics (Test Set) ===")
    print(f"Patients used: {results['num_patients']}")
    print(f"Faithfulness: {results['Faithfulness']:.4f}")
    print(f"Complexity:   {results['Complexity']:.4f}")
    print(f"Distinctness: {results['Distinctness']:.4f}")


if __name__ == '__main__':
    main()
