import os
import argparse
import torch
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
import datetime
from sklearn.metrics import average_precision_score
from data import D3_dataset_AP
from models import D3Scorer

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_eval_batch(samples):
    return samples


def to_int_label(label):
    if torch.is_tensor(label):
        return int(label.item())
    return int(label)


def score_grouped_samples(samples, scorer, device, temporal_mode):
    grouped = {}
    for offset, sample in enumerate(samples):
        if temporal_mode == 'time_norm':
            embeddings, timestamps, label = sample
        else:
            embeddings, label = sample
            timestamps = None
        grouped.setdefault(int(embeddings.shape[0]), []).append((offset, embeddings, timestamps, label))

    scored = []
    for group in grouped.values():
        offsets = [item[0] for item in group]
        labels = [to_int_label(item[3]) for item in group]
        batch_inputs = torch.stack([item[1] for item in group], dim=0).to(device)
        if temporal_mode == 'time_norm':
            batch_timestamps = torch.stack([item[2] for item in group], dim=0).to(device)
            _, _, batch_dis_std = scorer(batch_inputs, batch_timestamps)
        else:
            _, _, batch_dis_std = scorer(batch_inputs)

        batch_scores = batch_dis_std.cpu().flatten().numpy()
        for offset, label, score in zip(offsets, labels, batch_scores):
            scored.append((offset, label, float(score)))

    return sorted(scored, key=lambda item: item[0])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training script with configurable parameters.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--gpu-id', type=str, default="0",
                        help='CUDA GPU device ID(s), e.g., "0" or "1,2,3" (default: "5")')
    parser.add_argument('--loss', type=str, default='l2', choices=['l2', 'cos'],
                        help='Loss function type (default: l2)')
    parser.add_argument('--real-csv', type=str, default=None,
                        help='Path to the real data CSV file ')
    parser.add_argument('--fake-csv', type=str, default=None,
                        help='Path to the fake/synthetic data CSV file')
    parser.add_argument('--temporal-mode', type=str, default='legacy', choices=['legacy', 'time_norm'],
                        help='D3 temporal scoring mode (default: legacy)')
    parser.add_argument('--max-len', type=int, default=9999999,
                        help='Maximum samples to read from each CSV (default: all)')
    parser.add_argument('--batch-size', type=int, default=2048,
                        help='Evaluation batch size (default: 2048)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of DataLoader workers (default: 4)')
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError('--batch-size must be >= 1')
    if args.num_workers < 0:
        raise ValueError('--num-workers must be >= 0')

    seed = args.seed
    gpu_id = args.gpu_id
    loss_type = args.loss
    real_csv = args.real_csv
    fake_csv = args.fake_csv
    temporal_mode = args.temporal_mode
    max_len = args.max_len
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # real_csv = 'datasets/csv/t1.csv'
    # fake_csv = 'datasets/csv/t2.csv' 
    
    print(f"Starting AP evaluation with {loss_type} loss")
    print(f"Temporal Mode: {temporal_mode}")
    print(f"Real CSV: {real_csv}")
    print(f"Fake CSV: {fake_csv}")
    
    scorer = D3Scorer(loss_type=loss_type).to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        scorer = torch.nn.DataParallel(scorer)
    scorer.eval()
    
    # Load Dataset
    eval_dataset = D3_dataset_AP(real_csv=real_csv, fake_csv=fake_csv, max_len=max_len, temporal_mode=temporal_mode)
    print(f"Total samples: {len(eval_dataset)}")
    
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_eval_batch,
    )
    
    # Eval
    y_true, y_pred = [], []
    score_rows = []
    with torch.no_grad():
        for sample_idx, samples in enumerate(tqdm(eval_loader, desc="Evaluating")):
            scored_samples = score_grouped_samples(samples, scorer, device, temporal_mode)
            for offset, label, score in scored_samples:
                row_idx = sample_idx * eval_loader.batch_size + offset
                source_row = eval_dataset.df.iloc[row_idx]
                y_pred.append(score)
                y_true.append(label)
                score_rows.append({
                    "content_path": source_row["content_path"],
                    "type_id": source_row["type_id"],
                    "score": float(score),
                    "encoder_type": source_row.get("encoder_type", ""),
                    "source_fps": source_row.get("source_fps", ""),
                    "effective_fps": source_row.get("effective_fps", ""),
                    "temporal_mode": temporal_mode,
                })
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ap_score = average_precision_score(1-y_true, y_pred)
    
    os.makedirs("results", exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fake_name = os.path.splitext(os.path.basename(fake_csv))[0]
    output_file = f"results/result_{timestamp}.txt"
    score_file = f"results/result_{fake_name}_{timestamp}.csv"

    result_str = (
        f"AP Evaluation Results\n"
        f"Loss Type: {loss_type}\n"
        f"Temporal Mode: {temporal_mode}\n"
        f"Real CSV: {real_csv}\n"
        f"Fake CSV: {fake_csv}\n"
        f"Total Samples: {len(y_true)}\n"
        f"AP Score: {ap_score:.4f}\n"
        f"Score CSV: {score_file}\n"
    )
    
    print("\n" + "="*50)
    print(result_str.strip())
    print("="*50)

    with open(output_file, 'w') as f:
        f.write(result_str)

    pd.DataFrame(score_rows).sort_values("score", ascending=False).to_csv(
        score_file, encoding='utf-8', index=False
    )

    print(f"\nResults saved to {output_file}")
    print(f"Scores saved to {score_file}")
