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
from models import D3_model

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training script with configurable parameters.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--gpu-id', type=str, default="0",
                        help='CUDA GPU device ID(s), e.g., "0" or "1,2,3" (default: "5")')
    parser.add_argument('--loss', type=str, default='l2', choices=['l2', 'cos'],
                        help='Loss function type (default: l2)')
    parser.add_argument('--encoder', type=str, default='XCLIP-16', 
                        help='Encoder model name (default: XCLIP-16)',
                        choices=['CLIP-16', 'CLIP-32', 'XCLIP-16', 'XCLIP-32', 'DINO-base', 'DINO-large', 'ResNet-18', 'VGG-16', 'EfficientNet-b4', 'MobileNet-v3'])
    parser.add_argument('--real-csv', type=str, default=None,
                        help='Path to the real data CSV file ')
    parser.add_argument('--fake-csv', type=str, default=None,
                        help='Path to the fake/synthetic data CSV file')
    parser.add_argument('--temporal-mode', type=str, default='legacy', choices=['legacy', 'time_norm'],
                        help='D3 temporal scoring mode (default: legacy)')
    args = parser.parse_args()

    seed = args.seed
    gpu_id = args.gpu_id
    loss_type = args.loss
    encoder_type = args.encoder
    real_csv = args.real_csv
    fake_csv = args.fake_csv
    temporal_mode = args.temporal_mode

    # real_csv = 'datasets/csv/t1.csv'
    # fake_csv = 'datasets/csv/t2.csv' 
    
    print(f"Starting AP evaluation for {encoder_type} with {loss_type} loss")
    print(f"Temporal Mode: {temporal_mode}")
    print(f"Real CSV: {real_csv}")
    print(f"Fake CSV: {fake_csv}")
    
    # Load Model
    model = D3_model(encoder_type=encoder_type, loss_type=loss_type).cuda()
    model.eval()
    
    # Load Dataset
    eval_dataset = D3_dataset_AP(real_csv=real_csv, fake_csv=fake_csv, max_len=1000, temporal_mode=temporal_mode)
    print(f"Total samples: {len(eval_dataset)}")
    
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=1, 
        pin_memory=True,
        drop_last=False
    )
    
    # Eval
    y_true, y_pred = [], []
    score_rows = []
    with torch.no_grad():
        for sample_idx, batch in enumerate(tqdm(eval_loader, desc="Evaluating")):
            if temporal_mode == 'time_norm':
                batch_frames, batch_timestamps, batch_label = batch
                batch_inputs = batch_frames.cuda()
                batch_timestamps = batch_timestamps.cuda()
                _, _, batch_dis_std = model(batch_inputs, batch_timestamps)
            else:
                batch_frames, batch_label = batch
                batch_inputs = batch_frames.cuda()
                _, _, batch_dis_std = model(batch_inputs)
            batch_scores = batch_dis_std.cpu().flatten().numpy()
            batch_labels = batch_label.cpu().flatten().numpy()
            y_pred.extend(batch_scores)
            y_true.extend(batch_labels)

            for offset, (label, score) in enumerate(zip(batch_labels, batch_scores)):
                row_idx = sample_idx * eval_loader.batch_size + offset
                source_row = eval_dataset.df.iloc[row_idx]
                score_rows.append({
                    "content_path": source_row["content_path"],
                    "type_id": source_row["type_id"],
                    "score": float(score),
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
        f"Encoder: {encoder_type}\n"
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
