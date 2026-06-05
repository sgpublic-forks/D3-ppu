import argparse
import json
import os

import pandas as pd
from pandas import Series


def read_metadata(metadata_path):
    if not os.path.exists(metadata_path):
        return {}
    with open(metadata_path, 'r') as f:
        return json.load(f)


def main(dataset_path, encoder_type, temporal_mode):
    video_dir = os.path.join(dataset_path, 'video')
    embedding_root = os.path.join(dataset_path, 'embeddings', encoder_type, temporal_mode)
    csv_dir = os.path.join(dataset_path, 'csv', encoder_type, temporal_mode)

    os.makedirs(csv_dir, exist_ok=True)

    folders = [f for f in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, f))]
    if not folders:
        print(f"No folders found in {video_dir}")
        return

    print(f"Found {len(folders)} folders: {folders}")

    for folder in folders:
        is_real = folder.startswith('real')
        folder_path = os.path.join(embedding_root, folder)
        csv_path = os.path.join(csv_dir, f'{folder}.csv')

        if not os.path.exists(folder_path):
            print(f"Skipping {folder}: embedding directory not found")
            continue

        label = []
        content_paths = []
        embedding_paths = []
        metadata_paths = []
        str_labels = []
        frame_counts = []
        embedding_dims = []
        source_fps_values = []
        target_fps_values = []
        effective_fps_values = []
        encoder_values = []
        temporal_modes = []

        for root, dirs, files in os.walk(folder_path):
            if 'embedding.npz' not in files:
                continue

            embedding_path = os.path.join(root, 'embedding.npz')
            metadata_path = os.path.join(root, 'metadata.json')
            metadata = read_metadata(metadata_path)

            if is_real:
                label.append('0')
                str_labels.append('Real Video')
            else:
                label.append('1')
                str_labels.append('AI Video')

            content_paths.append(root)
            embedding_paths.append(embedding_path)
            metadata_paths.append(metadata_path if metadata else '')
            frame_counts.append(metadata.get('frame_count', ''))
            embedding_dims.append(metadata.get('embedding_dim', ''))
            source_fps_values.append(metadata.get('source_fps', ''))
            target_fps_values.append(metadata.get('target_fps', ''))
            effective_fps_values.append(metadata.get('effective_fps', ''))
            encoder_values.append(metadata.get('encoder_type', encoder_type))
            temporal_modes.append(metadata.get('temporal_mode', temporal_mode))

        dic = {
            'content_path': Series(data=content_paths),
            'embedding_path': Series(data=embedding_paths),
            'metadata_path': Series(data=metadata_paths),
            'type_id': Series(data=str_labels),
            'label': Series(data=label),
            'frame_len': Series(data=frame_counts),
            'embedding_dim': Series(data=embedding_dims),
            'encoder_type': Series(data=encoder_values),
            'source_fps': Series(data=source_fps_values),
            'target_fps': Series(data=target_fps_values),
            'effective_fps': Series(data=effective_fps_values),
            'temporal_mode': Series(data=temporal_modes),
        }

        print(f"Processing {folder}: {len(label)} samples")
        pd.DataFrame(dic).to_csv(csv_path, encoding='utf-8', index=False)
        print(f"Saved CSV: {csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate CSV files for video embeddings')
    parser.add_argument('--dataset-path', type=str, default='GenVideo')
    parser.add_argument('--encoder', type=str, default='XCLIP-16')
    parser.add_argument('--temporal-mode', type=str, default='legacy', choices=['legacy', 'time_norm'])
    args = parser.parse_args()

    main(args.dataset_path, args.encoder, args.temporal_mode)
