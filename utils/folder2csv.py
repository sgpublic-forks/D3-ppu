import os
import json
import pandas as pd
from pandas import Series
from glob import glob
import argparse

def main(dataset_path):
    video_dir = os.path.join(dataset_path, 'video')
    frames_dir = os.path.join(dataset_path, 'frames')
    csv_dir = os.path.join(dataset_path, 'csv')
    
    # 确保csv目录存在
    os.makedirs(csv_dir, exist_ok=True)
    
    # 获取video目录下的所有文件夹
    folders = [f for f in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, f))]
    
    if not folders:
        print(f"No folders found in {video_dir}")
        return
    
    print(f"Found {len(folders)} folders: {folders}")
    
    def count_images_in_folder(folder_path):
        image_count = 0
        image_names = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.png') or file_name.endswith('.jpg') or file_name.endswith('.jpeg'):
                image_count += 1
                image_names.append(int(file_name.split('.')[0]))
        image_names.sort()
        return image_count, image_names

    def read_metadata(folder_path):
        metadata_path = os.path.join(folder_path, 'metadata.json')
        if not os.path.exists(metadata_path):
            return metadata_path, {}
        with open(metadata_path, 'r') as f:
            return metadata_path, json.load(f)

    for folder in folders:
        # 根据文件夹名判断是否为真实视频
        is_real = folder.startswith('real')
        
        folder_path = os.path.join(frames_dir, folder)
        csv_path = os.path.join(csv_dir, f'{folder}.csv')
        
        # 检查frames目录下是否存在该文件夹
        if not os.path.exists(folder_path):
            print(f"Skipping {folder}: frames directory not found")
            continue
        
        all_dirs = []
        for root, dirs, files in os.walk(folder_path):
            for dir in dirs:
                all_dirs.append(os.path.join(root, dir))

        label = []
        save_path = []
        frame_counts = []
        frame_seq_counts = []
        content_paths = []
        str_labels = []
        metadata_paths = []
        source_fps_values = []
        target_fps_values = []
        effective_fps_values = []
        temporal_modes = []

        for video_path in all_dirs:
            frame_paths = glob(os.path.join(video_path, '*'))
            temp_frame_count, temp_frame_seqs = count_images_in_folder(video_path)
            if temp_frame_count == 0:
                continue
            metadata_path, metadata = read_metadata(video_path)

            for frame in frame_paths:
                if not frame.endswith(('.png', '.jpg', '.jpeg')):
                    continue
                # 简化路径处理
                rel_path = os.path.relpath(frame, dataset_path)
                content_path = os.path.dirname(rel_path)
                content_path = os.path.join(dataset_path, content_path)
                frame_path = os.path.join(dataset_path, rel_path)
                rel_metadata_path = os.path.relpath(metadata_path, dataset_path)
                csv_metadata_path = os.path.join(dataset_path, rel_metadata_path)

                print(content_path, frame_path)
                if is_real:
                    label.append('0')
                    str_labels.append('Real Video')
                else:
                    label.append('1')
                    str_labels.append('AI Video')
                frame_counts.append(temp_frame_count)
                frame_seq_counts.append(temp_frame_seqs)
                save_path.append(frame_path)
                content_paths.append(content_path)
                metadata_paths.append(csv_metadata_path if metadata else '')
                source_fps_values.append(metadata.get('source_fps', ''))
                target_fps_values.append(metadata.get('target_fps', ''))
                effective_fps_values.append(metadata.get('effective_fps', ''))
                temporal_modes.append(metadata.get('temporal_mode', 'legacy'))
                break

        dic = {
            'content_path': Series(data=content_paths),
            'image_path': Series(data=save_path),
            'type_id': Series(data=str_labels),
            'label': Series(data=label),
            'frame_len': Series(data=frame_counts),
            'frame_seq': Series(data=frame_seq_counts),
            'metadata_path': Series(data=metadata_paths),
            'source_fps': Series(data=source_fps_values),
            'target_fps': Series(data=target_fps_values),
            'effective_fps': Series(data=effective_fps_values),
            'temporal_mode': Series(data=temporal_modes),
        }

        print(f"Processing {folder}: {len(label)} samples")
        pd.DataFrame(dic).to_csv(csv_path, encoding='utf-8', index=False)
        print(f"Saved CSV: {csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process all video folders and generate CSV files')
    parser.add_argument('--dataset-path', type=str, default='GenVideo',
                        help="Path to the dataset directory (default: GenVideo)")
    args = parser.parse_args()

    main(args.dataset_path)
