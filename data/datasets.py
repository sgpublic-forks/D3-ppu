
import os
import re
import json
import pandas as pd
import albumentations
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


def crop_center_by_percentage(image, percentage):
    height, width = image.shape[:2]
    if width > height:
        left_pixels = int(width * percentage)
        right_pixels = int(width * percentage)
        start_x = left_pixels
        end_x = width - right_pixels
        cropped_image = image[:, start_x:end_x]
    else:
        up_pixels = int(height * percentage)
        down_pixels = int(height * percentage)
        start_y = up_pixels
        end_y = height - down_pixels
        cropped_image = image[start_y:end_y, :]
    return cropped_image


def get_number_from_filename(filename):
    match = re.match(r'(\d+)', filename)  
    if match:
        return int(match.group(1))  
    return float('inf')  


def read_video(folder_path, trans):
    frames = []
    image_paths = [
        file_name for file_name in os.listdir(folder_path)
        if file_name.endswith(('.png', '.jpg', '.jpeg'))
    ]
    image_paths = sorted(image_paths, key=get_number_from_filename)
    total_frames = len(image_paths)

    if total_frames < 8:
        raise ValueError(f"No enough frames found in {folder_path}.")

    set_frame = 8 if total_frames < 16 else 16
    max_frame = min(set_frame, total_frames)
    for i in range(max_frame):
        image_path = os.path.join(folder_path, image_paths[i])
        image = cv2.imread(image_path)
        image = crop_center_by_percentage(image, 0.1)
        augmented = trans(image=image)
        image = augmented["image"]
        frames.append(image.transpose(2, 0, 1)[np.newaxis, :])

    frames = np.concatenate(frames, 0)
    frames = torch.tensor(frames[np.newaxis, :]).squeeze(0)
    return frames


def read_time_norm_video(folder_path, metadata_path, trans):
    frames = []
    timestamps = []
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    frame_infos = metadata.get('frames', [])
    total_frames = len(frame_infos)

    if total_frames < 8:
        raise ValueError(f"No enough frames found in {folder_path}.")

    set_frame = 8 if total_frames < 16 else 16
    max_frame = min(set_frame, total_frames)
    for frame_info in frame_infos[:max_frame]:
        image_path = os.path.join(folder_path, frame_info['file'])
        image = cv2.imread(image_path)
        image = crop_center_by_percentage(image, 0.1)
        augmented = trans(image=image)
        image = augmented["image"]
        frames.append(image.transpose(2, 0, 1)[np.newaxis, :])
        timestamps.append(float(frame_info['timestamp']))

    frames = np.concatenate(frames, 0)
    frames = torch.tensor(frames[np.newaxis, :]).squeeze(0)
    timestamps = torch.tensor(timestamps, dtype=torch.float32)
    return frames, timestamps


def set_preprocessing(aug_type, aug_quality):
    aug_list = []
    aug_list.append(albumentations.Resize(224, 224))
    if aug_type == 'Gaussian_blur':
        aug_list.append(albumentations.GaussianBlur(blur_limit=(3, 7),sigma_limit=(aug_quality, aug_quality),p=1.0)) 
    if aug_type == 'JEPG_compression':
        aug_list.append(albumentations.ImageCompression(quality_lower=aug_quality, quality_upper=aug_quality))
    aug_list.append(albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0))
    return albumentations.Compose(aug_list)


class D3_dataset_AP(Dataset):

    def __init__(self, real_csv, fake_csv, max_len = 9999999, aug_type = None, aug_quality = None, temporal_mode = 'legacy'):
        super(D3_dataset_AP, self).__init__()
        df_real = pd.read_csv(real_csv).head(max_len)
        df_fake = pd.read_csv(fake_csv).head(max_len)
        self.df = pd.concat([df_real, df_fake], axis=0, ignore_index=True)
        self.trans = set_preprocessing(aug_type, aug_quality)
        self.temporal_mode = temporal_mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        label = self.df.loc[index]['label']
        frame_path = self.df.loc[index]['content_path']
        if self.temporal_mode == 'time_norm':
            metadata_path = self.df.loc[index].get('metadata_path', '')
            if not isinstance(metadata_path, str) or metadata_path == '':
                raise ValueError(f"metadata_path is required for time_norm mode: {frame_path}")
            frames, timestamps = read_time_norm_video(frame_path, metadata_path, trans=self.trans)
            return frames, timestamps, label

        frames = read_video(frame_path,trans=self.trans)
        return frames, label
