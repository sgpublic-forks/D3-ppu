
import pandas as pd
import albumentations
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


def read_embedding(embedding_path):
    data = np.load(embedding_path)
    embeddings = torch.tensor(data['embeddings'], dtype=torch.float32)
    timestamps = torch.tensor(data['timestamps'], dtype=torch.float32)

    if embeddings.shape[0] < 8:
        raise ValueError(f"No enough embeddings found in {embedding_path}.")
    return embeddings, timestamps


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

    def __init__(self, real_csv, fake_csv, max_len = 9999999, temporal_mode = 'legacy'):
        super(D3_dataset_AP, self).__init__()
        df_real = pd.read_csv(real_csv).head(max_len)
        df_fake = pd.read_csv(fake_csv).head(max_len)
        self.df = pd.concat([df_real, df_fake], axis=0, ignore_index=True)
        self.temporal_mode = temporal_mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        label = self.df.loc[index]['label']
        embedding_path = self.df.loc[index]['embedding_path']
        embeddings, timestamps = read_embedding(embedding_path)
        if self.temporal_mode == 'time_norm':
            return embeddings, timestamps, label

        return embeddings, label
