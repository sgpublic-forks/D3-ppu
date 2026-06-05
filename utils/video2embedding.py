import argparse
import json
import math
import os
import random

import cv2
import numpy as np
import torch
from moviepy.editor import VideoFileClip
from tqdm import tqdm

from data.datasets import crop_center_by_percentage, set_preprocessing
from models import D3_model
from utils.logging_utils import create_logger


VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.gif')


def iter_video_paths(dataset_path):
    video_root = os.path.join(dataset_path, 'video')
    for root, dirs, files in os.walk(video_root, followlinks=True):
        for file_name in files:
            if file_name.lower().endswith(VIDEO_EXTS):
                yield os.path.join(root, file_name)


def count_videos(dataset_path):
    return sum(1 for _ in iter_video_paths(dataset_path))


def get_video_length(file_path):
    video = VideoFileClip(file_path)
    try:
        return video.duration
    finally:
        video.close()


def get_source_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps and fps > 0:
            return float(fps)
    finally:
        cap.release()
    return None


def select_frames_by_time(frames, start_time, duration, source_fps, target_fps):
    if source_fps and source_fps > 0:
        effective_fps = min(float(source_fps), float(target_fps))
    else:
        effective_fps = float(target_fps)

    if not frames:
        return [], effective_fps

    if source_fps and source_fps <= target_fps:
        return frames, effective_fps

    interval = 1.0 / effective_fps
    sample_times = []
    current = start_time
    end_time = start_time + duration
    while current < end_time:
        sample_times.append(current)
        current += interval

    selected = []
    last_index = -1
    for sample_time in sample_times:
        best_index = min(
            range(len(frames)),
            key=lambda idx: abs(frames[idx]["timestamp"] - sample_time),
        )
        if best_index == last_index:
            continue
        selected.append(frames[best_index])
        last_index = best_index
    return selected, effective_fps


def read_sampled_frames(video_path, start_time, duration, target_fps):
    source_fps = get_source_fps(video_path)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

    frames = []
    source_frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if timestamp <= 0 and source_fps and source_fps > 0:
            timestamp = source_frame_index / source_fps
        if timestamp >= start_time + duration:
            break

        frames.append({
            "frame": frame,
            "timestamp": float(timestamp),
            "source_frame_index": int(source_frame_index),
        })
        source_frame_index += 1
    cap.release()

    selected_frames, effective_fps = select_frames_by_time(
        frames,
        start_time=start_time,
        duration=duration,
        source_fps=source_fps,
        target_fps=target_fps,
    )
    return selected_frames, source_fps, effective_fps


def preprocess_frames(selected_frames, trans, max_frames):
    tensors = []
    if len(selected_frames) < 8:
        raise ValueError("No enough frames sampled.")
    target_frame_count = 8 if len(selected_frames) < 16 else min(max_frames, len(selected_frames))
    kept_frames = selected_frames[:target_frame_count]
    for frame_info in kept_frames:
        image = crop_center_by_percentage(frame_info["frame"], 0.1)
        augmented = trans(image=image)
        image = augmented["image"]
        tensors.append(image.transpose(2, 0, 1)[np.newaxis, :])

    if not tensors:
        raise ValueError("No frames sampled.")

    frames = np.concatenate(tensors, 0)
    return torch.tensor(frames[np.newaxis, :]), kept_frames


def get_embedding_path(video_path, dataset_path, encoder_type, temporal_mode):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_root = os.path.join(dataset_path, 'video')
    rel_dir = os.path.dirname(os.path.relpath(video_path, video_root))
    return os.path.join(dataset_path, 'embeddings', encoder_type, temporal_mode, rel_dir, video_name)


def process_video(video_path, args, model, trans, device):
    output_dir = get_embedding_path(video_path, args.dataset_path, args.encoder, args.temporal_mode)
    embedding_path = os.path.join(output_dir, 'embedding.npz')
    metadata_path = os.path.join(output_dir, 'metadata.json')
    if os.path.exists(embedding_path) and os.path.exists(metadata_path) and not args.overwrite:
        return 'exists'

    duration = args.duration
    video_length = get_video_length(video_path)
    if video_length <= duration:
        start_time = 0
    else:
        start_time = math.floor(random.uniform(0, video_length - duration))

    selected_frames, source_fps, effective_fps = read_sampled_frames(
        video_path,
        start_time=start_time,
        duration=duration,
        target_fps=args.target_fps,
    )
    frame_tensor, kept_frames = preprocess_frames(selected_frames, trans, args.max_frames)
    frame_tensor = frame_tensor.to(device)

    with torch.no_grad():
        embeddings = model.encode_frames(frame_tensor).squeeze(0).cpu().numpy().astype(np.float32)

    timestamps = np.array([frame_info['timestamp'] for frame_info in kept_frames], dtype=np.float32)
    source_frame_indices = [frame_info['source_frame_index'] for frame_info in kept_frames]

    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(embedding_path, embeddings=embeddings, timestamps=timestamps)

    metadata = {
        'source_video': video_path,
        'encoder_type': args.encoder,
        'source_fps': source_fps,
        'target_fps': float(args.target_fps),
        'effective_fps': effective_fps,
        'temporal_mode': args.temporal_mode,
        'start_time': float(start_time),
        'duration': float(duration),
        'frame_count': int(embeddings.shape[0]),
        'embedding_dim': int(embeddings.shape[1]),
        'embedding_dtype': 'float32',
        'source_frame_indices': source_frame_indices,
        'timestamps': timestamps.tolist(),
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    return 'saved'


def main():
    parser = argparse.ArgumentParser(description='Convert videos to encoder embeddings.')
    parser.add_argument('--dataset-path', type=str, default='GenVideo')
    parser.add_argument('--encoder', type=str, default='XCLIP-16',
                        choices=['CLIP-16', 'CLIP-32', 'XCLIP-16', 'XCLIP-32', 'DINO-base', 'DINO-large',
                                 'ResNet-18', 'VGG-16', 'EfficientNet-b4', 'MobileNet-v3'])
    parser.add_argument('--temporal-mode', type=str, default='legacy', choices=['legacy', 'time_norm'])
    parser.add_argument('--target-fps', type=float, default=8)
    parser.add_argument('--duration', type=float, default=3)
    parser.add_argument('--max-frames', type=int, default=16)
    parser.add_argument('--gpu-id', type=str, default='0')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    random.seed(42)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger = create_logger('video2embedding', file_prefix='video2embedding')
    logger.info('Log file: %s', logger.log_path)
    logger.info('Counting videos in %s/video', args.dataset_path)
    video_count = count_videos(args.dataset_path)
    logger.info('Find %s videos', video_count)

    logger.info('Loading encoder: %s on %s', args.encoder, device)
    model = D3_model(encoder_type=args.encoder, loss_type='l2').to(device)
    model.eval()
    trans = set_preprocessing(None, None)
    logger.info('Encoder loaded')

    counts = {'saved': 0, 'exists': 0, 'errors': 0}
    progress = tqdm(
        iter_video_paths(args.dataset_path),
        total=video_count,
        desc='Embedding videos',
        unit='video',
    )
    for video_path in progress:
        try:
            status = process_video(video_path, args, model, trans, device)
            counts[status] += 1
        except Exception as e:
            counts['errors'] += 1
            logger.exception('%s error: %s', video_path, e)
        progress.set_postfix(counts)

    logger.info('Finished embedding videos: saved=%s exists=%s errors=%s', counts['saved'], counts['exists'], counts['errors'])


if __name__ == '__main__':
    main()
