import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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


def prepare_video(video_path, args, trans):
    output_dir = get_embedding_path(video_path, args.dataset_path, args.encoder, args.temporal_mode)
    embedding_path = os.path.join(output_dir, 'embedding.npz')
    metadata_path = os.path.join(output_dir, 'metadata.json')
    if os.path.exists(embedding_path) and os.path.exists(metadata_path) and not args.overwrite:
        return None

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
    timestamps = np.array([frame_info['timestamp'] for frame_info in kept_frames], dtype=np.float32)
    source_frame_indices = [frame_info['source_frame_index'] for frame_info in kept_frames]

    return {
        'video_path': video_path,
        'output_dir': output_dir,
        'embedding_path': embedding_path,
        'metadata_path': metadata_path,
        'frame_tensor': frame_tensor.squeeze(0),
        'timestamps': timestamps,
        'metadata': {
            'source_video': video_path,
            'encoder_type': args.encoder,
            'source_fps': source_fps,
            'target_fps': float(args.target_fps),
            'effective_fps': effective_fps,
            'temporal_mode': args.temporal_mode,
            'start_time': float(start_time),
            'duration': float(duration),
            'frame_count': len(kept_frames),
            'embedding_dim': None,
            'embedding_dtype': 'float32',
            'source_frame_indices': source_frame_indices,
            'timestamps': timestamps.tolist(),
        },
    }


def save_embedding(prepared, embeddings, compress=False):
    embeddings = embeddings.astype(np.float32)
    metadata = prepared['metadata']
    metadata['frame_count'] = int(embeddings.shape[0])
    metadata['embedding_dim'] = int(embeddings.shape[1])

    os.makedirs(prepared['output_dir'], exist_ok=True)
    save_fn = np.savez_compressed if compress else np.savez
    save_fn(prepared['embedding_path'], embeddings=embeddings, timestamps=prepared['timestamps'])
    with open(prepared['metadata_path'], 'w') as f:
        json.dump(metadata, f, indent=2)


def save_embedding_batch(batch, batch_embeddings, compress=False):
    saved = 0
    errors = []
    for prepared, embeddings in zip(batch, batch_embeddings):
        try:
            save_embedding(prepared, embeddings, compress=compress)
            saved += 1
        except Exception as e:
            errors.append((prepared['video_path'], e))
    return saved, errors


class SaveManager:
    def __init__(self, save_workers, save_pending_batches, compress, counts, logger):
        self.compress = compress
        self.counts = counts
        self.logger = logger
        self.executor = None
        self.pending = {}
        self.max_pending_batches = 0
        if save_workers > 0:
            self.executor = ThreadPoolExecutor(max_workers=save_workers)
            self.max_pending_batches = max(1, save_pending_batches)

    def submit_batch(self, batch, batch_embeddings):
        if self.executor is None:
            saved, errors = save_embedding_batch(batch, batch_embeddings, compress=self.compress)
            self.counts['saved'] += saved
            self.counts['errors'] += len(errors)
            for video_path, error in errors:
                self.logger.exception('%s save error: %s', video_path, error)
            return

        while len(self.pending) >= self.max_pending_batches:
            self.drain(block=True)

        future = self.executor.submit(save_embedding_batch, batch, batch_embeddings, self.compress)
        self.pending[future] = len(batch)
        self.drain(block=False)

    def drain(self, block=False):
        if not self.pending:
            return
        timeout = None if block else 0
        done, _ = wait(self.pending, timeout=timeout, return_when=FIRST_COMPLETED)
        for future in done:
            batch_size = self.pending.pop(future)
            try:
                saved, errors = future.result()
                self.counts['saved'] += saved
                self.counts['errors'] += len(errors)
                for video_path, error in errors:
                    self.logger.exception('%s save error: %s', video_path, error)
            except Exception as e:
                self.counts['errors'] += batch_size
                self.logger.exception('Save batch error (%s videos): %s', batch_size, e)

    def close(self):
        while self.pending:
            self.drain(block=True)
        if self.executor is not None:
            self.executor.shutdown(wait=True)

    @property
    def pending_count(self):
        return len(self.pending)


def process_video_batch(batch, model, device):
    if not batch:
        return []

    frame_tensor = torch.stack([item['frame_tensor'] for item in batch], dim=0).to(device)
    with torch.no_grad():
        batch_embeddings = model.encode_frames(frame_tensor).cpu().numpy().astype(np.float32)

    return batch_embeddings


def flush_bucket(buckets, frame_count, model, device, save_manager):
    batch = buckets.get(frame_count, [])
    if not batch:
        return 0
    batch_embeddings = process_video_batch(batch, model, device)
    save_manager.submit_batch(batch, batch_embeddings)
    buckets[frame_count] = []
    return len(batch)


def flush_all_buckets(buckets, model, device, save_manager):
    saved = 0
    for frame_count in list(buckets.keys()):
        saved += flush_bucket(buckets, frame_count, model, device, save_manager)
    return saved


def process_video(video_path, args, model, trans, device):
    prepared = prepare_video(video_path, args, trans)
    if prepared is None:
        return 'exists'
    batch_embeddings = process_video_batch([prepared], model, device)
    save_embedding_batch([prepared], batch_embeddings, compress=args.compress)
    return 'saved'


def update_progress_postfix(progress, counts, save_manager):
    postfix = counts.copy()
    postfix['pending_save_batches'] = save_manager.pending_count
    progress.set_postfix(postfix)


def add_prepared_to_bucket(prepared, buckets, counts, args, model, device, logger, save_manager):
    if prepared is None:
        counts['exists'] += 1
        return

    frame_count = int(prepared['frame_tensor'].shape[0])
    bucket = buckets.setdefault(frame_count, [])
    bucket.append(prepared)
    if len(bucket) < args.batch_size:
        return

    try:
        counts['encoded'] += flush_bucket(buckets, frame_count, model, device, save_manager)
    except Exception as e:
        failed = len(buckets.get(frame_count, []))
        counts['errors'] += failed
        logger.exception('Failed to encode batch with %s frames (%s videos): %s', frame_count, failed, e)
        buckets[frame_count] = []


def process_videos_serial(video_paths, args, model, trans, device, logger, counts, buckets, progress, save_manager):
    for video_path in video_paths:
        try:
            prepared = prepare_video(video_path, args, trans)
            progress.update(1)
            update_progress_postfix(progress, counts, save_manager)
            add_prepared_to_bucket(prepared, buckets, counts, args, model, device, logger, save_manager)
        except Exception as e:
            counts['errors'] += 1
            logger.exception('%s error: %s', video_path, e)
            progress.update(1)
        save_manager.drain(block=False)
        update_progress_postfix(progress, counts, save_manager)


def process_videos_threaded(video_paths, args, model, trans, device, logger, counts, buckets, progress, save_manager):
    max_inflight = max(1, args.num_workers * args.prefetch_factor)
    video_iter = iter(video_paths)
    pending = {}
    exhausted = False

    def submit_until_full(executor):
        nonlocal exhausted
        while not exhausted and len(pending) < max_inflight:
            try:
                video_path = next(video_iter)
            except StopIteration:
                exhausted = True
                break
            pending[executor.submit(prepare_video, video_path, args, trans)] = video_path

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        submit_until_full(executor)
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                video_path = pending.pop(future)
                try:
                    prepared = future.result()
                    progress.update(1)
                    update_progress_postfix(progress, counts, save_manager)
                    add_prepared_to_bucket(prepared, buckets, counts, args, model, device, logger, save_manager)
                except Exception as e:
                    counts['errors'] += 1
                    logger.exception('%s error: %s', video_path, e)
                    progress.update(1)
                save_manager.drain(block=False)
                update_progress_postfix(progress, counts, save_manager)
            submit_until_full(executor)


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
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--prefetch-factor', type=int, default=2)
    parser.add_argument('--save-workers', type=int, default=4)
    parser.add_argument('--save-pending-batches', type=int, default=16)
    parser.add_argument('--save-prefetch-factor', type=int, default=None,
                        help='Deprecated alias for --save-pending-batches')
    parser.add_argument('--compress', action='store_true')
    parser.add_argument('--gpu-id', type=str, default='0')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError('--batch-size must be >= 1')
    if args.num_workers < 0:
        raise ValueError('--num-workers must be >= 0')
    if args.prefetch_factor < 1:
        raise ValueError('--prefetch-factor must be >= 1')
    if args.save_workers < 0:
        raise ValueError('--save-workers must be >= 0')
    if args.save_prefetch_factor is not None:
        args.save_pending_batches = args.save_prefetch_factor
    if args.save_pending_batches < 1:
        raise ValueError('--save-pending-batches must be >= 1')

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
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.info('Using %s GPUs with DataParallel', torch.cuda.device_count())
        model.encoder = torch.nn.DataParallel(model.encoder)
    model.eval()
    trans = set_preprocessing(None, None)
    logger.info('Encoder loaded')
    logger.info(
        'Embedding pipeline: batch_size=%s num_workers=%s prefetch_factor=%s save_workers=%s save_pending_batches=%s compress=%s',
        args.batch_size,
        args.num_workers,
        args.prefetch_factor,
        args.save_workers,
        args.save_pending_batches,
        args.compress,
    )

    counts = {'encoded': 0, 'saved': 0, 'exists': 0, 'errors': 0}
    buckets = {}
    save_manager = SaveManager(
        save_workers=args.save_workers,
        save_pending_batches=args.save_pending_batches,
        compress=args.compress,
        counts=counts,
        logger=logger,
    )
    progress = tqdm(
        total=video_count,
        desc='Embedding videos',
        unit='video',
    )
    try:
        video_paths = iter_video_paths(args.dataset_path)
        if args.num_workers == 0:
            process_videos_serial(video_paths, args, model, trans, device, logger, counts, buckets, progress, save_manager)
        else:
            process_videos_threaded(video_paths, args, model, trans, device, logger, counts, buckets, progress, save_manager)

        for frame_count in list(buckets.keys()):
            try:
                counts['encoded'] += flush_bucket(buckets, frame_count, model, device, save_manager)
            except Exception as e:
                failed = len(buckets.get(frame_count, []))
                counts['errors'] += failed
                logger.exception('Failed to encode remaining batch with %s frames (%s videos): %s', frame_count, failed, e)
                buckets[frame_count] = []
            update_progress_postfix(progress, counts, save_manager)
    finally:
        save_manager.close()
        update_progress_postfix(progress, counts, save_manager)
        progress.close()

    logger.info('Finished embedding videos: saved=%s exists=%s errors=%s', counts['saved'], counts['exists'], counts['errors'])


if __name__ == '__main__':
    main()
