import os
import json
from glob import glob
from moviepy.editor import VideoFileClip
import multiprocessing
import math
import random
import subprocess
import cv2


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


def extract_time_norm_frames(video_path, image_path, start_time, duration, target_fps):
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

    os.makedirs(image_path, exist_ok=True)
    metadata_frames = []
    for output_index, frame_info in enumerate(selected_frames, start=1):
        file_name = f"{output_index}.jpg"
        cv2.imwrite(os.path.join(image_path, file_name), frame_info["frame"])
        metadata_frames.append({
            "file": file_name,
            "timestamp": frame_info["timestamp"],
            "source_frame_index": frame_info["source_frame_index"],
        })

    metadata = {
        "source_video": video_path,
        "source_fps": source_fps,
        "target_fps": float(target_fps),
        "effective_fps": effective_fps,
        "temporal_mode": "time_norm",
        "start_time": float(start_time),
        "duration": float(duration),
        "frames": metadata_frames,
    }
    with open(os.path.join(image_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def process_video(video_path, dataset_path, temporal_mode, target_fps):
    video_name = video_path.split('/')[-1]
    video_name = video_name.split('.')[:-1]
    video_name = '.'.join(video_name)

    video_root = os.path.join(dataset_path, 'video')
    path = os.path.dirname(os.path.relpath(video_path, video_root))
    image_path = f'{dataset_path}/frames/'+path+'/'+ video_name+'/'

    metadata_path = os.path.join(image_path, "metadata.json")
    frames_exist = os.path.exists(image_path)
    metadata_exists = os.path.exists(metadata_path)
    if frames_exist and (temporal_mode == "legacy" or metadata_exists):
        print(video_name, "frames exist")
    else:
        print(video_name, end='\r')
        try:
            command = None
            try:
                if frames_exist and temporal_mode == "time_norm":
                    for file_name in os.listdir(image_path):
                        if file_name.endswith(('.png', '.jpg', '.jpeg')) or file_name == 'metadata.json':
                            os.remove(os.path.join(image_path, file_name))

                duration = 3
                video_length = get_video_length(video_path)
                if video_length <= 3:
                    start_time = 0
                else:
                    start_time = math.floor(random.uniform(0, video_length-3))

                if temporal_mode == "legacy":
                    os.makedirs(os.path.dirname(image_path), exist_ok=True)
                    command = [
                        "ffmpeg",
                        "-y",
                        "-loglevel", "error",
                        "-ss", str(start_time),
                        "-t", str(duration),
                        "-i", video_path,
                        "-vf", f"fps={target_fps}",
                        f"{image_path}%d.jpg",
                    ]
                    subprocess.run(command, check=True, capture_output=True, text=True)
                else:
                    extract_time_norm_frames(video_path, image_path, start_time, duration, target_fps)
            except Exception as e:
                with open('error.log', 'a') as f:
                    f.write(f"{video_name}\n  error: {e}\n  cmd: {command}\n")
                print(f"{video_name} error")
        except:
            with open('error.log', 'a') as f:
                f.write(f"{video_name} skipped\n")

import argparse

if __name__ == '__main__':

    random.seed(42)

    parser = argparse.ArgumentParser(description='Specify the dataset path.')
    parser.add_argument('--dataset-path', type=str, default='datasets', 
                        help='Path to the dataset directory (default: datasets)')
    parser.add_argument('--temporal-mode', type=str, default='legacy', choices=['legacy', 'time_norm'],
                        help='Frame extraction mode (default: legacy)')
    parser.add_argument('--target-fps', type=float, default=8,
                        help='Target sampling rate for extraction (default: 8)')
    args = parser.parse_args()
    dataset_path = args.dataset_path

    video_paths = glob(f"{dataset_path}/video/**", recursive=True)
    video_paths = [vp for vp in video_paths if vp.endswith(('.mp4', '.avi', '.mov', '.mkv', '.gif'))]

    print(f"Find {len(video_paths)} videos!")
    args_list = [(vp, dataset_path, args.temporal_mode, args.target_fps) for vp in video_paths]

    with multiprocessing.Pool(processes=4) as pool:
        pool.starmap(process_video, args_list)
