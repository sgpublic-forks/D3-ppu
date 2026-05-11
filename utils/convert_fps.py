#!/usr/bin/env python3
import os
import subprocess
import argparse
from pathlib import Path

def convert_video_to_fps(input_path, output_path, fps=3):
    """Convert video to specified fps using ffmpeg"""
    cmd = [
        'ffmpeg',
        '-i', str(input_path),
        '-vf', f'fps={fps}',
        '-y',  # Overwrite output file
        str(output_path)
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed {input_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert all videos in a folder to specified fps')
    parser.add_argument('--folder', type=str, required=True,
                        help='Subfolder name in video directory (e.g., real_ysjf)')
    parser.add_argument('--dataset-path', type=str, default='GenVideo',
                        help='Dataset root directory (default: GenVideo)')
    parser.add_argument('--fps', type=int, default=3,
                        help='Target frame rate (default: 3)')
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    input_dir = dataset_path / 'video' / args.folder
    
    if not input_dir.exists():
        print(f"Error: Input directory does not exist {input_dir}")
        return
    
    output_dir = dataset_path / 'video' / f"{args.folder}_{args.fps}fps"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target frame rate: {args.fps} fps")
    
    # Supported video formats
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    
    video_files = [f for f in input_dir.iterdir() 
                   if f.is_file() and f.suffix.lower() in video_extensions]
    
    if not video_files:
        print("No video files found")
        return
    
    print(f"Found {len(video_files)} video files")
    
    success_count = 0
    for video_file in video_files:
        output_file = output_dir / f"{video_file.stem}_{args.fps}fps{video_file.suffix}"
        print(f"Converting: {video_file.name} -> {output_file.name}")
        
        if convert_video_to_fps(video_file, output_file, args.fps):
            success_count += 1
    
    print(f"\nConversion completed! Success: {success_count}/{len(video_files)}")
    print(f"Output directory: {output_dir}")

if __name__ == '__main__':
    main()