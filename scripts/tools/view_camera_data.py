#!/usr/bin/env python3
"""
Script to view recorded camera data from HDF5 dataset files.

Usage:
    # List all episodes
    python scripts/tools/view_camera_data.py --dataset_file ./datasets/bread_demos.hdf5 --mode list
    
    # Display camera feeds in real-time
    python scripts/tools/view_camera_data.py --dataset_file ./datasets/bread_demos.hdf5 --episode 0 --mode display
    
    # Export images and create videos
    python scripts/tools/view_camera_data.py --dataset_file ./datasets/bread_demos.hdf5 --episode 0 --mode export
"""

import argparse
import h5py
import numpy as np
import cv2
import os
from pathlib import Path


def display_camera_feeds(dataset_file: str, episode_index: int = 0, fps: int = 30):
    """Display wrist and front camera feeds side-by-side in real-time."""
    print(f"Opening dataset: {dataset_file}")
    
    with h5py.File(dataset_file, "r") as f:
        episodes = sorted(list(f["data"].keys()))
        print(f"Found {len(episodes)} episodes")
        
        if episode_index >= len(episodes):
            print(f"Error: Episode {episode_index} not found. Max index: {len(episodes)-1}")
            return
        
        episode_name = episodes[episode_index]
        print(f"Loading episode: {episode_name}")
        
        obs_group = f["data"][episode_name]["obs"]
        
        has_wrist = "wrist" in obs_group
        has_front = "front" in obs_group
        
        if not has_wrist and not has_front:
            print("Error: No camera data found")
            return
        
        wrist_images = obs_group["wrist"][:] if has_wrist else None
        front_images = obs_group["front"][:] if has_front else None
        
        num_frames = len(wrist_images) if has_wrist else len(front_images)
        print(f"Total frames: {num_frames} ({num_frames / fps:.2f}s)")
        print("Press 'q' to quit, 'p' to pause, SPACE to step")
        
        frame_idx = 0
        paused = False
        
        while frame_idx < num_frames:
            frames = []
            
            if has_wrist:
                wrist = wrist_images[frame_idx]
                if wrist.max() <= 1.0:
                    wrist = (wrist * 255).astype(np.uint8)
                frames.append(cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR))
            
            if has_front:
                front = front_images[frame_idx]
                if front.max() <= 1.0:
                    front = (front * 255).astype(np.uint8)
                frames.append(cv2.cvtColor(front, cv2.COLOR_RGB2BGR))
            
            combined = np.hstack(frames) if len(frames) == 2 else frames[0]
            cv2.putText(combined, f"Frame: {frame_idx}/{num_frames-1}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Camera Feeds", combined)
            
            key = cv2.waitKey(0 if paused else int(1000/fps)) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
            elif key == ord(' '):
                paused = True
                frame_idx += 1
            elif not paused:
                frame_idx += 1
        
        cv2.destroyAllWindows()


def export_camera_data(dataset_file: str, episode_index: int = 0, output_dir: str = "./camera_exports"):
    """Export camera images and create videos."""
    with h5py.File(dataset_file, "r") as f:
        episodes = sorted(list(f["data"].keys()))
        episode_name = episodes[episode_index]
        
        output_path = Path(output_dir) / episode_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        obs_group = f["data"][episode_name]["obs"]
        
        for cam_name in ["wrist", "front"]:
            if cam_name in obs_group:
                images = obs_group[cam_name][:]
                cam_dir = output_path / cam_name
                cam_dir.mkdir(exist_ok=True)
                
                print(f"Exporting {len(images)} {cam_name} frames...")
                for t, frame in enumerate(images):
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    cv2.imwrite(str(cam_dir / f"frame_{t:04d}.png"), 
                               cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                
                # Create video
                video_path = output_path / f"{cam_name}_camera.mp4"
                cmd = f"ffmpeg -y -framerate 30 -pattern_type glob -i '{cam_dir}/*.png' -c:v libx264 -pix_fmt yuv420p {video_path}"
                os.system(cmd)
                print(f"Created: {video_path}")


def list_episodes(dataset_file: str):
    """List all episodes with metadata."""
    with h5py.File(dataset_file, "r") as f:
        episodes = sorted(list(f["data"].keys()))
        print(f"Total episodes: {len(episodes)}\n")
        
        for idx, ep in enumerate(episodes):
            obs = f["data"][ep].get("obs", {})
            print(f"[{idx}] {ep}")
            print(f"    Wrist: {'✓' if 'wrist' in obs else '✗'}")
            print(f"    Front: {'✓' if 'front' in obs else '✗'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--mode", choices=["display", "export", "list"], default="display")
    parser.add_argument("--output_dir", default="./camera_exports")
    parser.add_argument("--fps", type=int, default=30)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_file):
        print(f"Error: File not found: {args.dataset_file}")
        return
    
    if args.mode == "list":
        list_episodes(args.dataset_file)
    elif args.mode == "display":
        display_camera_feeds(args.dataset_file, args.episode, args.fps)
    elif args.mode == "export":
        export_camera_data(args.dataset_file, args.episode, args.output_dir)


if __name__ == "__main__":
    main()
