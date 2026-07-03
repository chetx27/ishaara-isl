"""
gloss_to_video_stitcher.py

Phase 6: Video Generation via Lookup-and-Stitch

LIMITATION STATEMENT: 
This script implements video generation through reference clip retrieval and 
concatenation with crossfade transitions. It does NOT perform true generative 
sign synthesis (e.g., pose sequence generation models like Generative Adversarial 
Networks or Diffusion models). True generative synthesis is a distinct research 
problem requiring significantly more data and specialized architectures (e.g., 
Pose-to-Video models).

Author: Ishaara System
"""

import cv2
import os
import numpy as np

def generate_video_from_gloss(gloss_list, ref_dir, output_path, crossfade_frames=5):
    """
    Given a list of gloss tokens, look up their reference videos in ref_dir,
    and stitch them together into a single output video with a short crossfade.
    """
    clips = []
    
    for gloss in gloss_list:
        clip_path = os.path.join(ref_dir, f"{gloss.upper()}.mp4")
        if not os.path.exists(clip_path):
            print(f"Warning: Reference clip for {gloss} not found. Skipping.")
            continue
            
        cap = cv2.VideoCapture(clip_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        if frames:
            clips.append(frames)
            
    if not clips:
        print("No valid clips found to stitch.")
        return False
        
    height, width, _ = clips[0][0].shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
    
    stitched_frames = []
    
    for i in range(len(clips)):
        current_clip = clips[i]
        
        if i == 0:
            # First clip, just add all frames except the last `crossfade_frames`
            if len(clips) > 1 and len(current_clip) > crossfade_frames:
                stitched_frames.extend(current_clip[:-crossfade_frames])
            else:
                stitched_frames.extend(current_clip)
        else:
            prev_clip = clips[i-1]
            
            # Perform crossfade if both clips have enough frames
            if len(prev_clip) > crossfade_frames and len(current_clip) > crossfade_frames:
                fade_out = prev_clip[-crossfade_frames:]
                fade_in = current_clip[:crossfade_frames]
                
                for j in range(crossfade_frames):
                    alpha = (j + 1) / (crossfade_frames + 1)
                    beta = 1.0 - alpha
                    blended = cv2.addWeighted(fade_in[j], alpha, fade_out[j], beta, 0)
                    stitched_frames.append(blended)
                    
                # Add the rest of the current clip
                if i < len(clips) - 1:
                    stitched_frames.extend(current_clip[crossfade_frames:-crossfade_frames])
                else:
                    stitched_frames.extend(current_clip[crossfade_frames:])
            else:
                # Fallback to direct concat if clips are too short
                stitched_frames.extend(current_clip)
                
    for frame in stitched_frames:
        out.write(frame)
        
    out.release()
    print(f"Stitched video saved to {output_path}")
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gloss_seq", type=str, required=True, help="Comma separated glosses")
    parser.add_argument("--out", type=str, default="output.mp4")
    args = parser.parse_args()
    
    glosses = [g.strip() for g in args.gloss_seq.split(",")]
    
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ref_dir = os.path.join(base_path, "data", "reference_videos")
    
    generate_video_from_gloss(glosses, ref_dir, args.out)
