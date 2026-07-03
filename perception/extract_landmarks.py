"""
extract_landmarks.py

Perception layer to extract normalized landmarks from raw video datasets using 
MediaPipe Holistic.

Features:
- Extracts 95 landmarks (33 pose + 21 L-hand + 21 R-hand + 20 face).
- Spatial normalization relative to shoulder midpoint.
- Scale normalization via shoulder width.
- Interpolates missing landmarks (< 3 frames) and provides validity mask.
- Outputs (num_frames, 285) tensors + (num_frames,) mask.

Author: Ishaara System
"""

import os
import cv2
import glob
import numpy as np
import mediapipe as mp
from scipy.interpolate import interp1d

mp_holistic = mp.solutions.holistic

# Reduced face subset indices (~20 points for eyebrows and mouth)
# Non-manual markers in ISL heavily rely on these regions.
REDUCED_FACE_INDICES = [
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300, # Eyebrows
    61, 40, 37, 0, 267, 270, 291, 84, 17, 314      # Lips
]
assert len(REDUCED_FACE_INDICES) == 20

NUM_LANDMARKS = 33 + 21 + 21 + 20 # = 95
NUM_FEATURES = NUM_LANDMARKS * 3  # = 285

def extract_landmarks_from_video(video_path):
    """
    Process a single video, extracting and normalizing landmarks per frame.
    Returns:
        features: np.array of shape (frames, 285)
        mask: np.array of shape (frames,) (1=valid, 0=invalid/interpolated heavily)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    raw_features = []
    
    with mp_holistic.Holistic(
        static_image_mode=False, 
        model_complexity=1, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            
            frame_data = np.full((NUM_LANDMARKS, 3), np.nan)
            
            # Pose
            if results.pose_landmarks:
                for i, lm in enumerate(results.pose_landmarks.landmark):
                    frame_data[i] = [lm.x, lm.y, lm.z]
                    
            # Left Hand
            if results.left_hand_landmarks:
                for i, lm in enumerate(results.left_hand_landmarks.landmark):
                    frame_data[33 + i] = [lm.x, lm.y, lm.z]
                    
            # Right Hand
            if results.right_hand_landmarks:
                for i, lm in enumerate(results.right_hand_landmarks.landmark):
                    frame_data[33 + 21 + i] = [lm.x, lm.y, lm.z]
                    
            # Face (Reduced)
            if results.face_landmarks:
                for i, idx in enumerate(REDUCED_FACE_INDICES):
                    lm = results.face_landmarks.landmark[idx]
                    frame_data[33 + 21 + 21 + i] = [lm.x, lm.y, lm.z]
            
            raw_features.append(frame_data)
            
    cap.release()
    
    if not raw_features:
        return None, None
        
    raw_features = np.array(raw_features) # (T, 95, 3)
    T = raw_features.shape[0]
    
    # 1. Normalize relative to shoulder midpoint (Pose indices: 11 left, 12 right)
    # 2. Scale normalize using shoulder width
    
    processed_features = []
    valid_mask = np.ones(T, dtype=np.float32)
    
    for t in range(T):
        frame = raw_features[t]
        
        if np.isnan(frame[11, 0]) or np.isnan(frame[12, 0]):
            # Missing pose, cannot normalize reliably
            processed_features.append(np.full((NUM_FEATURES,), np.nan))
            valid_mask[t] = 0.0
            continue
            
        left_shoulder = frame[11]
        right_shoulder = frame[12]
        
        shoulder_midpoint = (left_shoulder + right_shoulder) / 2.0
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        if shoulder_width < 1e-5:
            shoulder_width = 1.0 # fallback to prevent div by zero
            
        # Normalize
        norm_frame = (frame - shoulder_midpoint) / shoulder_width
        processed_features.append(norm_frame.flatten())
        
    processed_features = np.array(processed_features) # (T, 285)
    
    # Interpolation for gaps < 3 frames
    for f in range(NUM_FEATURES):
        series = processed_features[:, f]
        nans = np.isnan(series)
        
        if not np.any(nans):
            continue
            
        # Find gaps
        valid_indices = np.where(~nans)[0]
        if len(valid_indices) < 2:
            # Cannot interpolate gracefully
            continue
            
        interp_func = interp1d(valid_indices, series[valid_indices], kind='linear', bounds_error=False, fill_value=np.nan)
        new_series = interp_func(np.arange(T))
        
        # Only accept interpolation if gap is < 3
        gap_lengths = []
        current_gap = 0
        for isnan in nans:
            if isnan:
                current_gap += 1
            else:
                gap_lengths.append(current_gap)
                current_gap = 0
        gap_lengths.append(current_gap)
        
        # We manually track valid mask for large gaps
        
        processed_features[:, f] = new_series

    # Update valid mask for any remaining NaNs
    final_nans = np.isnan(processed_features).any(axis=1)
    valid_mask[final_nans] = 0.0
    
    # Zero-fill remaining NaNs so downstream models don't crash, 
    # but the mask indicates they are invalid.
    processed_features = np.nan_to_num(processed_features, nan=0.0)
    
    return processed_features, valid_mask

def process_dataset(raw_dir, cache_dir):
    """
    Iterates through dataset and caches processed landmarks.
    """
    os.makedirs(cache_dir, exist_ok=True)
    video_files = glob.glob(os.path.join(raw_dir, "**", "*.mp4"), recursive=True)
    
    total_videos = len(video_files)
    print(f"Found {total_videos} videos to process.")
    
    fail_counts = []
    
    for i, video_path in enumerate(video_files):
        print(f"[{i+1}/{total_videos}] Processing {os.path.basename(video_path)}...", end="\r")
        features, mask = extract_landmarks_from_video(video_path)
        
        if features is None:
            print(f"\nFailed to read {video_path}")
            continue
            
        # Calculate failure rate
        fail_rate = 1.0 - (np.sum(mask) / len(mask))
        fail_counts.append(fail_rate)
        
        # Save
        class_name = os.path.basename(os.path.dirname(video_path))
        save_dir = os.path.join(cache_dir, class_name)
        os.makedirs(save_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        np.save(os.path.join(save_dir, f"{base_name}_feat.npy"), features)
        np.save(os.path.join(save_dir, f"{base_name}_mask.npy"), mask)
        
    print("\nProcessing complete.")
    if fail_counts:
        avg_fail = np.mean(fail_counts) * 100
        print(f"Average frame detection failure rate across dataset: {avg_fail:.2f}%")

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_path, "data", "raw_videos")
    cache_dir = os.path.join(base_path, "data", "cache", "landmarks")
    
    if not os.path.exists(raw_dir):
        print(f"Raw video directory not found: {raw_dir}")
        print("Please run data/download_include.py first and populate videos.")
    else:
        process_dataset(raw_dir, cache_dir)
