"""
live_stream_extractor.py

Perception layer component for real-time live webcam extraction.
Extracts the exact same 95 landmarks, applies identical spatial/scale normalization,
and handles streaming input for the FastAPI backend.

Author: Ishaara System
"""

import numpy as np
import mediapipe as mp
import cv2

mp_holistic = mp.solutions.holistic

REDUCED_FACE_INDICES = [
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300, # Eyebrows
    61, 40, 37, 0, 267, 270, 291, 84, 17, 314      # Lips
]
NUM_LANDMARKS = 33 + 21 + 21 + 20
NUM_FEATURES = NUM_LANDMARKS * 3

class LiveExtractor:
    def __init__(self):
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, image_rgb):
        """
        Process a single RGB frame for real-time inference.
        Returns:
            features: np.array of shape (285,) or None if normalization fails
            is_valid: bool indicating if hand/pose detection succeeded
        """
        image_rgb.flags.writeable = False
        results = self.holistic.process(image_rgb)
        
        frame_data = np.full((NUM_LANDMARKS, 3), np.nan)
        
        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                frame_data[i] = [lm.x, lm.y, lm.z]
                
        if results.left_hand_landmarks:
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                frame_data[33 + i] = [lm.x, lm.y, lm.z]
                
        if results.right_hand_landmarks:
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                frame_data[33 + 21 + i] = [lm.x, lm.y, lm.z]
                
        if results.face_landmarks:
            for i, idx in enumerate(REDUCED_FACE_INDICES):
                lm = results.face_landmarks.landmark[idx]
                frame_data[33 + 21 + 21 + i] = [lm.x, lm.y, lm.z]
                
        # Normalization
        if np.isnan(frame_data[11, 0]) or np.isnan(frame_data[12, 0]):
            return np.zeros(NUM_FEATURES, dtype=np.float32), False, []
            
        left_shoulder = frame_data[11]
        right_shoulder = frame_data[12]
        
        shoulder_midpoint = (left_shoulder + right_shoulder) / 2.0
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        if shoulder_width < 1e-5:
            shoulder_width = 1.0
            
        norm_frame = (frame_data - shoulder_midpoint) / shoulder_width
        flat_features = norm_frame.flatten()
        
        # In ISL, one hand may be occluded or out of frame.
        # We consider the frame valid as long as pose is detected (which we verified above)
        # and at least one hand is detected.
        has_left = results.left_hand_landmarks is not None
        has_right = results.right_hand_landmarks is not None
        is_valid = has_left or has_right
        
        # Zero-fill for the model input
        flat_features = np.nan_to_num(flat_features, nan=0.0)
        
        # We can also return the raw landmarks for visual mapping on the frontend if needed
        raw_landmarks = frame_data[:, :2] # Just x, y for drawing
        raw_landmarks = np.nan_to_num(raw_landmarks, nan=-1.0).tolist()
        
        return flat_features.astype(np.float32), is_valid, raw_landmarks

    def close(self):
        self.holistic.close()
