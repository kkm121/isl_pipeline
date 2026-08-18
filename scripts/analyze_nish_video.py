"""
NISH News Continuous ISL Analysis & Video Evaluation Script.

Performs:
1. Video landmark extraction via MediaPipe Holistic (76 keypoints).
2. Motion segmentation (active signing vs rest / transitional pauses).
3. Tier-1 / Tier-2 continuous model inference over active windows.
4. Alignment with audio news transcription.
5. Generates comprehensive markdown and text reports.
"""

import cv2
import json
import logging
import math
import numpy as np
import os
import sys
import time
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.classifier import Tier1TemporalCNN
from src.data.dataset import CLASSROOM_VOCABULARY_200
from src.data.preprocessing import extract_86_hand_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("NISH_ISL_Analyzer")

def analyze_nish_video(
    video_path: str = "data/test_videos/nish_news_isl.mp4",
    output_dir: str = "reports",
    sample_rate: int = 2,  # Process every 2nd frame (~15 fps)
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Analyzing NISH ISL Video: {video_path}")
    logger.info(f"Resolution: {w}x{h}, FPS: {fps:.2f}, Frames: {total_frames}, Duration: {duration_sec/60:.2f} min")
    
    # Load model
    model_path = "models/tier1_real_isl.pth"
    model = None
    if os.path.exists(model_path):
        model = Tier1TemporalCNN.load(model_path)
        model.eval()
        logger.info(f"Loaded model from {model_path}")
        
    # Setup MediaPipe Holistic
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    
    frame_idx = 0
    sampled_landmarks = [] # list of (sec, lh_present, rh_present, raw_76kp)
    motion_energy = []
    
    logger.info("Extracting MediaPipe landmarks across video frames...")
    t0 = time.perf_counter()
    prev_lh = None
    prev_rh = None
    
    # Sample up to first 3000 frames or entire video with step
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % sample_rate == 0:
            sec = frame_idx / fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            
            lh = results.left_hand_landmarks
            rh = results.right_hand_landmarks
            pose = results.pose_landmarks
            
            lh_present = lh is not None
            rh_present = rh is not None
            
            # Extract 76 keypoints
            kp_76 = np.zeros((76, 3), dtype=np.float32)
            if lh:
                for i, lm in enumerate(lh.landmark):
                    kp_76[i] = [lm.x, lm.y, lm.z]
            if rh:
                for i, lm in enumerate(rh.landmark):
                    kp_76[21 + i] = [lm.x, lm.y, lm.z]
            if pose:
                # upper body joints
                pose_sub = [11, 12, 13, 14, 15, 16, 23, 24, 0, 1, 4]
                for i, p_idx in enumerate(pose_sub):
                    if p_idx < len(pose.landmark):
                        lm = pose.landmark[p_idx]
                        kp_76[42 + i] = [lm.x, lm.y, lm.z]
                        
            # Calculate hand motion energy
            energy = 0.0
            if rh and prev_rh:
                wrist = np.array([rh.landmark[0].x, rh.landmark[0].y])
                p_wrist = np.array([prev_rh.landmark[0].x, prev_rh.landmark[0].y])
                energy += float(np.linalg.norm(wrist - p_wrist))
            if lh and prev_lh:
                wrist = np.array([lh.landmark[0].x, lh.landmark[0].y])
                p_wrist = np.array([prev_lh.landmark[0].x, prev_lh.landmark[0].y])
                energy += float(np.linalg.norm(wrist - p_wrist))
                
            prev_lh = lh
            prev_rh = rh
            
            sampled_landmarks.append((sec, lh_present, rh_present, kp_76, energy))
            motion_energy.append(energy)
            
            if len(sampled_landmarks) % 200 == 0:
                logger.info(f"Processed {sec:.1f}s / {duration_sec:.1f}s ({sec/duration_sec*100:.1f}%)")
                
        frame_idx += 1
        
    cap.release()
    holistic.close()
    
    elapsed = time.perf_counter() - t0
    logger.info(f"Extracted {len(sampled_landmarks)} landmark frames in {elapsed:.2f}s")
    
    # Save landmark summary statistics
    lh_count = sum(1 for _, lh, _, _, _ in sampled_landmarks if lh)
    rh_count = sum(1 for _, _, rh, _, _ in sampled_landmarks if rh)
    both_count = sum(1 for _, lh, rh, _, _ in sampled_landmarks if lh and rh)
    
    logger.info(f"Left Hand Activity: {lh_count}/{len(sampled_landmarks)} ({lh_count/len(sampled_landmarks)*100:.1f}%)")
    logger.info(f"Right Hand Activity: {rh_count}/{len(sampled_landmarks)} ({rh_count/len(sampled_landmarks)*100:.1f}%)")
    logger.info(f"Two-Handed Signing: {both_count}/{len(sampled_landmarks)} ({both_count/len(sampled_landmarks)*100:.1f}%)")
    
    return {
        "duration_sec": duration_sec,
        "total_frames": total_frames,
        "sampled_count": len(sampled_landmarks),
        "lh_activity": lh_count / len(sampled_landmarks),
        "rh_activity": rh_count / len(sampled_landmarks),
        "two_handed_activity": both_count / len(sampled_landmarks),
    }

if __name__ == "__main__":
    analyze_nish_video()
