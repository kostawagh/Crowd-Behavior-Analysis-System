"""
main_pipeline.py
Real-Time Crowd Risk Prediction — Detection + Tracking Pipeline

Changes vs original:
  - CONFIG expanded with head-detection, preprocessing, tiling, ByteTrack flags
  - Loosened tracker params (min_hits, max_age, iou_track) for dense crowds
  - Lowered conf_thres + raised iou_thres + larger img_size for head models
  - Added conf_low_thres for ByteTrack's dual-threshold association
"""

import cv2
import numpy as np
import torch
import time
import os
import winsound

from modules.crowd_Detector_Yolo   import CrowdDetector
# from modules.crowd_Detector_Rtdetr   import CrowdDetector
from modules.crowd_Tracker    import CrowdTracker
from modules.trajectory_Logger     import TrackLogger
from modules.tracking_Visualizer import Visualizer

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
CONFIG = {
    # ── Input / output ───────────────────
    "video_path"  : "data/umn1.mp4",
    "output_dir"  : "output/tracking",

    # ── Model ────────────────────────────
    "model_path"  : "models/yolov8n_crowdhuman.pt",
    #"model_path"  : "models/rtdetr-l.pt",
    #"model_path"  : "models/yolov8m.pt",

    # Adjusts NMS and overlap thresholds automatically.
    "detect_heads": False,

    # ── Detection thresholds ─────────────
    # Lowered conf (0.25→0.20): head detections are inherently lower confidence
    # Raised iou (0.45→0.60): heads are close but distinct — allow tighter packing
    # Larger img_size (960→1280): heads are small targets, need higher resolution
    "conf_thres"  : 0.2,
    "iou_thres"   : 0.5,
    "img_size"    : 640,

    # ── Preprocessing ────────────────────
    # True = unsharp mask + CLAHE before inference (helps with motion blur)
    "use_preprocess": True,

    # ── Tiled inference ──────────────────
    # True = full frame + 4 quadrant tiles merged (catches small/distant persons)
    # ~5× compute — raise frame_skip to 3+ if using this
    "use_tiling"  : True,

    # ── Tracking ─────────────────────────
    # ByteTrack uses both high+low conf dets to recover occluded persons.
    # Requires: ultralytics >= 8.0
    # Falls back to SORT automatically if import fails.
    "use_bytetrack"   : False,

    # Dual-threshold for ByteTrack's low-conf association pass
    "conf_low_thres"  : 0.10,

    # Loosened vs original (min_hits 3→2, max_age 10→20, iou_track 0.30→0.20)
    # Reason: head detections are noisier; longer max_age bridges occlusion gaps
    "min_hits"        : 2,
    "max_age"         : 20,
    "iou_track"       : 0.20,

    # ── Misc ─────────────────────────────
    "frame_skip"      : 2,
    "min_track_len"   : 5,
    "display_w"       : 900,
    "display_h"       : 600,
}

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    print(f"GPU available: {torch.cuda.is_available()}")
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    video_name = os.path.splitext(os.path.basename(CONFIG["video_path"]))[0]

    # --- Init modules ---
    detector   = CrowdDetector(CONFIG)
    tracker    = CrowdTracker(CONFIG)
    logger     = TrackLogger(CONFIG, video_name)
    visualizer = Visualizer(CONFIG)

    cap = cv2.VideoCapture(CONFIG["video_path"])
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {CONFIG['video_path']}")

    cv2.namedWindow("Crowd Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Crowd Tracking", CONFIG["display_w"], CONFIG["display_h"])

    frame_no = 0

    with logger:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_no += 1
            if frame_no % CONFIG["frame_skip"] != 0:
                continue

            # 1. Detect
            dets, inference_ms = detector.detect(frame)

            # 2. Track
            tracks = tracker.update(dets)

            # 3. Log
            logger.log_frame(frame_no, tracks, inference_ms)

            # 4. Visualize
            display = visualizer.draw(frame, tracks, frame_no, inference_ms)
            cv2.imshow("Crowd Tracking", display)

            print(f"Frame {frame_no:04d} | Det: {len(dets):3d} | "
                  f"Tracked: {len(tracks):3d} | "
                  f"Inf: {inference_ms:.1f}ms")

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

    # Post-process: remove short/noisy tracks from CSV
    logger.filter_short_tracks(CONFIG["min_track_len"])

    print(f"\nDone. CSV saved: {logger.csv_path}")

if __name__ == "__main__":
    main()
