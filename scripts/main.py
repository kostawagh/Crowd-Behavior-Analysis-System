"""
main.py
────────────────────────────────────────────────────────────
Single entry point for the full Crowd Risk Prediction system.

USAGE — single video:
    1. Set VIDEO_PATH below.
    2. Run:  python main.py

USAGE — batch (multiple videos):
    1. Set BATCH_VIDEOS list below (VIDEO_PATH is ignored in batch mode).
    2. Set BATCH_MODE = True.
    3. Run:  python main.py

The three stages run in order:
    Stage 1 — Detection + Tracking  (runDetectTrack)
    Stage 2 — Analytics Pipeline    (runPipeline)
    Stage 3 — Visualization         (runVisualization)

────────────────────────────────────────────────────────────
"""

import os
import sys

# ═══════════════════════════════════════════════════════════
# Video Path ⭐
# ═══════════════════════════════════════════════════════════
VIDEO_PATH = "data/umn1.mp4"          # ← only line you need to change for a single video

# ── Batch mode ──────────────────────────────────────────────
# Set BATCH_MODE = True and list your videos below.
BATCH_MODE   = False
BATCH_VIDEOS = [
    "data/umn3.mp4",
]

# ── Optional overrides ──────────────────────────────────────
OUTPUT_ROOT   = "output"                        # root output folder   ⭐
MODEL_PATH    = "models/yolov8n_crowdhuman.pt"  # detection model used ⭐
# MODEL_PATH  = "models/rtdetr-l.pt"  # uncomment to switch to RT-DETR

GRID_X        = 5
GRID_Y        = 5
WEIGHTS       = None   # None = use defaults

# ── Stage toggles (set False to skip a stage) ───────────────
RUN_DETECT_TRACK  = True
RUN_PIPELINE      = True
RUN_VISUALIZATION = True

# ── Detection / tracking knobs ──────────────────────────────
DETECT_CONFIG = {
    "conf_thres"      : 0.2,
    "iou_thres"       : 0.5,
    "img_size"        : 640,
    "detect_heads"    : False,
    "use_preprocess"  : True,
    "use_tiling"      : True,
    "use_bytetrack"   : False,
    "conf_low_thres"  : 0.10,
    "min_hits"        : 2,
    "max_age"         : 20,
    "iou_track"       : 0.20,
    "frame_skip"      : 2,
    "min_track_len"   : 5,
    "display_w"       : 900,
    "display_h"       : 600,
}

# ── Visualization knobs ─────────────────────────────────────
VIZ_CONFIG = {
    "risk_threshold"    : 0.65,
    "min_event_length"  : 3,
    "show_plots"        : False,   # False = save to disk, don't block
    "run_live_heatmap"  : True,    # False = skip the blocking live heatmap
}

# ═══════════════════════════════════════════════════════════
# Derived paths — do not edit
# ═══════════════════════════════════════════════════════════
VIDEO_NAME   = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
TRACKING_DIR = os.path.join(OUTPUT_ROOT, "tracking")
RISK_CSV     = os.path.join(OUTPUT_ROOT, "risk",     f"risk_{VIDEO_NAME}.csv")
FEATURES_CSV = os.path.join(OUTPUT_ROOT, "features", f"features_{VIDEO_NAME}.csv")


# ═══════════════════════════════════════════════════════════
# Stage 1 — Detection + Tracking
# ═══════════════════════════════════════════════════════════
def run_detect_track():
    print("\n" + "─" * 60)
    print("STAGE 1 — Detection + Tracking")
    print("─" * 60)

    import cv2
    import torch
    from modules.crowd_Detector_Yolo  import CrowdDetector          #⭐
    # from modules.crowd_Detector_Rtdetr import CrowdDetector       #⭐
    from modules.crowd_Tracker        import CrowdTracker
    from modules.trajectory_Logger    import TrackLogger
    from modules.tracking_Visualizer  import Visualizer

    config = {
        "video_path" : VIDEO_PATH,
        "output_dir" : TRACKING_DIR,
        "model_path" : MODEL_PATH,
        **DETECT_CONFIG,
    }

    os.makedirs(TRACKING_DIR, exist_ok=True)

    print(f"GPU available: {torch.cuda.is_available()}")
    print(f"Video        : {VIDEO_PATH}")
    print(f"Model        : {MODEL_PATH}")

    detector   = CrowdDetector(config)
    tracker    = CrowdTracker(config)
    logger     = TrackLogger(config, VIDEO_NAME)
    visualizer = Visualizer(config)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

    cv2.namedWindow("Crowd Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Crowd Tracking", config["display_w"], config["display_h"])

    frame_no = 0

    with logger:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_no += 1
            if frame_no % config["frame_skip"] != 0:
                continue

            try:
                dets, inference_ms = detector.detect(frame)
            except Exception as e:
                print(f"[WARN] Frame {frame_no}: detection failed — {e}")
                continue

            tracks = tracker.update(dets)
            logger.log_frame(frame_no, tracks, inference_ms)

            display = visualizer.draw(frame, tracks, frame_no, inference_ms)
            cv2.imshow("Crowd Tracking", display)

            print(f"Frame {frame_no:04d} | Det: {len(dets):3d} | "
                  f"Tracked: {len(tracks):3d} | Inf: {inference_ms:.1f}ms")

            if cv2.waitKey(1) & 0xFF == 27:
                print("[INFO] ESC pressed — stopping early.")
                break

    cap.release()
    cv2.destroyAllWindows()

    logger.filter_short_tracks(config["min_track_len"])
    print(f"\n[Stage 1 done] CSV saved: {logger.csv_path}")


# ═══════════════════════════════════════════════════════════
# Stage 2 — Analytics Pipeline
# ═══════════════════════════════════════════════════════════
def run_pipeline():
    print("\n" + "─" * 60)
    print("STAGE 2 — Analytics Pipeline")
    print("─" * 60)

    from modules.pipeline import AnalyticsPipeline

    pipeline = AnalyticsPipeline(
        video_name  = VIDEO_NAME,
        video_path  = VIDEO_PATH,
        output_root = OUTPUT_ROOT,
        grid_x      = GRID_X,
        grid_y      = GRID_Y,
        weights     = WEIGHTS,
    )

    paths = pipeline.run()

    print(f"[Stage 2 done] Outputs:")
    for key, path in paths.items():
        print(f"  {key:<12} → {path}")


# ═══════════════════════════════════════════════════════════
# Stage 3 — Visualization
# ═══════════════════════════════════════════════════════════
def run_visualization():
    print("\n" + "─" * 60)
    print("STAGE 3 — Visualization")
    print("─" * 60)

    # Guard: check files exist before importing heavy deps
    for label, path in [("risk CSV", RISK_CSV), ("features CSV", FEATURES_CSV)]:
        if not os.path.exists(path):
            print(f"[ERROR] {label} not found: {path}")
            print("        Did Stage 2 complete successfully?")
            sys.exit(1)

    from modules.crowd_Visualization import AnomalyDetector, CrowdPlotter, RiskVisualizer

    show = VIZ_CONFIG["show_plots"]

    # Anomaly detection
    detector = AnomalyDetector(
        RISK_CSV,
        risk_threshold  = VIZ_CONFIG["risk_threshold"],
        min_event_length= VIZ_CONFIG["min_event_length"],
    )
    anomalies = detector.detect()   # list of (start_frame, end_frame) tuples
    detector.summary()
    if anomalies:
        print(f"[Anomalies] {len(anomalies)} event(s) detected:")
        for start, end in anomalies:
            print(f"  frames {start} → {end}")
    else:
        print("[Anomalies] No anomalous events detected.")
    detector.plot(save=True, show=show)

    # Plots
    plotter = CrowdPlotter(output_dir=os.path.join(OUTPUT_ROOT, "plots"))
    plotter.plot_risk(RISK_CSV,     show=show)
    plotter.plot_velocity(FEATURES_CSV, show=show)

    # Live heatmap (blocks until ESC)
    if VIZ_CONFIG["run_live_heatmap"]:
        print("\n[INFO] Starting live heatmap — press ESC to exit.")
        viz = RiskVisualizer(VIDEO_PATH, RISK_CSV)
        viz.run()
    else:
        print("[INFO] Live heatmap skipped (run_live_heatmap=False).")

    print(f"\n[Stage 3 done]")


# ═══════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] Video not found: {VIDEO_PATH}")
        print("        Update VIDEO_PATH at the top of main.py and try again.")
        sys.exit(1)

    print(f"\n{'═' * 60}")
    print(f"  Crowd Risk Prediction Pipeline")
    print(f"  Video : {VIDEO_PATH}")
    print(f"  Name  : {VIDEO_NAME}")
    print(f"{'═' * 60}")

    try:
        if RUN_DETECT_TRACK:
            run_detect_track()

        if RUN_PIPELINE:
            run_pipeline()

        if RUN_VISUALIZATION:
            run_visualization()

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        sys.exit(0)

    # Cross-platform completion sound
    try:
        import winsound
        for freq in (800, 1000, 1200):
            winsound.Beep(freq, 200)
    except ImportError:
        print("\a") 

    print(f"\n{'═' * 60}")
    print(f"  All done!  Outputs in: {OUTPUT_ROOT}/")
    print(f"{'═' * 60}\n")