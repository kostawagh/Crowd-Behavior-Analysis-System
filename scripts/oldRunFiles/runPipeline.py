"""
run_analytics.py
Execution script for the analytics pipeline.

This is the only file you need to edit when running on a new video.
Just change VIDEO_NAME and VIDEO_PATH below, then run:

    python run_analytics.py
"""

from modules.pipeline import AnalyticsPipeline

# ─────────────────────────────────────────
# CONFIG — edit these
# ─────────────────────────────────────────
VIDEO_NAME  = "umn1"                  # must match tracking_<name>.csv in output/tracking/
VIDEO_PATH  = "data/umn1.mp4"    # original video (needed only for frame dimensions)
OUTPUT_ROOT = "output"               # root folder — all sub-dirs created automatically

# Optional: override grid size or risk weights
GRID_X   = 5
GRID_Y   = 5
WEIGHTS  = None    # None = use defaults {density:0.25, congestion:0.30, turbulence:0.20, flow_conflict:0.25}

# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────
if __name__ == "__main__":
    pipeline = AnalyticsPipeline(
        video_name  = VIDEO_NAME,
        video_path  = VIDEO_PATH,
        output_root = OUTPUT_ROOT,
        grid_x      = GRID_X,
        grid_y      = GRID_Y,
        weights     = WEIGHTS,
    )

    paths = pipeline.run()

    # paths dict contains all output file locations:
    # paths["features"]  — motion features CSV
    # paths["analytics"] — grid analytics CSV
    # paths["plot"]      — analytics plot PNG
    # paths["risk"]      — risk scores CSV


# ─────────────────────────────────────────
# Run multiple videos in one go:
# ─────────────────────────────────────────
# VIDEOS = [
#     ("umn1", "data/umn/umn1.mp4"),
#     ("umn2", "data/umn/umn2.mp4"),
#     ("umn3", "data/umn/umn3.mp4"),
# ]
# for name, path in VIDEOS:
#     AnalyticsPipeline(name, path, OUTPUT_ROOT).run()