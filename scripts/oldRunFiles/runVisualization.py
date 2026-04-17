from modules.crowd_Visualization import AnomalyDetector, CrowdPlotter, RiskVisualizer

videoName = 'umn1'

# Anomaly detection
detector = AnomalyDetector(f"output/risk/risk_{videoName}.csv", risk_threshold=0.65, min_event_length=3)
anomalies = detector.detect()   # returns list of (start, end) tuples
detector.summary()              # prints to console
detector.plot(save=True, show=False)  # save without blocking

# Plots
plotter = CrowdPlotter(output_dir="output/plots")
plotter.plot_risk(f"output/risk/risk_{videoName}.csv", show=False)
plotter.plot_velocity(f"output/features/features_{videoName}.csv", show=False)

# Live heatmap (blocks until ESC)
viz = RiskVisualizer(f"data/{videoName}.mp4", f"output/risk/risk_{videoName}.csv")
viz.run()