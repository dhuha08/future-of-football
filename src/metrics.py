"""
Compute per-player metrics from positions CSV.

"""

import pandas as pd
import numpy as np
import os

def compute_metrics(csv_path):
    df = pd.read_csv(csv_path)
    out = []

    for tid, g in df.groupby('track_id'):
        g = g.sort_values('time')
        xs = g['x'].values
        ys = g['y'].values
        dist = np.sum(np.sqrt(np.diff(xs)**2 + np.diff(ys)**2))
        speeds = g['speed_px_per_s'].values
        avg_speed = np.nanmean(speeds)
        max_speed = np.nanmax(speeds)
        out.append({
            'track_id': int(tid),
            'distance_px': float(dist),
            'avg_speed_px_s': float(avg_speed),
            'max_speed_px_s': float(max_speed)
        })

    return pd.DataFrame(out)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--positions', required=True, help="CSV from detect_objects.py")
    args = parser.parse_args()

    metrics_df = compute_metrics(args.positions)
    out_file = args.positions.replace('.csv', '_metrics.csv')
    os.makedirs(os.path.dirname(out_file) or '.', exist_ok=True)
    metrics_df.to_csv(out_file, index=False)
    print(f"Saved metrics -> {out_file}")
    print(metrics_df)
