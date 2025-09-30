"""
Compute per-player metrics from positions CSV for AI-driven mistake detection.
Includes: distance, speed, touches per dribble, possession-based metrics, passing/shooting info.
"""

import pandas as pd
import numpy as np
import os


# Helper functions

def compute_distance(p1, p2):
    return np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)

def touches_per_dribble(player_frames):
    touches = 0
    consecutive = 0
    for i in range(len(player_frames)):
        if player_frames.iloc[i].get('has_ball', False):
            consecutive += 1
        else:
            if consecutive > 1:
                touches += consecutive
            consecutive = 0
    return touches

def distance_to_nearest_defender(player, frame_data):
    defenders = frame_data[frame_data['team'] != player['team']]
    if defenders.empty:
        return np.nan
    distances = defenders.apply(lambda d: compute_distance(player, d), axis=1)
    return distances.min()

def open_teammate_available(player, frame_data, threshold=3.0):
    teammates = frame_data[frame_data['team'] == player['team']]
    defenders = frame_data[frame_data['team'] != player['team']]
    for _, mate in teammates.iterrows():
        dist_to_def = defenders.apply(lambda d: compute_distance(mate, d), axis=1).min()
        if dist_to_def > threshold:
            return True
    return False

def in_scoring_position(player):
    x, y = player['x'], player['y']
    return x > 80 and 15 < y < 35  # adjust field scale as needed

def pass_speed(player, target, fps=30):
    dist = compute_distance(player, target)
    return dist / (1/fps)


# Main computation function

def compute_metrics(csv_path, ball_csv=None):
    df = pd.read_csv(csv_path)
    
    # Basic distance and speed metrics from old script
    out_basic = []
    for tid, g in df.groupby('track_id'):
        g = g.sort_values('time')
        xs = g['x'].values
        ys = g['y'].values
        dist = np.sum(np.sqrt(np.diff(xs)**2 + np.diff(ys)**2))
        speeds = g.get('speed_px_per_s', np.zeros(len(xs)))
        avg_speed = np.nanmean(speeds)
        max_speed = np.nanmax(speeds)
        out_basic.append({
            'track_id': int(tid),
            'distance_px': float(dist),
            'avg_speed_px_s': float(avg_speed),
            'max_speed_px_s': float(max_speed)
        })
    metrics_df = pd.DataFrame(out_basic)

    # If ball CSV is provided, compute AI-related metrics
    if ball_csv and os.path.exists(ball_csv):
        ball_data = pd.read_csv(ball_csv)
        df = df.merge(ball_data[['frame','track_id','has_ball']], on=['frame','track_id'], how='left')
        df['has_ball'] = df['has_ball'].fillna(False)

        df['touches_per_dribble'] = 0
        df['distance_to_nearest_defender'] = 0
        df['open_teammate_available'] = False
        df['in_scoring_position'] = False
        df['pass_speed'] = 0
        df['pass_success'] = True
        df['shot_on_target'] = False

        for frame_idx in df['frame'].unique():
            frame_data = df[df['frame'] == frame_idx]
            for idx, player in frame_data.iterrows():
                df.at[idx, 'distance_to_nearest_defender'] = distance_to_nearest_defender(player, frame_data)
                df.at[idx, 'open_teammate_available'] = open_teammate_available(player, frame_data)
                df.at[idx, 'in_scoring_position'] = in_scoring_position(player)
                # Pass/shot heuristics can be refined later
        # Compute touches per dribble per player
        for tid in df['track_id'].unique():
            player_frames = df[df['track_id'] == tid]
            df.loc[df['track_id'] == tid, 'touches_per_dribble'] = touches_per_dribble(player_frames)

        # Merge these new metrics into metrics_df
        agg_cols = ['touches_per_dribble','distance_to_nearest_defender',
                    'open_teammate_available','in_scoring_position',
                    'pass_speed','pass_success','shot_on_target']
        df_agg = df.groupby('track_id')[agg_cols].max().reset_index()
        metrics_df = metrics_df.merge(df_agg, on='track_id', how='left')

    return metrics_df


# Command line interface

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--positions', required=True, help="CSV from detect_objects.py")
    parser.add_argument('--ball', required=False, help="Optional ball tracking CSV")
    args = parser.parse_args()

    metrics_df = compute_metrics(args.positions, args.ball)
    out_file = args.positions.replace('.csv', '_metrics.csv')
    os.makedirs(os.path.dirname(out_file) or '.', exist_ok=True)
    metrics_df.to_csv(out_file, index=False)
    print(f"Saved metrics -> {out_file}")
    print(metrics_df)

