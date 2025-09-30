"""
AI-driven mistake detection and advice for all player videos.
Uses updated metrics.py for per-player metrics computation.
"""

import cv2
import pandas as pd
import numpy as np
import glob
import os
import subprocess


# AI advice helpers

def find_similar_legends(player_data, legend_df, mistake_type):
    action = "dribbling" if mistake_type in ['too_many_touches','loses_possession','dribble_into_pressure'] \
             else "passing" if mistake_type in ['pass_to_crowded_area','slow_pass','wrong_teammate','no_pass_opportunity'] \
             else "goal"
    relevant = legend_df[legend_df['action_type'] == action]
    if relevant.empty:
        return None
    relevant['dist'] = np.sqrt((relevant['x'] - player_data['x'])**2 + (relevant['y'] - player_data['y'])**2)
    return relevant.nsmallest(5,'dist')

def generate_advice(player_data, mistake_type, legend_df):
    similar = find_similar_legends(player_data, legend_df, mistake_type)
    if similar is None or similar.empty:
        return "Advice: Consider safer or more effective options."
    best_action_row = similar.loc[similar['success'].idxmax()]
    return f"Advice: {best_action_row['advice']}"

def highlight_mistake(frame, player_data, advice_text):
    # Draw circle on player
    cv2.circle(frame, (int(player_data['x']), int(player_data['y'])), 25, (0,0,255), 2)
    # Overlay advice
    cv2.putText(frame, advice_text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Video Analysis", frame)
    cv2.waitKey(3000)  # pause 3 seconds


# Mistake definitions

mistakes = {
    "dribbling": {
        "too_many_touches": lambda d: d['touches_per_dribble'] > 3,
        "loses_possession": lambda d: not d.get('has_ball', True),
        "dribble_into_pressure": lambda d: d['distance_to_nearest_defender'] < 1.5
    },
    "passing": {
        "pass_to_crowded_area": lambda d: d['distance_to_nearest_defender'] < 1.5,
        "slow_pass": lambda d: d['pass_speed'] < 5,
        "wrong_teammate": lambda d: not d.get('pass_success', True),
        "no_pass_opportunity": lambda d: d.get('has_ball', False) and d.get('open_teammate_available', False)
    },
    "goal": {
        "missed_goal": lambda d: not d.get('shot_on_target', False) and d.get('in_scoring_position', False)
    }
}


# Analyze single video

def analyze_video(video_path, tracking_csv, ball_csv, legend_df):
    # Step 1: Compute metrics using metrics.py
    metrics_out = tracking_csv.replace('.csv','_metrics.csv')
    cmd = ['python', 'metrics.py', '--positions', tracking_csv]
    if ball_csv and os.path.exists(ball_csv):
        cmd += ['--ball', ball_csv]
    subprocess.run(cmd, check=True)
    
    metrics_df = pd.read_csv(metrics_out)

    # Step 2: Open video
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_data = metrics_df[metrics_df['frame'] == frame_idx] if 'frame' in metrics_df.columns else metrics_df
        for _, player in frame_data.iterrows():
            for action, checks in mistakes.items():
                for mistake_type, condition in checks.items():
                    if condition(player):
                        advice = generate_advice(player, mistake_type, legend_df)
                        highlight_mistake(frame, player, advice)
        frame_idx +=1
        cv2.imshow("Video Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# Loop through all videos

def analyze_all_videos(video_folder='videos/players/',
                       tracking_folder='tracking_data/',
                       ball_folder='ball_data/',
                       legend_csv='legend_tracking.csv'):
    
    legend_df = pd.read_csv(legend_csv)
    video_files = glob.glob(os.path.join(video_folder, "*.mp4"))
    
    for video_file in video_files:
        base = os.path.basename(video_file).replace('.mp4','')
        tracking_csv = os.path.join(tracking_folder, f"{base}_tracking.csv")
        ball_csv = os.path.join(ball_folder, f"{base}_ball.csv")
        if not os.path.exists(tracking_csv):
            print(f"Skipping {video_file}: tracking CSV not found")
            continue
        print(f"Analyzing {video_file}...")
        analyze_video(video_file, tracking_csv, ball_csv if os.path.exists(ball_csv) else None, legend_df)


# Run

if __name__ == "__main__":
    analyze_all_videos()
