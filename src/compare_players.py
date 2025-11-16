"""
Compare players from a match against legends benchmark
Outputs a CSV with closest legend and similarity score
"""

import pandas as pd
import numpy as np
import os

# CONFIG 
MATCH_CSV = "data/positions/match1_positions_metrics.csv"
LEGENDS_CSV = "data/positions/legends_benchmark.csv"
OUTPUT_CSV = "data/positions/match1_vs_legends.csv"

# LOAD DATA
players = pd.read_csv(MATCH_CSV)
benchmarks = pd.read_csv(LEGENDS_CSV)

# METRICS TO COMPARE
metrics_cols = ['distance_px','avg_speed_px_s','max_speed_px_s']

# NORMALIZE AGAINST LEGENDS RANGE
legend_min = benchmarks[metrics_cols].min()
legend_max = benchmarks[metrics_cols].max()

players_norm = players.copy()
for col in metrics_cols:
    players_norm[col+'_norm'] = (players[col] - legend_min[col]) / (legend_max[col] - legend_min[col])

# COMPUTE DISTANCE TO EACH LEGEND 
results = []
for _, p in players_norm.iterrows():
    p_norm = [p[c+'_norm'] for c in metrics_cols]
    for _, l in benchmarks.iterrows():
        l_norm = [(l[c]-legend_min[c])/(legend_max[c]-legend_min[c]) for c in metrics_cols]
        dist = np.linalg.norm(np.array(p_norm)-np.array(l_norm))
        score = 1 - dist  # higher score = closer to legend
        results.append({
            'track_id': p['track_id'],
            'legend': l['player_name'],
            'distance': dist,
            'score': score
        })

similarity_df = pd.DataFrame(results)

# FIND CLOSEST LEGEND FOR EACH PLAYER
closest = similarity_df.loc[similarity_df.groupby('track_id')['distance'].idxmin()]

# SAVE OUTPUT
os.makedirs(os.path.dirname(OUTPUT_CSV) or '.', exist_ok=True)
closest.to_csv(OUTPUT_CSV, index=False)

print(f"Comparison table saved -> {OUTPUT_CSV}")
print(closest)

# GENERATE FEEDBACK FOR EACH PLAYER

feedback_strings = []

for _, row in closest.iterrows():
    text = generate_feedback(row, metrics_cols)
    feedback_strings.append(f"=== Player {row['track_id']} ===\n{text}\n")

# Save feedback as text file
with open(OUTPUT_FEEDBACK, "w") as f:
    f.write("\n".join(feedback_strings))

print(f"\nFeedback saved -> {OUTPUT_FEEDBACK}")
print("\n".join(feedback_strings))