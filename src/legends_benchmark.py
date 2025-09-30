import os, glob, pandas as pd

metrics_files = glob.glob("data/positions/*_positions_metrics.csv")
dfs = []

for f in metrics_files:
    df = pd.read_csv(f)
    # infer player name from filename
    player_name = os.path.basename(f).split("_")[0]
    df['player_name'] = player_name
    dfs.append(df)

benchmarks = pd.concat(dfs, ignore_index=True)
benchmarks.to_csv("data/positionns/legends_benchmark.csv", index=False)