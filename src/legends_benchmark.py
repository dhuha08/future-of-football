import os, glob, pandas as pd

# Where legend metrics are stored
legend_files = glob.glob("data/positions/*_positions_metrics.csv")

dfs = []
for f in legend_files:
    df = pd.read_csv(f)
    # Extract legend name from filename
    filename = os.path.basename(f)
    legend_name = filename.split("_")[0].capitalize()  # Messi, Ronaldo, etc.
    df['player_name'] = legend_name
    dfs.append(df)

# Combine all legends into one CSV
benchmarks = pd.concat(dfs, ignore_index=True)
os.makedirs("data/positions", exist_ok=True)
benchmarks.to_csv("data/positions/legends_benchmark.csv", index=False)

print("Legends benchmark created -> data/positions/legends_benchmark.csv")
