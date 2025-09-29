# src/evaluation.py

import numpy as np
import json
from pathlib import Path


class MetricsExtractor:
    def __init__(self, output_dir="data/metrics"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_from_track(self, track):
        """
        Given a player's track (list of (x, y, t)),
        compute metrics like avg/max speed and total distance.
        """
        positions = track  # expect [(x, y, t), ...]

        if len(positions) < 2:
            return {
                "avg_speed": 0.0,
                "max_speed": 0.0,
                "total_distance": 0.0
            }

        speeds = []
        total_distance = 0.0

        for i in range(1, len(positions)):
            x1, y1, t1 = positions[i - 1]
            x2, y2, t2 = positions[i]
            dt = t2 - t1

            if dt > 0:
                dist = np.linalg.norm([x2 - x1, y2 - y1])
                total_distance += dist
                speeds.append(dist / dt)

        metrics = {
            "avg_speed": float(np.mean(speeds)) if speeds else 0.0,
            "max_speed": float(np.max(speeds)) if speeds else 0.0,
            "total_distance": float(total_distance)
        }

        return metrics

    def save_metrics(self, player_id, metrics, match_name="unknown"):
        filepath = self.output_dir / f"{match_name}_{player_id}.json"
        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=2)


class BenchmarkEvaluator:
    def __init__(self, benchmark_file="data/legend_benchmark.json"):
        self.benchmark_file = Path(benchmark_file)
        self.benchmark = self._load_benchmark()

    def _load_benchmark(self):
        if self.benchmark_file.exists():
            with open(self.benchmark_file, "r") as f:
                return json.load(f)
        return {}

    def compare(self, player_metrics):
        """
        Compare a player's metrics to the benchmark.
        Currently: percentage similarity per metric.
        """
        results = {}
        for key, value in player_metrics.items():
            if key in self.benchmark:
                benchmark_value = self.benchmark[key]
                similarity = 1 - abs(value - benchmark_value) / (benchmark_value + 1e-6)
                results[key] = round(similarity * 100, 2)  # %
        return results
