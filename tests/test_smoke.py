import subprocess
import sys
import tempfile
import unittest
import ast
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class ReplicationSmokeTest(unittest.TestCase):
    def test_runner_creates_key_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "run_replication.py"),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            expected = {
                "table0_baseline_summary.csv",
                "table1_sample_counts.csv",
                "table2_k_sensitivity_summary.csv",
                "table3_anova_computed.csv",
                "table4_manova_computed.csv",
                "table4_manova_comparison.csv",
                "table5_classifier_metrics.csv",
                "table5_classifier_comparison.csv",
                "fig1_k_sensitivity.png",
                "fig6_classifier_roc_curves.png",
            }
            files = {path.name for path in output_dir.iterdir()}
            missing = expected - files
            self.assertFalse(missing, f"Missing expected outputs: {sorted(missing)}")

            baseline = pd.read_csv(output_dir / "table0_baseline_summary.csv").iloc[0]
            self.assertEqual(int(baseline["n_init"]), 1)
            self.assertEqual(
                ast.literal_eval(baseline["cluster_sizes"]),
                {0: 56, 1: 33, 2: 35, 3: 2, 4: 17},
            )
            sensitivity = pd.read_csv(output_dir / "table2_k_sensitivity_summary.csv")
            self.assertEqual(
                dict(zip(sensitivity["n_init"], sensitivity["best_k_by_silhouette"])),
                {1: 5, 5: 6, 10: 4},
            )


if __name__ == "__main__":
    unittest.main()
