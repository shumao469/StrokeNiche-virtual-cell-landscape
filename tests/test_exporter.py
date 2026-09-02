import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_exporter_rejects_conflicting_baseline_rows(tmp_path):
    rows = []
    for perturbation, dc, dr in [("p1", -0.1, 0.1), ("p2", -0.2, 0.2)]:
        for index in range(3):
            rows.append(
                {
                    "obs_name": f"c{index}",
                    "latent1": float(index) + (0.5 if perturbation == "p2" and index == 0 else 0),
                    "latent2": float(index),
                    "perturbation": perturbation,
                    "baseline_core": 0.5,
                    "baseline_repair": 0.5,
                    "candidate_mean_delta_core": dc,
                    "candidate_mean_delta_repair": dr,
                }
            )
    source = tmp_path / "response.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    script = Path(__file__).resolve().parents[1] / "scripts" / "export_visualizer_inputs.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--response-table",
            str(source),
            "--cells-out",
            str(tmp_path / "cells.csv"),
            "--effects-out",
            str(tmp_path / "effects.csv"),
            "--allow-legacy-baseline",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Conflicting baseline values" in result.stderr


def test_exporter_requires_identical_observation_coverage(tmp_path):
    rows = []
    for perturbation, observations in [("p1", ["001", "002", "003"]), ("p2", ["001"])]:
        for index, obs_name in enumerate(observations):
            rows.append(
                {
                    "obs_name": obs_name,
                    "latent1": float(index),
                    "latent2": float(index),
                    "perturbation": perturbation,
                    "baseline_core": 0.5,
                    "baseline_repair": 0.5,
                    "candidate_mean_delta_core": -0.1,
                    "candidate_mean_delta_repair": 0.1,
                }
            )
    source = tmp_path / "response.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    script = Path(__file__).resolve().parents[1] / "scripts" / "export_visualizer_inputs.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--response-table",
            str(source),
            "--cells-out",
            str(tmp_path / "cells.csv"),
            "--effects-out",
            str(tmp_path / "effects.csv"),
            "--allow-legacy-baseline",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "identical observation set" in result.stderr


def test_exporter_preserves_leading_zero_observation_ids(tmp_path):
    rows = []
    for perturbation in ("p1", "p2"):
        for index, obs_name in enumerate(("001", "002", "003")):
            rows.append(
                {
                    "obs_name": obs_name,
                    "latent1": float(index),
                    "latent2": float(index),
                    "perturbation": perturbation,
                    "baseline_core": 0.5,
                    "baseline_repair": 0.5,
                    "candidate_mean_delta_core": -0.1,
                    "candidate_mean_delta_repair": 0.1,
                }
            )
    source = tmp_path / "response.csv"
    cells = tmp_path / "cells.csv"
    effects = tmp_path / "effects.csv"
    manifest = tmp_path / "manifest.json"
    pd.DataFrame(rows).to_csv(source, index=False)
    script = Path(__file__).resolve().parents[1] / "scripts" / "export_visualizer_inputs.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--response-table",
            str(source),
            "--cells-out",
            str(cells),
            "--effects-out",
            str(effects),
            "--manifest",
            str(manifest),
            "--allow-legacy-baseline",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert pd.read_csv(cells, dtype={"obs_name": "string"})["obs_name"].tolist() == [
        "001",
        "002",
        "003",
    ]
    coverage = json.loads(manifest.read_text(encoding="utf-8"))["coverage"]
    assert coverage["observations_per_perturbation"] == 3


def test_exporter_rejects_duplicate_perturbation_observation_pairs(tmp_path):
    rows = []
    for obs_name in ("001", "002", "003"):
        rows.append(
            {
                "obs_name": obs_name,
                "latent1": 0.0,
                "latent2": 0.0,
                "perturbation": "p1",
                "baseline_core": 0.5,
                "baseline_repair": 0.5,
                "candidate_mean_delta_core": -0.1,
                "candidate_mean_delta_repair": 0.1,
            }
        )
    rows.append(rows[0].copy())
    source = tmp_path / "response.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    script = Path(__file__).resolve().parents[1] / "scripts" / "export_visualizer_inputs.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--response-table",
            str(source),
            "--cells-out",
            str(tmp_path / "cells.csv"),
            "--effects-out",
            str(tmp_path / "effects.csv"),
            "--allow-legacy-baseline",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "exactly one row per perturbation and observation" in result.stderr


def test_exporter_rejects_empty_table_cleanly(tmp_path):
    source = tmp_path / "empty_response.csv"
    pd.DataFrame(
        columns=[
            "obs_name",
            "latent1",
            "latent2",
            "perturbation",
            "baseline_core",
            "baseline_repair",
            "candidate_mean_delta_core",
            "candidate_mean_delta_repair",
        ]
    ).to_csv(source, index=False)
    script = Path(__file__).resolve().parents[1] / "scripts" / "export_visualizer_inputs.py"
    cells = tmp_path / "cells.csv"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--response-table",
            str(source),
            "--cells-out",
            str(cells),
            "--effects-out",
            str(tmp_path / "effects.csv"),
            "--allow-legacy-baseline",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "at least one data row" in result.stderr
    assert not cells.exists()


def test_exporter_rejects_inconsistent_short_label(tmp_path):
    rows = []
    for index, obs_name in enumerate(("001", "002", "003")):
        rows.append(
            {
                "obs_name": obs_name,
                "latent1": float(index),
                "latent2": float(index),
                "perturbation": "p1",
                "short_label": "Ccl2" if index < 2 else "Ccr2",
                "baseline_core": 0.5,
                "baseline_repair": 0.5,
                "candidate_mean_delta_core": -0.1,
                "candidate_mean_delta_repair": 0.1,
            }
        )
    source = tmp_path / "response.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    script = Path(__file__).resolve().parents[1] / "scripts" / "export_visualizer_inputs.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--response-table",
            str(source),
            "--cells-out",
            str(tmp_path / "cells.csv"),
            "--effects-out",
            str(tmp_path / "effects.csv"),
            "--allow-legacy-baseline",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "short_label" in result.stderr
