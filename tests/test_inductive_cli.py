import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_inductive_cli_rebuilds_safe_grouped_npz(tmp_path):
    n = 12
    obs = np.array([f"cell_{i}" for i in range(n)], dtype=object)
    input_npz = tmp_path / "legacy.npz"
    np.savez_compressed(
        input_npz,
        obs_names=obs,
        coords=np.column_stack([np.arange(n), np.arange(n) % 2]).astype(float),
        edge_index=np.vstack([np.arange(n - 1), np.arange(1, n)]),
        region_raw=np.array(["region"] * n, dtype=object),
    )
    group_table = tmp_path / "groups.csv"
    pd.DataFrame(
        {
            "obs_name": obs.astype(str),
            "animal": [f"animal_{i // 2}" for i in range(n)],
            "section": [f"animal_{i // 2}_section" for i in range(n)],
        }
    ).to_csv(group_table, index=False)
    output = tmp_path / "rebuilt.npz"
    report = tmp_path / "report.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "make_inductive_split.py"
    result = subprocess.run(
        [
            sys.executable, str(script),
            "--input-npz", str(input_npz),
            "--group-table", str(group_table),
            "--split-group-col", "animal",
            "--graph-group-col", "section",
            "--output-npz", str(output),
            "--report", str(report),
            "--knn-k", "1",
            "--trust-pickle-input",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    with np.load(output, allow_pickle=False) as rebuilt:
        # Every output key must be accessible without enabling pickle.
        for key in rebuilt.files:
            _ = rebuilt[key]
        assert rebuilt["obs_names"].dtype.kind == "U"
        assert rebuilt["region_raw"].dtype.kind == "U"
        labels = np.select(
            [rebuilt["train_mask"], rebuilt["val_mask"], rebuilt["test_mask"]],
            [0, 1, 2],
            default=-1,
        )
        edges = rebuilt["edge_index"]
        assert np.all(labels[edges[0]] == labels[edges[1]])
        assert np.all(rebuilt["graph_group"][edges[0]] == rebuilt["graph_group"][edges[1]])
    report_data = json.loads(report.read_text(encoding="utf-8"))
    assert report_data["output_npz_sha256"] == file_sha256(output)
    assert report_data["input_npz_file"] == input_npz.name
    assert report_data["output_npz_file"] == output.name
    assert str(tmp_path) not in report.read_text(encoding="utf-8")
    assert set(report_data["object_arrays_converted_to_unicode"]) == {"obs_names", "region_raw"}


def test_inductive_cli_rejects_input_output_collision(tmp_path):
    input_npz = tmp_path / "same.npz"
    np.savez_compressed(input_npz, obs_names=np.array(["a", "b", "c"]), coords=np.eye(3, 2))
    groups = tmp_path / "groups.csv"
    pd.DataFrame(
        {"obs_name": ["a", "b", "c"], "animal": ["a", "b", "c"], "section": ["a", "b", "c"]}
    ).to_csv(groups, index=False)
    script = Path(__file__).resolve().parents[1] / "scripts" / "make_inductive_split.py"
    result = subprocess.run(
        [
            sys.executable, str(script),
            "--input-npz", str(input_npz),
            "--group-table", str(groups),
            "--split-group-col", "animal",
            "--graph-group-col", "section",
            "--output-npz", str(input_npz),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "must be distinct" in result.stderr


def test_inductive_cli_accepts_same_split_and_graph_group_column(tmp_path):
    n = 12
    obs = np.array([f"cell_{i}" for i in range(n)])
    sections = np.array([f"section_{i // 2}" for i in range(n)])
    input_npz = tmp_path / "legacy.npz"
    np.savez_compressed(
        input_npz,
        obs_names=obs,
        coords=np.column_stack([np.arange(n), np.arange(n) % 2]).astype(float),
    )
    group_table = tmp_path / "groups.csv"
    pd.DataFrame({"obs_name": obs, "section": sections}).to_csv(group_table, index=False)
    output = tmp_path / "rebuilt.npz"
    script = Path(__file__).resolve().parents[1] / "scripts" / "make_inductive_split.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input-npz",
            str(input_npz),
            "--group-table",
            str(group_table),
            "--split-group-col",
            "section",
            "--graph-group-col",
            "section",
            "--output-npz",
            str(output),
            "--knn-k",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    with np.load(output, allow_pickle=False) as rebuilt:
        assert np.array_equal(rebuilt["split_group"], rebuilt["graph_group"])
        edges = rebuilt["edge_index"]
        assert np.all(rebuilt["graph_group"][edges[0]] == rebuilt["graph_group"][edges[1]])


def test_inductive_cli_rejects_output_without_npz_suffix(tmp_path):
    input_npz = tmp_path / "legacy.npz"
    np.savez_compressed(
        input_npz,
        obs_names=np.array(["a", "b", "c"]),
        coords=np.eye(3, 2),
    )
    groups = tmp_path / "groups.csv"
    pd.DataFrame(
        {"obs_name": ["a", "b", "c"], "section": ["a", "b", "c"]}
    ).to_csv(groups, index=False)
    script = Path(__file__).resolve().parents[1] / "scripts" / "make_inductive_split.py"
    output_without_suffix = tmp_path / "rebuilt"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input-npz",
            str(input_npz),
            "--group-table",
            str(groups),
            "--split-group-col",
            "section",
            "--graph-group-col",
            "section",
            "--output-npz",
            str(output_without_suffix),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "must end with .npz" in result.stderr
    assert not output_without_suffix.exists()
    assert not output_without_suffix.with_suffix(".npz").exists()


def test_inductive_cli_rejects_structured_dtype_containing_objects(tmp_path):
    obs = np.array(["a", "b", "c"])
    structured = np.empty(3, dtype=[("label", object)])
    structured["label"] = ["x", "y", "z"]
    input_npz = tmp_path / "legacy_structured.npz"
    np.savez_compressed(
        input_npz,
        obs_names=obs,
        coords=np.eye(3, 2),
        structured_metadata=structured,
    )
    groups = tmp_path / "groups.csv"
    pd.DataFrame(
        {"obs_name": obs, "section": ["a", "b", "c"]}
    ).to_csv(groups, index=False)
    output = tmp_path / "rebuilt.npz"
    script = Path(__file__).resolve().parents[1] / "scripts" / "make_inductive_split.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input-npz",
            str(input_npz),
            "--group-table",
            str(groups),
            "--split-group-col",
            "section",
            "--graph-group-col",
            "section",
            "--output-npz",
            str(output),
            "--trust-pickle-input",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "object-containing structured dtype" in result.stderr
    assert not output.exists()


def test_inductive_cli_rejects_malformed_legacy_edges_before_writing(tmp_path):
    obs = np.array([f"cell_{i}" for i in range(12)])
    input_npz = tmp_path / "legacy_bad_edges.npz"
    np.savez_compressed(
        input_npz,
        obs_names=obs,
        coords=np.column_stack([np.arange(12), np.arange(12) % 2]).astype(float),
        edge_index=np.array([0, 1, 2]),
    )
    groups = tmp_path / "groups.csv"
    pd.DataFrame(
        {"obs_name": obs, "section": [f"section_{i // 2}" for i in range(12)]}
    ).to_csv(groups, index=False)
    output = tmp_path / "rebuilt.npz"
    report = tmp_path / "report.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "make_inductive_split.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input-npz",
            str(input_npz),
            "--group-table",
            str(groups),
            "--split-group-col",
            "section",
            "--graph-group-col",
            "section",
            "--output-npz",
            str(output),
            "--report",
            str(report),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "shape (2, n_edges)" in result.stderr
    assert not output.exists()
    assert not report.exists()
