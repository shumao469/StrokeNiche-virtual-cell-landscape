#!/usr/bin/env python3
"""Create group-isolated masks and remove graph edges that cross splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from strokeniche_vc.splitting import (  # noqa: E402
    assert_group_isolation,
    build_group_isolated_knn,
    group_split_masks,
    scale_coordinates_train_only,
    split_labels,
)
from strokeniche_vc import __version__  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_object_arrays(arrays: dict[str, np.ndarray]) -> list[str]:
    """Convert string-like object arrays to fixed Unicode and reject all others."""

    converted: list[str] = []
    for key, value in list(arrays.items()):
        array = np.asarray(value)
        if not array.dtype.hasobject:
            continue
        if array.dtype != np.dtype("O"):
            raise ValueError(
                f"Unsafe object-containing structured dtype for array {key!r}: "
                f"{array.dtype}; convert it explicitly before rebuilding the NPZ"
            )
        normalized = []
        for item in array.ravel():
            if isinstance(item, (bytes, np.bytes_)):
                normalized.append(bytes(item).decode("utf-8"))
            elif isinstance(item, (str, np.str_)):
                normalized.append(str(item))
            else:
                raise ValueError(
                    f"Unsafe object-dtype array {key!r} contains non-string values; "
                    "convert it explicitly before rebuilding the NPZ"
                )
        arrays[key] = np.asarray(normalized, dtype=str).reshape(array.shape)
        converted.append(key)
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace spot-random masks with grouped masks, train-only scaling, and a within-group graph."
    )
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--group-table", required=True)
    parser.add_argument("--id-col", default="obs_name")
    parser.add_argument(
        "--split-group-col",
        required=True,
        help="Independent biological source used for train/validation/test assignment (for example animal)",
    )
    parser.add_argument(
        "--graph-group-col",
        required=True,
        help="Physical coordinate frame used for kNN construction (for example tissue section)",
    )
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--knn-k", type=int, default=12)
    parser.add_argument(
        "--trust-pickle-input",
        action="store_true",
        help="Allow legacy object arrays only when the NPZ is from a trusted source.",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing output and report files")
    args = parser.parse_args()

    input_path = Path(args.input_npz).resolve()
    group_path = Path(args.group_table).resolve()
    output = Path(args.output_npz).resolve()
    if output.suffix.lower() != ".npz":
        raise ValueError("output NPZ path must end with .npz")
    report_path = (
        Path(args.report).resolve()
        if args.report
        else output.with_suffix(".split_report.json").resolve()
    )
    if output in {input_path, group_path} or report_path in {input_path, group_path, output}:
        raise ValueError("Input, group table, output NPZ, and report paths must be distinct")
    existing = [path for path in (output, report_path) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Outputs already exist; use --force to replace them: {existing}")

    try:
        with np.load(input_path, allow_pickle=args.trust_pickle_input) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
    except ValueError as exc:
        if "Object arrays cannot be loaded" in str(exc):
            raise ValueError(
                "The NPZ contains pickled object arrays. Re-run with "
                "--trust-pickle-input only after verifying the local artifact."
            ) from exc
        raise
    sanitized_object_keys = sanitize_object_arrays(arrays)
    if "obs_names" not in arrays or "coords" not in arrays:
        raise ValueError("input NPZ must contain obs_names and raw coords")

    # Group identifiers are semantic strings; never infer numeric IDs and drop
    # meaningful leading zeroes.
    groups = pd.read_csv(group_path, dtype="string")
    # A biological sample can also be the physical coordinate frame.  Keep the
    # selected column names unique so that this valid same-column case does not
    # create a two-dimensional duplicate-column selection in pandas.
    group_columns = tuple(
        dict.fromkeys((args.id_col, args.split_group_col, args.graph_group_col))
    )
    missing = [c for c in group_columns if c not in groups.columns]
    if missing:
        raise ValueError(f"group table is missing columns: {missing}")
    if groups[list(group_columns)].isna().any(axis=None):
        raise ValueError("group-table IDs and group labels must not be missing")
    for column in group_columns:
        groups[column] = groups[column].astype(str).str.strip()
    if any(groups[column].eq("").any() for column in group_columns):
        raise ValueError("group-table IDs and group labels must not be blank")
    for column in dict.fromkeys((args.split_group_col, args.graph_group_col)):
        conflicts = groups.groupby(args.id_col)[column].nunique()
        conflicts = conflicts[conflicts > 1].index.tolist()
        if conflicts:
            raise ValueError(f"IDs have conflicting {column} assignments: {conflicts[:10]}")
    assignments = groups[list(group_columns)].drop_duplicates(args.id_col)
    if args.graph_group_col != args.split_group_col:
        nesting = assignments.groupby(args.graph_group_col)[args.split_group_col].nunique()
        non_nested = nesting[nesting > 1].index.tolist()
        if non_nested:
            raise ValueError(
                "Each graph group must be nested in one split group; conflicting graph groups: "
                f"{non_nested[:10]}"
            )
    split_map = assignments.set_index(args.id_col)[args.split_group_col]
    graph_map = assignments.set_index(args.id_col)[args.graph_group_col]
    obs_raw = pd.Series(arrays["obs_names"], name=args.id_col)
    if obs_raw.isna().any():
        raise ValueError("NPZ observation IDs must not be missing")
    obs = obs_raw.astype(str).str.strip()
    if obs.eq("").any() or obs.duplicated().any():
        raise ValueError("NPZ observation IDs must be non-blank and unique")
    split_values = obs.map(split_map).astype("string")
    graph_values = obs.map(graph_map).astype("string")
    missing_rows = split_values.isna() | graph_values.isna()
    if missing_rows.any():
        examples = obs[missing_rows].head(10).tolist()
        raise ValueError(f"Missing group assignment for {missing_rows.sum()} observations; examples: {examples}")

    train, val, test = group_split_masks(
        split_values.to_numpy(str),
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed,
    )
    labels = split_labels(train, val, test)
    assert_group_isolation(split_values.to_numpy(str), labels)
    original_edges = np.asarray(arrays.get("edge_index", np.empty((2, 0), dtype=np.int64)))
    if original_edges.size == 0:
        edges_before = 0
    elif original_edges.ndim == 2 and original_edges.shape[0] == 2:
        edges_before = int(original_edges.shape[1])
    else:
        raise ValueError("legacy edge_index must be empty or have shape (2, n_edges)")
    scaled_coords, scaler_mean, scaler_scale = scale_coordinates_train_only(
        arrays["coords"], train
    )
    isolated_edges = build_group_isolated_knn(
        scaled_coords,
        graph_values.to_numpy(str),
        k=args.knn_k,
    )
    if isolated_edges.size and np.any(labels[isolated_edges[0]] != labels[isolated_edges[1]]):
        raise AssertionError("rebuilt graph contains a cross-split edge")

    arrays.update(
        train_mask=train,
        val_mask=val,
        test_mask=test,
        coords_scaled=scaled_coords,
        edge_index=isolated_edges,
        split_group=split_values.to_numpy(str),
        graph_group=graph_values.to_numpy(str),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)

    report = {
        "schema_version": 1,
        "strokeniche_vc_version": __version__,
        # Keep the audit report portable and safe to share: hashes identify the
        # artifacts without disclosing local usernames or study directories.
        "input_npz_file": input_path.name,
        "output_npz_file": output.name,
        "input_npz_sha256": sha256(input_path),
        "group_table_sha256": sha256(group_path),
        "output_npz_sha256": sha256(output),
        "split_group_col": args.split_group_col,
        "graph_group_col": args.graph_group_col,
        "n_split_groups": int(split_values.nunique()),
        "n_graph_groups": int(graph_values.nunique()),
        "n_train": int(train.sum()),
        "n_val": int(val.sum()),
        "n_test": int(test.sum()),
        "split_group_counts": {
            "train": int(np.unique(split_values.to_numpy(str)[train]).size),
            "validation": int(np.unique(split_values.to_numpy(str)[val]).size),
            "test": int(np.unique(split_values.to_numpy(str)[test]).size),
        },
        "edges_before": edges_before,
        "edges_after_group_isolated_rebuild": int(isolated_edges.shape[1]),
        "graph_mode": "knn_rebuilt_within_group_only",
        "knn_k": int(args.knn_k),
        "random_seed": int(args.seed),
        "coordinate_scaler_fit": "train_rows_only",
        "coordinate_scaler_mean": scaler_mean.tolist(),
        "coordinate_scaler_scale": scaler_scale.tolist(),
        "object_arrays_converted_to_unicode": sanitized_object_keys,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
