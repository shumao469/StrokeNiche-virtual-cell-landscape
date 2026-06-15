#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
65b_auto_find_expr_regulator_and_run_gene_level.py

Purpose
-------
Automatically find expression matrix and external regulator/GMT resources,
standardize paths, and re-run Step65 with gene-level dynamic markers.

This script is designed for the current StrokeNiche pipeline.

It searches local WSL paths for:
  1. Expression data:
       .h5ad
       .csv/.tsv/.txt/.csv.gz/.tsv.gz with obs_name/cell barcode and gene columns
       optionally transposed gene x cell matrices
  2. Regulator-target tables:
       TRRUST / DoRothEA / ChEA / ENCODE / TF-target tables
  3. GMT files:
       MSigDB / Hallmark / Reactome / KEGG / GO / ChEA / DoRothEA .gmt

It then:
  - chooses the best expression candidate by obs_name overlap with Step64c assignments
  - writes a standardized expression CSV if needed
  - merges regulator-target resources into one CSV
  - writes an executable run script
  - optionally runs Step65

Outputs
-------
outdir/
  step65b_expression_candidates.csv
  step65b_regulator_candidates.csv
  step65b_gmt_candidates.csv
  step65b_selected_resources.json
  step65b_standardized_expression_obs_by_gene.csv
  step65b_combined_regulator_targets.csv
  step65b_run_step65_gene_level.sh
  step65b_detection_report.txt
  plus Step65 gene-level outputs if --run is provided
"""

from pathlib import Path
import argparse
import csv
import gzip
import json
import os
import re
import shlex
import subprocess
import warnings

import numpy as np
import pandas as pd


BASE = Path("/mnt/h/vir/ST")

DEFAULT_STEP65_SCRIPT = BASE / "65_track_dynamic_marker_regulator_network.py"

DEFAULT_IN64C_STATE = BASE / "results/step8_strokeniche_perturbmap/dynamics_graph_64c_refined_labels_state_celltype"
DEFAULT_IN64B_STATE = BASE / "results/step8_strokeniche_perturbmap/dynamics_graph_64b_fixed_region_probs_state_celltype"

DEFAULT_IN64C_KMEANS = BASE / "results/step8_strokeniche_perturbmap/dynamics_graph_64c_refined_labels_kmeans"
DEFAULT_IN64B_KMEANS = BASE / "results/step8_strokeniche_perturbmap/dynamics_graph_64b_fixed_region_probs_kmeans"

DEFAULT_OUTROOT = BASE / "results/step8_strokeniche_perturbmap"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def log(msg):
    print(msg, flush=True)


def norm_key(x):
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def gene_symbol(x):
    s = str(x).strip()
    s = re.sub(r"^gene[_\-]?", "", s, flags=re.I)
    return s.upper()


def safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default


def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def split_roots(s):
    roots = []
    for part in str(s).split(","):
        p = Path(part.strip())
        if p.exists():
            roots.append(p)
    return roots


def is_hidden_or_bad_dir(name):
    bad = {
        ".git", ".cache", "__pycache__", "node_modules",
        ".ipynb_checkpoints", "lost+found",
    }
    return name in bad or name.startswith(".")


def file_endswith(path, suffixes):
    s = str(path).lower()
    return any(s.endswith(x.lower()) for x in suffixes)


def open_text_maybe_gz(path):
    path = Path(path)
    if str(path).lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


def guess_sep(path):
    s = str(path).lower()
    if s.endswith(".tsv") or s.endswith(".tsv.gz"):
        return "\t"
    if s.endswith(".txt") or s.endswith(".txt.gz"):
        # Inspect first line.
        try:
            with open_text_maybe_gz(path) as f:
                line = f.readline()
            return "\t" if line.count("\t") >= line.count(",") else ","
        except Exception:
            return "\t"
    return ","


def read_table_head(path, nrows=200):
    sep = guess_sep(path)
    try:
        return pd.read_csv(path, sep=sep, nrows=nrows, low_memory=False)
    except Exception:
        try:
            alt = "\t" if sep == "," else ","
            return pd.read_csv(path, sep=alt, nrows=nrows, low_memory=False)
        except Exception:
            return None


def read_table_full(path, usecols=None):
    sep = guess_sep(path)
    try:
        return pd.read_csv(path, sep=sep, usecols=usecols, low_memory=False)
    except Exception:
        alt = "\t" if sep == "," else ","
        return pd.read_csv(path, sep=alt, usecols=usecols, low_memory=False)


def load_assign_obs(in64c):
    p = Path(in64c) / "step64c_track_assignment_by_cell.refined_labels.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p, low_memory=False)
    if "obs_name" not in df.columns:
        raise RuntimeError(f"{p} lacks obs_name")
    obs = set(df["obs_name"].astype(str))
    return obs, df


def collect_files(roots, max_files=200000):
    expr_suffix = [
        ".h5ad",
        ".csv", ".tsv", ".txt",
        ".csv.gz", ".tsv.gz", ".txt.gz",
    ]
    gmt_suffix = [".gmt", ".gmt.gz"]

    files = []
    n = 0
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not is_hidden_or_bad_dir(d)]
            for fn in filenames:
                p = Path(dirpath) / fn
                low = str(p).lower()
                if file_endswith(low, expr_suffix) or file_endswith(low, gmt_suffix):
                    files.append(p)
                    n += 1
                    if n >= max_files:
                        return files
    return files


# -----------------------------------------------------------------------------
# Expression detection
# -----------------------------------------------------------------------------

EXPR_FILENAME_BONUS_TERMS = [
    "expr", "expression", "count", "counts", "matrix", "rna", "st", "spatial",
    "h5ad", "adata", "anndata", "seurat", "lognorm", "normalized",
]

NON_GENE_PREFIXES = [
    "module_", "neighbor_", "prob_", "core_", "peri_", "remote_", "repair_",
    "delta_", "score_", "rank", "track", "node", "cluster", "celltype",
    "state", "region", "time", "sample", "coord", "latent", "umap",
    "rctd_", "dominant_", "predicted_",
]

COMMON_STROKE_GENES = {
    "SPP1","APOE","CCL2","TGFB1","LGALS9","VEGFA","TREM2","CD44",
    "CCR2","KDR","FLT1","ACKR1","GPX4","SLC7A11","FTH1","TNF",
    "IL1B","GFAP","AQP4","CLDN5","PECAM1","VWF","HIF1A","NFE2L2",
    "HMOX1","NQO1","TYROBP","AIF1","CSF1R","STAT3","RELA","NFKB1",
    "COL1A1","FN1","MMP9","BDNF","SNAP25","MBP","PLP1",
}


def looks_like_gene_col(c):
    s = str(c).strip()
    ns = norm_key(s)
    if not s:
        return False
    if any(ns.startswith(p.rstrip("_")) for p in NON_GENE_PREFIXES):
        return False
    if gene_symbol(s) in COMMON_STROKE_GENES:
        return True
    if re.match(r"^[A-Za-z][A-Za-z0-9\-\.]{1,15}$", s):
        # Avoid obvious metadata.
        bad = {"obs_name", "barcode", "cell_id", "sample", "timepoint", "region", "state", "celltype"}
        return ns not in bad
    return False


def inspect_h5ad(path, obs_set, min_overlap=50):
    try:
        import anndata as ad
    except Exception as e:
        return {
            "path": str(path),
            "kind": "h5ad",
            "status": f"skip_no_anndata:{e}",
            "score": -1,
        }

    try:
        adata = ad.read_h5ad(path, backed="r")
        obs_names = [str(x) for x in adata.obs_names]
        var_names = [str(x) for x in adata.var_names]
        overlap = len(set(obs_names) & obs_set)
        n_obs = len(obs_names)
        n_vars = len(var_names)

        fname = str(path).lower()
        bonus = sum(1 for t in EXPR_FILENAME_BONUS_TERMS if t in fname)
        stroke_gene_hits = len(set(gene_symbol(g) for g in var_names[:min(len(var_names), 10000)]) & COMMON_STROKE_GENES)

        status = "ok" if overlap >= min_overlap and n_vars >= 100 else "low_overlap_or_too_few_genes"
        score = overlap * 1000 + min(n_vars, 30000) + bonus * 500 + stroke_gene_hits * 200
        if status != "ok":
            score -= 1000000

        return {
            "path": str(path),
            "kind": "h5ad",
            "orientation": "cells_by_genes",
            "status": status,
            "n_obs": n_obs,
            "n_genes": n_vars,
            "obs_overlap": overlap,
            "gene_like_cols": n_vars,
            "stroke_gene_hits": stroke_gene_hits,
            "filename_bonus": bonus,
            "score": score,
        }
    except Exception as e:
        return {
            "path": str(path),
            "kind": "h5ad",
            "status": f"error:{type(e).__name__}:{e}",
            "score": -1,
        }


def inspect_csv_expression(path, obs_set, min_overlap=50, min_gene_cols=100):
    head = read_table_head(path, nrows=500)
    if head is None or head.empty:
        return {
            "path": str(path),
            "kind": "table",
            "status": "cannot_read_head",
            "score": -1,
        }

    cols = list(head.columns)
    norm_cols = {norm_key(c): c for c in cols}

    obs_col = None
    for k in ["obs_name", "barcode", "cell_id", "spot_id", "cell", "cell_name", "index"]:
        if k in norm_cols:
            obs_col = norm_cols[k]
            break
    if obs_col is None:
        obs_col = cols[0]

    # rows = cells orientation
    overlap_rows = 0
    try:
        overlap_rows = len(set(head[obs_col].astype(str)) & obs_set)
    except Exception:
        overlap_rows = 0

    numeric_gene_cols = []
    gene_like_cols = []
    for c in cols:
        if c == obs_col:
            continue
        if looks_like_gene_col(c):
            gene_like_cols.append(c)
            x = pd.to_numeric(head[c], errors="coerce")
            if x.notna().mean() >= 0.30 and x.nunique(dropna=True) > 1:
                numeric_gene_cols.append(c)

    stroke_gene_hits_rows = len(set(gene_symbol(c) for c in cols) & COMMON_STROKE_GENES)

    # columns = cells orientation
    overlap_cols = len(set(map(str, cols)) & obs_set)
    first_col_gene_like = False
    try:
        first_vals = head[cols[0]].astype(str).head(200).map(gene_symbol)
        first_col_gene_like = len(set(first_vals) & COMMON_STROKE_GENES) >= 2
    except Exception:
        pass

    fname = str(path).lower()
    bonus = sum(1 for t in EXPR_FILENAME_BONUS_TERMS if t in fname)

    candidates = []

    if overlap_rows >= min_overlap and len(numeric_gene_cols) >= min_gene_cols:
        score = overlap_rows * 1000 + len(numeric_gene_cols) * 10 + stroke_gene_hits_rows * 500 + bonus * 500
        candidates.append({
            "orientation": "rows_cells",
            "obs_col": obs_col,
            "obs_overlap": overlap_rows,
            "n_genes": len(numeric_gene_cols),
            "gene_like_cols": len(gene_like_cols),
            "stroke_gene_hits": stroke_gene_hits_rows,
            "score": score,
            "status": "ok",
        })

    if overlap_cols >= min_overlap and first_col_gene_like:
        # Estimate n genes from rows; actual full count unknown, use large proxy.
        score = overlap_cols * 1000 + 5000 + bonus * 500 + 1000
        candidates.append({
            "orientation": "cols_cells",
            "obs_col": cols[0],
            "obs_overlap": overlap_cols,
            "n_genes": 5000,
            "gene_like_cols": 5000,
            "stroke_gene_hits": len(set(first_vals) & COMMON_STROKE_GENES),
            "score": score,
            "status": "ok",
        })

    if not candidates:
        return {
            "path": str(path),
            "kind": "table",
            "orientation": "",
            "obs_col": obs_col,
            "status": "not_expression_candidate",
            "n_cols": len(cols),
            "obs_overlap_rows_head": overlap_rows,
            "obs_overlap_cols": overlap_cols,
            "n_genes": len(numeric_gene_cols),
            "gene_like_cols": len(gene_like_cols),
            "stroke_gene_hits": stroke_gene_hits_rows,
            "filename_bonus": bonus,
            "score": -1,
        }

    best = sorted(candidates, key=lambda x: x["score"], reverse=True)[0]
    best.update({
        "path": str(path),
        "kind": "table",
        "n_cols": len(cols),
        "filename_bonus": bonus,
    })
    return best


def find_expression_candidates(files, obs_set, min_overlap=50, min_gene_cols=100):
    rows = []
    for p in files:
        low = str(p).lower()
        if low.endswith(".gmt") or low.endswith(".gmt.gz"):
            continue

        # Avoid obvious outputs from our own pipeline unless they look like expression.
        if any(x in low for x in [
            "step65_", "step64_", "track_dynamic", "dynamics_graph",
            "regulator", "enrichment", "neighbor_bridge", "final_bbb",
            "perturbation_ranking", "report",
        ]):
            continue

        if low.endswith(".h5ad"):
            rows.append(inspect_h5ad(p, obs_set, min_overlap=min_overlap))
        elif file_endswith(low, [".csv", ".tsv", ".txt", ".csv.gz", ".tsv.gz", ".txt.gz"]):
            rows.append(inspect_csv_expression(p, obs_set, min_overlap=min_overlap, min_gene_cols=min_gene_cols))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df


def standardize_expr_candidate(candidate, obs_set, out_csv, max_genes=3000):
    path = Path(candidate["path"])
    kind = candidate.get("kind", "")
    orientation = candidate.get("orientation", "")

    if kind == "h5ad":
        # Let Step65 handle h5ad directly. No conversion required.
        return "", str(path), {
            "standardized": False,
            "mode": "h5ad_direct",
            "expr_h5ad": str(path),
        }

    if kind != "table":
        raise RuntimeError(f"Unsupported expression kind: {kind}")

    if orientation == "rows_cells":
        obs_col = candidate.get("obs_col") or "obs_name"
        head = read_table_head(path, nrows=100)
        cols = list(head.columns)

        gene_cols = []
        for c in cols:
            if c == obs_col:
                continue
            if looks_like_gene_col(c):
                gene_cols.append(c)

        # Load candidate columns only.
        usecols = [obs_col] + gene_cols
        df = read_table_full(path, usecols=usecols)
        df = df.rename(columns={obs_col: "obs_name"})
        df["obs_name"] = df["obs_name"].astype(str)

        df = df[df["obs_name"].isin(obs_set)].copy()

        # Keep numeric genes.
        numeric_genes = []
        for c in gene_cols:
            if c not in df.columns:
                continue
            df[c] = pd.to_numeric(df[c], errors="coerce")
            if df[c].notna().mean() >= 0.30 and df[c].nunique(dropna=True) > 1:
                numeric_genes.append(c)

        if len(numeric_genes) > max_genes:
            var = df[numeric_genes].var(axis=0, skipna=True).sort_values(ascending=False)
            numeric_genes = var.head(max_genes).index.tolist()

        out = df[["obs_name"] + numeric_genes].copy()
        out.to_csv(out_csv, index=False)

        return str(out_csv), "", {
            "standardized": True,
            "mode": "rows_cells_csv",
            "source": str(path),
            "obs_col": obs_col,
            "n_obs": int(len(out)),
            "n_genes": int(len(numeric_genes)),
            "expr_csv": str(out_csv),
        }

    if orientation == "cols_cells":
        # Transposed matrix: first column genes, columns cells.
        df = read_table_full(path)
        gene_col = df.columns[0]
        obs_cols = [c for c in df.columns[1:] if str(c) in obs_set]

        if len(obs_cols) == 0:
            raise RuntimeError("Transposed expression matrix has no obs columns overlapping assignments.")

        genes = df[gene_col].astype(str).map(gene_symbol)
        keep_gene_mask = genes.map(lambda g: bool(re.match(r"^[A-Z0-9][A-Z0-9\\-\\.]{1,20}$", g))).values

        df = df.loc[keep_gene_mask, [gene_col] + obs_cols].copy()
        df[gene_col] = df[gene_col].astype(str).map(gene_symbol)

        # Convert to numeric and choose top variable genes.
        mat = df[obs_cols].apply(pd.to_numeric, errors="coerce")
        var = mat.var(axis=1, skipna=True).fillna(0)
        top_idx = var.sort_values(ascending=False).head(max_genes).index

        df_top = df.loc[top_idx, [gene_col] + obs_cols].copy()
        df_top = df_top.drop_duplicates(subset=[gene_col])

        mat_top = df_top.set_index(gene_col)[obs_cols].T
        mat_top.index.name = "obs_name"
        out = mat_top.reset_index()
        out.columns = ["obs_name"] + [gene_symbol(c) for c in out.columns[1:]]
        out.to_csv(out_csv, index=False)

        return str(out_csv), "", {
            "standardized": True,
            "mode": "cols_cells_transposed_csv",
            "source": str(path),
            "n_obs": int(out.shape[0]),
            "n_genes": int(out.shape[1] - 1),
            "expr_csv": str(out_csv),
        }

    raise RuntimeError(f"Unsupported expression orientation: {orientation}")


# -----------------------------------------------------------------------------
# Regulator / GMT detection and merge
# -----------------------------------------------------------------------------

REG_FILENAME_TERMS = [
    "trrust", "dorothea", "chea", "encode", "tf", "transcription",
    "regulator", "regulon", "target", "tftarget", "network",
]

GMT_FILENAME_TERMS = [
    "msigdb", "hallmark", "reactome", "kegg", "gobp", "go_bp",
    "c2.", "c3.", "c5.", "c7.", "chea", "dorothea", "tf",
]


def inspect_regulator_table(path, min_pairs=10):
    head = read_table_head(path, nrows=100)
    if head is None or head.empty:
        # Try no-header TRRUST-like.
        try:
            sep = guess_sep(path)
            tmp = pd.read_csv(path, sep=sep, header=None, nrows=100)
            if tmp.shape[1] >= 2:
                return {
                    "path": str(path),
                    "kind": "regulator_table",
                    "status": "ok_no_header",
                    "reg_col": "__col0__",
                    "target_col": "__col1__",
                    "database": Path(path).stem,
                    "n_head_rows": int(tmp.shape[0]),
                    "score": 500 + tmp.shape[0],
                }
        except Exception:
            pass

        return {
            "path": str(path),
            "kind": "regulator_table",
            "status": "cannot_read",
            "score": -1,
        }

    cols = list(head.columns)
    norm_cols = {norm_key(c): c for c in cols}

    reg_col = None
    for k in ["regulator", "tf", "source", "transcription_factor", "term", "pathway", "geneset", "gene_set"]:
        if k in norm_cols:
            reg_col = norm_cols[k]
            break

    target_col = None
    for k in ["target", "target_gene", "gene", "genesymbol", "gene_symbol", "targets"]:
        if k in norm_cols:
            target_col = norm_cols[k]
            break

    low = str(path).lower()
    bonus = sum(1 for t in REG_FILENAME_TERMS if t in low)

    if reg_col and target_col:
        n_pairs_head = int(head[[reg_col, target_col]].dropna().shape[0])
        status = "ok" if n_pairs_head >= 1 else "empty_pairs"
        score = bonus * 1000 + n_pairs_head * 10
        return {
            "path": str(path),
            "kind": "regulator_table",
            "status": status,
            "reg_col": reg_col,
            "target_col": target_col,
            "database": Path(path).stem,
            "n_head_rows": int(head.shape[0]),
            "n_pairs_head": n_pairs_head,
            "filename_bonus": bonus,
            "score": score,
        }

    # Filename says regulator but headers are unknown.
    if bonus > 0 and head.shape[1] >= 2:
        return {
            "path": str(path),
            "kind": "regulator_table",
            "status": "possible_unknown_header",
            "reg_col": cols[0],
            "target_col": cols[1],
            "database": Path(path).stem,
            "n_head_rows": int(head.shape[0]),
            "filename_bonus": bonus,
            "score": bonus * 500,
        }

    return {
        "path": str(path),
        "kind": "regulator_table",
        "status": "not_regulator_table",
        "filename_bonus": bonus,
        "score": -1,
    }


def inspect_gmt(path):
    low = str(path).lower()
    bonus = sum(1 for t in GMT_FILENAME_TERMS if t in low)

    n_sets = 0
    n_genes = 0
    try:
        with open_text_maybe_gz(path) as f:
            for i, line in enumerate(f):
                if i >= 2000:
                    break
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3:
                    n_sets += 1
                    n_genes += max(0, len(parts) - 2)
    except Exception as e:
        return {
            "path": str(path),
            "kind": "gmt",
            "status": f"error:{e}",
            "score": -1,
        }

    status = "ok" if n_sets > 0 else "empty"
    score = bonus * 1000 + min(n_sets, 10000) + min(n_genes, 200000) / 100

    return {
        "path": str(path),
        "kind": "gmt",
        "status": status,
        "n_sets_head": n_sets,
        "n_genes_head": n_genes,
        "filename_bonus": bonus,
        "score": score,
    }


def find_regulator_and_gmt_candidates(files):
    reg_rows = []
    gmt_rows = []

    for p in files:
        low = str(p).lower()

        # Skip expression-like massive matrices unless filename suggests regulator.
        if low.endswith(".gmt") or low.endswith(".gmt.gz"):
            gmt_rows.append(inspect_gmt(p))
            continue

        if not file_endswith(low, [".csv", ".tsv", ".txt", ".csv.gz", ".tsv.gz", ".txt.gz"]):
            continue

        if any(t in low for t in REG_FILENAME_TERMS):
            reg_rows.append(inspect_regulator_table(p))

    reg_df = pd.DataFrame(reg_rows)
    gmt_df = pd.DataFrame(gmt_rows)

    if not reg_df.empty:
        reg_df = reg_df.sort_values("score", ascending=False).reset_index(drop=True)
    if not gmt_df.empty:
        gmt_df = gmt_df.sort_values("score", ascending=False).reset_index(drop=True)

    return reg_df, gmt_df


def load_regulator_candidate(row):
    path = Path(row["path"])
    status = row.get("status", "")

    if status == "ok_no_header":
        sep = guess_sep(path)
        df = pd.read_csv(path, sep=sep, header=None, low_memory=False)
        reg_col = df.columns[0]
        target_col = df.columns[1]
    else:
        df = read_table_full(path)
        reg_col = row.get("reg_col")
        target_col = row.get("target_col")

    if reg_col not in df.columns or target_col not in df.columns:
        return pd.DataFrame()

    database = str(row.get("database") or Path(path).stem)

    out_rows = []
    for _, r in df[[reg_col, target_col]].dropna().iterrows():
        reg = str(r[reg_col]).strip()
        targets = str(r[target_col]).replace(",", ";").replace("|", ";").split(";")
        for t in targets:
            t = gene_symbol(t)
            if t and t != "NAN":
                out_rows.append({
                    "regulator": reg,
                    "target": t,
                    "database": database,
                    "source_file": str(path),
                })

    return pd.DataFrame(out_rows).drop_duplicates()


def load_gmt_candidate(path, max_sets=20000):
    rows = []
    n = 0
    with open_text_maybe_gz(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            for g in parts[2:]:
                gs = gene_symbol(g)
                if gs and gs != "NAN":
                    rows.append({
                        "regulator": name,
                        "target": gs,
                        "database": "gmt",
                        "source_file": str(path),
                    })
            n += 1
            if n >= max_sets:
                break
    return pd.DataFrame(rows).drop_duplicates()


def build_combined_regulator_db(reg_candidates, gmt_candidates, out_csv, max_reg_tables=5, max_gmt_files=3):
    frames = []

    if reg_candidates is not None and not reg_candidates.empty:
        ok = reg_candidates[reg_candidates["score"] > 0].head(max_reg_tables)
        for _, row in ok.iterrows():
            try:
                df = load_regulator_candidate(row)
                if not df.empty:
                    frames.append(df)
            except Exception as e:
                log(f"[WARN] failed loading regulator table {row['path']}: {e}")

    if gmt_candidates is not None and not gmt_candidates.empty:
        ok = gmt_candidates[gmt_candidates["score"] > 0].head(max_gmt_files)
        for _, row in ok.iterrows():
            try:
                df = load_gmt_candidate(row["path"])
                if not df.empty:
                    frames.append(df)
            except Exception as e:
                log(f"[WARN] failed loading GMT {row['path']}: {e}")

    if not frames:
        return "", {
            "combined": False,
            "reason": "no_external_regulator_or_gmt_loaded",
        }

    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    out = out[out["regulator"].astype(str).ne("")]
    out = out[out["target"].astype(str).ne("")]
    out.to_csv(out_csv, index=False)

    return str(out_csv), {
        "combined": True,
        "path": str(out_csv),
        "n_pairs": int(len(out)),
        "n_regulators": int(out["regulator"].nunique()),
        "n_targets": int(out["target"].nunique()),
        "databases": out["database"].value_counts().head(30).to_dict(),
        "source_files": out["source_file"].drop_duplicates().head(20).tolist(),
    }


# -----------------------------------------------------------------------------
# Run Step65
# -----------------------------------------------------------------------------

def shell_quote(x):
    return shlex.quote(str(x))


def build_step65_command(args, expr_csv, expr_h5ad, regulator_csv, outdir):
    cmd = [
        args.python,
        str(args.step65_script),
        "--in64c", str(args.in64c),
        "--in64b", str(args.in64b),
        "--outdir", str(outdir),
        "--max_genes", str(args.max_genes),
        "--min_cells_per_timepoint", str(args.min_cells_per_timepoint),
        "--min_abs_delta_module", str(args.min_abs_delta_module),
        "--min_abs_delta_gene", str(args.min_abs_delta_gene),
        "--top_n_modules_for_enrichment", str(args.top_n_modules_for_enrichment),
        "--top_n_genes_for_enrichment", str(args.top_n_genes_for_enrichment),
        "--min_overlap", str(args.min_overlap_enrichment),
        "--top_tracks_fig", str(args.top_tracks_fig),
        "--top_modules_fig", str(args.top_modules_fig),
        "--dpi", str(args.dpi),
    ]

    if expr_csv:
        cmd.extend(["--expr_csv", str(expr_csv)])
    if expr_h5ad:
        cmd.extend(["--expr_h5ad", str(expr_h5ad)])
    if regulator_csv:
        cmd.extend(["--regulator_table", str(regulator_csv)])

    return cmd


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--variant", choices=["state_celltype", "kmeans"], default="state_celltype")

    ap.add_argument("--in64c", default="")
    ap.add_argument("--in64b", default="")
    ap.add_argument("--step65_script", default=str(DEFAULT_STEP65_SCRIPT))

    ap.add_argument("--search_roots", default="/mnt/h/vir/ST,/mnt/h/Data,/mnt/h/vir")
    ap.add_argument("--outdir", default="")
    ap.add_argument("--python", default="/home/shu/miniconda/envs/nicheformer_env/bin/python")

    ap.add_argument("--min_obs_overlap", type=int, default=50)
    ap.add_argument("--min_gene_cols", type=int, default=100)
    ap.add_argument("--max_genes", type=int, default=3000)
    ap.add_argument("--max_files", type=int, default=200000)

    ap.add_argument("--max_reg_tables", type=int, default=5)
    ap.add_argument("--max_gmt_files", type=int, default=3)

    ap.add_argument("--min_cells_per_timepoint", type=int, default=10)
    ap.add_argument("--min_abs_delta_module", type=float, default=0.03)
    ap.add_argument("--min_abs_delta_gene", type=float, default=0.10)
    ap.add_argument("--top_n_modules_for_enrichment", type=int, default=8)
    ap.add_argument("--top_n_genes_for_enrichment", type=int, default=50)
    ap.add_argument("--min_overlap_enrichment", type=int, default=1)
    ap.add_argument("--top_tracks_fig", type=int, default=12)
    ap.add_argument("--top_modules_fig", type=int, default=12)
    ap.add_argument("--dpi", type=int, default=600)

    ap.add_argument("--allow_module_only_if_no_expr", action="store_true")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()

    if not args.in64c:
        args.in64c = str(DEFAULT_IN64C_STATE if args.variant == "state_celltype" else DEFAULT_IN64C_KMEANS)
    if not args.in64b:
        args.in64b = str(DEFAULT_IN64B_STATE if args.variant == "state_celltype" else DEFAULT_IN64B_KMEANS)

    if not args.outdir:
        args.outdir = str(DEFAULT_OUTROOT / f"track_dynamic_marker_regulator_65b_{args.variant}_gene_auto")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log("=" * 100)
    log("Step 65b auto-detect expression/regulator resources and run gene-level Step65")
    log("=" * 100)
    log(f"variant={args.variant}")
    log(f"in64c={args.in64c}")
    log(f"in64b={args.in64b}")
    log(f"outdir={outdir}")

    obs_set, assign = load_assign_obs(args.in64c)
    roots = split_roots(args.search_roots)

    log(f"search_roots={roots}")
    log(f"assignment obs_name n={len(obs_set)}")

    files = collect_files(roots, max_files=args.max_files)
    log(f"candidate files collected={len(files)}")

    expr_candidates = find_expression_candidates(
        files,
        obs_set=obs_set,
        min_overlap=args.min_obs_overlap,
        min_gene_cols=args.min_gene_cols,
    )
    expr_candidates.to_csv(outdir / "step65b_expression_candidates.csv", index=False)

    reg_candidates, gmt_candidates = find_regulator_and_gmt_candidates(files)
    reg_candidates.to_csv(outdir / "step65b_regulator_candidates.csv", index=False)
    gmt_candidates.to_csv(outdir / "step65b_gmt_candidates.csv", index=False)

    selected_expr = None
    expr_csv = ""
    expr_h5ad = ""
    expr_meta = {}

    ok_expr = expr_candidates[expr_candidates["score"] > 0].copy() if not expr_candidates.empty else pd.DataFrame()

    if not ok_expr.empty:
        selected_expr = ok_expr.iloc[0].to_dict()
        log(f"selected expression candidate: {selected_expr['path']}")
        expr_csv, expr_h5ad, expr_meta = standardize_expr_candidate(
            selected_expr,
            obs_set=obs_set,
            out_csv=outdir / "step65b_standardized_expression_obs_by_gene.csv",
            max_genes=args.max_genes,
        )
    else:
        msg = "No suitable expression matrix found."
        log("[WARN] " + msg)
        expr_meta = {"standardized": False, "mode": "no_expression_found", "warning": msg}
        if not args.allow_module_only_if_no_expr:
            log("[STOP] Use --allow_module_only_if_no_expr to run without expression.")
            # Still write reports before exiting.

    regulator_csv, regulator_meta = build_combined_regulator_db(
        reg_candidates=reg_candidates,
        gmt_candidates=gmt_candidates,
        out_csv=outdir / "step65b_combined_regulator_targets.csv",
        max_reg_tables=args.max_reg_tables,
        max_gmt_files=args.max_gmt_files,
    )

    selected = {
        "variant": args.variant,
        "in64c": str(args.in64c),
        "in64b": str(args.in64b),
        "search_roots": [str(x) for x in roots],
        "n_candidate_files": int(len(files)),
        "selected_expression": selected_expr,
        "expression_meta": expr_meta,
        "expr_csv_for_step65": expr_csv,
        "expr_h5ad_for_step65": expr_h5ad,
        "regulator_meta": regulator_meta,
        "regulator_csv_for_step65": regulator_csv,
        "top_expression_candidates": ok_expr.head(20).to_dict(orient="records") if not ok_expr.empty else [],
        "top_regulator_candidates": reg_candidates.head(20).to_dict(orient="records") if not reg_candidates.empty else [],
        "top_gmt_candidates": gmt_candidates.head(20).to_dict(orient="records") if not gmt_candidates.empty else [],
    }

    (outdir / "step65b_selected_resources.json").write_text(
        json.dumps(selected, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    should_build_run = bool(expr_csv or expr_h5ad or args.allow_module_only_if_no_expr)

    cmd = []
    if should_build_run:
        cmd = build_step65_command(
            args=args,
            expr_csv=expr_csv,
            expr_h5ad=expr_h5ad,
            regulator_csv=regulator_csv,
            outdir=outdir,
        )

        run_sh = outdir / "step65b_run_step65_gene_level.sh"
        with open(run_sh, "w", encoding="utf-8") as f:
            f.write("#!/usr/bin/env bash\n")
            f.write("set -euo pipefail\n\n")
            f.write("cd /mnt/h/vir/ST\n\n")
            f.write(" ".join(shell_quote(x) for x in cmd))
            f.write(f" 2>&1 | tee {shell_quote(outdir / 'run_65b_gene_level_step65.log')}\n")
        try:
            os.chmod(run_sh, 0o755)
        except Exception:
            pass
    else:
        run_sh = ""

    lines = []
    lines.append("Step65b auto-detection report")
    lines.append("=" * 100)
    lines.append(json.dumps(selected, indent=2, ensure_ascii=False, default=str))
    lines.append("")
    lines.append("Top expression candidates:")
    lines.append(expr_candidates.head(30).to_string(index=False) if not expr_candidates.empty else "None")
    lines.append("")
    lines.append("Top regulator table candidates:")
    lines.append(reg_candidates.head(30).to_string(index=False) if not reg_candidates.empty else "None")
    lines.append("")
    lines.append("Top GMT candidates:")
    lines.append(gmt_candidates.head(30).to_string(index=False) if not gmt_candidates.empty else "None")
    lines.append("")
    lines.append("Step65 command:")
    lines.append(" ".join(shell_quote(x) for x in cmd) if cmd else "No command built because no expression matrix was found.")
    lines.append("")
    lines.append(f"Run script: {run_sh}")

    (outdir / "step65b_detection_report.txt").write_text("\n".join(lines), encoding="utf-8")

    log("=" * 100)
    log("DETECTION DONE")
    log("=" * 100)
    log("\n".join(lines[:80]))

    if args.run and should_build_run:
        log("=" * 100)
        log("RUNNING Step65 gene-level analysis")
        log("=" * 100)
        subprocess.run(cmd, check=True)
        log("=" * 100)
        log("DONE Step65b + Step65 gene-level run")
        log("=" * 100)
    elif args.run and not should_build_run:
        raise RuntimeError("Requested --run but no expression matrix found and module-only fallback is not allowed.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
