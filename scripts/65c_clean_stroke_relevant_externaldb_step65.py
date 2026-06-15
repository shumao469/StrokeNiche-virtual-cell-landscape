#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
65c_clean_stroke_relevant_externaldb_step65.py

Purpose
-------
Step 65c: clean gene-level dynamic markers for StrokeNiche tracks.

This script does NOT overwrite Step65b. It creates a new output directory.

It performs:
  1. Read spatial_all.h5ad or another h5ad.
  2. Keep only stroke-relevant genes.
  3. Remove mitochondrial, ribosomal and hemoglobin genes.
  4. Optionally remove predicted/low-confidence genes.
  5. Export filtered obs_name x gene CSV.
  6. Search and integrate true external resources:
       TRRUST / DoRothEA / ChEA / ENCODE / MSigDB GMT
  7. Re-run Step65 with filtered gene CSV and external regulator table.
  8. Save all audit files separately.

Default input:
  h5ad:
    /mnt/h/vir/ST/results/virtual_cell_h5ad/spatial_all.h5ad

Default Step64c:
  /mnt/h/vir/ST/results/step8_strokeniche_perturbmap/dynamics_graph_64c_refined_labels_state_celltype

Default Step64b:
  /mnt/h/vir/ST/results/step8_strokeniche_perturbmap/dynamics_graph_64b_fixed_region_probs_state_celltype

Default output:
  /mnt/h/vir/ST/results/step8_strokeniche_perturbmap/track_dynamic_marker_regulator_65c_state_celltype_gene_clean_stroke_externaldb

Outputs
-------
outdir/
  step65c_filtered_stroke_relevant_expression.csv
  step65c_gene_filter_audit.csv
  step65c_genes_kept.csv
  step65c_genes_removed_blacklist.csv
  step65c_external_regulator_candidates.csv
  step65c_external_gmt_candidates.csv
  step65c_external_regulator_targets.csv
  step65c_selected_resources.json
  step65c_run_step65_clean_gene_externaldb.sh
  step65c_detection_report.txt

If --run is used, also Step65 outputs:
  step65_track_dynamic_genes.csv
  step65_regulator_enrichment.csv
  step65_track_regulator_summary.csv
  Fig_StrokeNiche_TrackDynamicMarkerRegulator_annotated.*
  Fig_StrokeNiche_TrackDynamicMarkerRegulator_clean_no_text.*
"""

from pathlib import Path
import argparse
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

DEFAULT_H5AD = BASE / "results/virtual_cell_h5ad/spatial_all.h5ad"
DEFAULT_STEP65 = BASE / "65_track_dynamic_marker_regulator_network.py"

DEFAULT_IN64C = BASE / "results/step8_strokeniche_perturbmap/dynamics_graph_64c_refined_labels_state_celltype"
DEFAULT_IN64B = BASE / "results/step8_strokeniche_perturbmap/dynamics_graph_64b_fixed_region_probs_state_celltype"

DEFAULT_OUT = BASE / "results/step8_strokeniche_perturbmap/track_dynamic_marker_regulator_65c_state_celltype_gene_clean_stroke_externaldb"


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def log(x):
    print(x, flush=True)


def norm_key(x):
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def gene_upper(x):
    return str(x).strip().upper()


def shell_quote(x):
    return shlex.quote(str(x))


def open_text_maybe_gz(path):
    p = Path(path)
    if str(p).lower().endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8", errors="ignore")
    return open(p, "r", encoding="utf-8", errors="ignore")


def guess_sep(path):
    s = str(path).lower()
    if s.endswith(".tsv") or s.endswith(".tsv.gz"):
        return "\t"
    if s.endswith(".csv") or s.endswith(".csv.gz"):
        return ","
    try:
        with open_text_maybe_gz(path) as f:
            line = f.readline()
        return "\t" if line.count("\t") >= line.count(",") else ","
    except Exception:
        return "\t"


def read_table(path, nrows=None):
    sep = guess_sep(path)
    try:
        return pd.read_csv(path, sep=sep, nrows=nrows, low_memory=False)
    except Exception:
        alt = "," if sep == "\t" else "\t"
        return pd.read_csv(path, sep=alt, nrows=nrows, low_memory=False)


def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def load_assignment_obs(in64c):
    p = Path(in64c) / "step64c_track_assignment_by_cell.refined_labels.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p, low_memory=False)
    if "obs_name" not in df.columns:
        raise RuntimeError(f"{p} lacks obs_name")
    return set(df["obs_name"].astype(str)), df


# -----------------------------------------------------------------------------
# Stroke-relevant genes and blacklist
# -----------------------------------------------------------------------------

def build_stroke_gene_catalog():
    catalog = {
        "core_inflammation_chemokine": [
            "Tnf","Il1b","Il1a","Il6","Il10","Il18","Nfkb1","Nfkbia","Rela",
            "Ccl2","Ccl3","Ccl4","Ccl5","Ccl7","Ccl8","Cxcl1","Cxcl2","Cxcl10",
            "Ccr2","Ccr5","Cxcr2","Cxcr4","Ackr1","Ptgs2","Nos2","Socs3",
            "Tlr2","Tlr4","Myd88","Nlrp3","Casp1","Pycard","Aif1","Lyz2",
        ],
        "microglia_macrophage_phagocytosis": [
            "Trem2","Apoe","Tyrobp","Csf1r","C1qa","C1qb","C1qc","C3","C4b",
            "Itgam","Cd68","Lpl","Ctsb","Ctsd","Ctss","Ctsz","Hexb","Grn",
            "Gpnmb","Lgals3","Laptm5","Mpeg1","Fcer1g","Fcgr3","Cd74","H2-Aa",
            "H2-Ab1","H2-Eb1","Spp1","Cd44",
        ],
        "astrocyte_reactive_gliosis": [
            "Gfap","Vim","Aqp4","Slc1a2","Slc1a3","Aldh1l1","Sox9","Stat3",
            "Lcn2","Serpina3n","Clu","C3","Timp1","Emp1","Gja1","Gjb6","Ednrb",
            "S100b","Fabp7","Mt1","Mt2",
        ],
        "endothelial_bbb_vascular": [
            "Pecam1","Cldn5","Ocln","Tjp1","Cdh5","Vwf","Kdr","Flt1","Tek",
            "Eng","Esam","Icam1","Vcam1","Sele","Selp","Nos3","Klf2","Klf4",
            "Abcb1a","Abcg2","Slco1a4","Mfsd2a","Plvap","Angpt1","Angpt2",
            "Vegfa","Pgf","Epas1","Rgs5","Pdgfrb","Acta2","Tagln","Des","Mcam",
        ],
        "hypoxia_metabolism_mitochondrial_stress": [
            "Hif1a","Epas1","Vegfa","Slc2a1","Slc16a1","Slc16a3","Ldha","Ldhb",
            "Pdk1","Pgk1","Eno1","Gapdh","Aldoa","Bnip3","Bnip3l","Adm","Eglin1",
            "Eglin3","Car9","Hk2","Pkm","Mif",
        ],
        "ferroptosis_redox": [
            "Gpx4","Slc7a11","Fth1","Ftl1","Ftl","Nfe2l2","Keap1","Hmox1","Nqo1",
            "Gclc","Gclm","Txnrd1","Prdx1","Prdx2","Prdx6","Sod1","Sod2","Cat",
            "Alox5","Alox15","Acsl4","Lpcat3","Sat1","Fsp1","Aifm2","Gss","Gsr",
        ],
        "ecm_remodeling_repair": [
            "Tgfb1","Tgfb2","Tgfbr1","Tgfbr2","Smad2","Smad3","Smad4","Serpine1",
            "Ctgf","Fn1","Col1a1","Col1a2","Col3a1","Col4a1","Col4a2","Col6a1",
            "Mmp2","Mmp3","Mmp9","Mmp14","Timp1","Timp2","Thbs1","Thbs2",
            "Postn","Dcn","Lum","Sparc","Spp1","Cd44","Itgav","Itgb1","Itgb3",
        ],
        "neuronal_synaptic_recovery": [
            "Rbfox3","Map2","Tubb3","Snap25","Syt1","Syt11","Syn1","Dlg4","Grin1",
            "Gria1","Gria2","Camk2a","Camk2b","Bdnf","Ntrk2","Gap43","Dcx",
            "Nefl","Nefm","Nefh","Syp","Stmn1","Stmn2","Sema3a","Nrg1","Reln",
        ],
        "oligodendrocyte_myelin": [
            "Mbp","Plp1","Mog","Mag","Mobp","Cnp","Olig1","Olig2","Sox10","Pdgfra",
            "Cspg4","Bcan","Enpp6","Mal","Myrf","Cldn11","Tspan2","Apod","Trf",
        ],
        "cell_death_stress_heatshock": [
            "Bax","Bcl2","Bcl2l1","Casp3","Casp8","Casp9","Apaf1","Fas","Fasl",
            "Atf3","Ddit3","Hspa1a","Hspa1b","Hsp90aa1","Hspb1","Dusp1","Egr1",
            "Jun","Jund","Fos","Fosb","Ier2","Ier3",
        ],
        "complement_coagulation_platelet": [
            "C1qa","C1qb","C1qc","C3","C4b","Cfb","Cfd","Serping1","Fga","Fgb",
            "Fgg","F2r","F3","Plat","Plau","Serpine1","Itga2b","Gp1ba","Ppbp",
        ],
        "candidate_axes_from_perturbmap": [
            "Spp1","Apoe","Ccl2","Tgfb1","Lgals9","Vegfa","Trem2","Cd44",
            "Ccr2","Kdr","Flt1","Ackr1","Gpx4","Slc7a11","Fth1","Tnf",
            "Il1b","Stat3","Rela","Hif1a","Nfe2l2","Spi1","Irf1","Smad3",
        ],
    }

    rows = []
    for cat, genes in catalog.items():
        for g in genes:
            rows.append({"gene": gene_upper(g), "source_category": cat})
    return pd.DataFrame(rows).drop_duplicates()


def is_blacklisted_gene(g, drop_predicted=True):
    """
    Remove genes that usually dominate technical/damage-driven lists:
      - mitochondrial: mt-*
      - ribosomal: Rpl*, Rps*, Mrpl*, Mrps*
      - hemoglobin: Hba*, Hbb*, Hbg*, Hbd*, Hbe*, Hbz*, Hbq*, Hbm*
      - optional predicted Gm genes
      - Malat1 and other common technical/non-specific genes
    """
    raw = str(g).strip()
    gu = raw.upper()

    if raw.lower().startswith("mt-") or gu.startswith("MT-"):
        return True, "mitochondrial"

    ribo_prefix = ("RPL", "RPS", "MRPL", "MRPS")
    if gu.startswith(ribo_prefix):
        return True, "ribosomal"

    hb_prefix = ("HBA", "HBB", "HBG", "HBD", "HBE", "HBZ", "HBQ", "HBM")
    if gu.startswith(hb_prefix):
        return True, "hemoglobin"

    if gu in {"MALAT1", "XIST"}:
        return True, "technical_lncRNA"

    if drop_predicted and re.match(r"^GM[0-9]+", gu):
        return True, "predicted_Gm_gene"

    return False, ""


def select_filtered_genes(adata, obs_set, args):
    var_names = [str(x) for x in adata.var_names]
    var_upper = [gene_upper(x) for x in var_names]

    catalog = build_stroke_gene_catalog()

    # Optional external stroke gene list.
    extra_rows = []
    if args.stroke_gene_list:
        p = Path(args.stroke_gene_list)
        if p.exists():
            df = read_table(p)
            if "gene" in df.columns:
                genes = df["gene"].astype(str).tolist()
            else:
                genes = df.iloc[:, 0].astype(str).tolist()
            for g in genes:
                extra_rows.append({"gene": gene_upper(g), "source_category": "user_stroke_gene_list"})
    if extra_rows:
        catalog = pd.concat([catalog, pd.DataFrame(extra_rows)], ignore_index=True).drop_duplicates()

    stroke_set = set(catalog["gene"].astype(str))

    rows = []
    keep_indices = []
    remove_rows = []

    for i, (raw, gu) in enumerate(zip(var_names, var_upper)):
        black, reason = is_blacklisted_gene(raw, drop_predicted=args.drop_predicted_gm)

        in_stroke = gu in stroke_set

        if black:
            remove_rows.append({
                "gene_raw": raw,
                "gene_upper": gu,
                "reason": reason,
                "in_stroke_catalog": bool(in_stroke),
            })
            continue

        if in_stroke:
            cats = sorted(catalog.loc[catalog["gene"].eq(gu), "source_category"].unique())
            rows.append({
                "gene_raw": raw,
                "gene_upper": gu,
                "keep_reason": "stroke_relevant_catalog",
                "source_category": ";".join(cats),
                "adata_var_index": i,
            })
            keep_indices.append(i)

    kept = pd.DataFrame(rows).drop_duplicates(subset=["gene_raw"])
    removed = pd.DataFrame(remove_rows)

    if kept.empty:
        raise RuntimeError("No stroke-relevant genes found after filtering. Provide --stroke_gene_list or relax filters.")

    # Optional variance ranking if too many genes.
    if len(kept) > args.max_clean_genes:
        idx = kept["adata_var_index"].astype(int).values
        X = adata[:, idx].X
        try:
            import scipy.sparse as sp
            if sp.issparse(X):
                means = np.asarray(X.mean(axis=0)).ravel()
                sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
                vars_ = sq - means ** 2
            else:
                vars_ = np.asarray(X).var(axis=0)
        except Exception:
            vars_ = np.asarray(X).var(axis=0)

        kept = kept.copy()
        kept["variance"] = vars_
        kept = kept.sort_values("variance", ascending=False).head(args.max_clean_genes).copy()
    else:
        kept["variance"] = np.nan

    return kept.reset_index(drop=True), removed.reset_index(drop=True), catalog


def export_filtered_expression_h5ad(h5ad_path, obs_set, kept_genes, out_csv):
    try:
        import anndata as ad
    except Exception as e:
        raise RuntimeError(f"anndata is required for h5ad export: {e}")

    adata = ad.read_h5ad(h5ad_path)

    obs_names = [str(x) for x in adata.obs_names]
    obs_keep_idx = [i for i, x in enumerate(obs_names) if x in obs_set]

    if len(obs_keep_idx) == 0:
        raise RuntimeError("No overlapping obs_name between h5ad and Step64c assignments.")

    gene_idx = kept_genes["adata_var_index"].astype(int).tolist()
    gene_names = kept_genes["gene_raw"].astype(str).tolist()

    X = adata[obs_keep_idx, gene_idx].X

    try:
        import scipy.sparse as sp
        if sp.issparse(X):
            X = X.toarray()
    except Exception:
        pass

    X = np.asarray(X)

    out = pd.DataFrame(X, columns=gene_names)
    out.insert(0, "obs_name", [obs_names[i] for i in obs_keep_idx])

    out.to_csv(out_csv, index=False)

    return {
        "expr_csv": str(out_csv),
        "n_obs_exported": int(out.shape[0]),
        "n_genes_exported": int(out.shape[1] - 1),
    }


# -----------------------------------------------------------------------------
# External database discovery
# -----------------------------------------------------------------------------

REG_TERMS = [
    "trrust", "dorothea", "chea", "encode", "tf_target", "tf-target",
    "tftarget", "regulon", "regulator", "transcription_factor",
]
GMT_TERMS = [
    "msigdb", "hallmark", "reactome", "kegg", "gobp", "go_bp",
    "c2.", "c3.", "c5.", "c7.", "chea", "dorothea", "encode", "tf",
]


def split_roots(s):
    roots = []
    for part in str(s).split(","):
        p = Path(part.strip())
        if p.exists():
            roots.append(p)
    return roots


def is_bad_dir(d):
    return d.startswith(".") or d in {"__pycache__", "node_modules", ".git", ".cache"}


def collect_candidate_files(roots, max_files=200000):
    files = []
    suffixes = (".csv", ".tsv", ".txt", ".csv.gz", ".tsv.gz", ".txt.gz", ".gmt", ".gmt.gz")
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not is_bad_dir(d)]
            for fn in filenames:
                p = Path(dirpath) / fn
                low = str(p).lower()
                if low.endswith(suffixes):
                    # Exclude our internal Step65 / Figure6 generated network unless user explicitly passes them.
                    if any(x in low for x in [
                        "track_dynamic_marker_regulator_65",
                        "step65_track_network_edges",
                        "figure6_regulator_communication",
                        "fig6b_cell_cell_communication",
                    ]):
                        continue
                    files.append(p)
                    if len(files) >= max_files:
                        return files
    return files


def inspect_gmt(path):
    low = str(path).lower()
    bonus = sum(1 for x in GMT_TERMS if x in low)

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
                    n_genes += len(parts) - 2
    except Exception as e:
        return {"path": str(path), "status": f"error:{e}", "score": -1}

    score = bonus * 1000 + n_sets + min(n_genes, 100000) / 100
    return {
        "path": str(path),
        "kind": "gmt",
        "status": "ok" if n_sets > 0 else "empty",
        "n_sets_head": n_sets,
        "n_genes_head": n_genes,
        "filename_bonus": bonus,
        "score": score,
    }


def inspect_reg_table(path):
    low = str(path).lower()
    bonus = sum(1 for x in REG_TERMS if x in low)
    if bonus <= 0:
        return {"path": str(path), "kind": "regulator_table", "status": "not_external_regulator_name", "score": -1}

    try:
        head = read_table(path, nrows=100)
    except Exception as e:
        return {"path": str(path), "kind": "regulator_table", "status": f"read_error:{e}", "score": -1}

    if head is None or head.empty:
        return {"path": str(path), "kind": "regulator_table", "status": "empty", "score": -1}

    cols = list(head.columns)
    nc = {norm_key(c): c for c in cols}

    reg_col = None
    for k in ["regulator", "tf", "source", "transcription_factor", "term", "geneset", "gene_set"]:
        if k in nc:
            reg_col = nc[k]
            break

    target_col = None
    for k in ["target", "target_gene", "gene", "genesymbol", "gene_symbol"]:
        if k in nc:
            target_col = nc[k]
            break

    # TRRUST can be no-header or unusual table; use first two columns if filename strongly suggests.
    if reg_col is None or target_col is None:
        if bonus >= 1 and len(cols) >= 2:
            reg_col = cols[0]
            target_col = cols[1]
            status = "possible_external_regulator_first_two_columns"
        else:
            status = "missing_regulator_target_columns"
            return {"path": str(path), "kind": "regulator_table", "status": status, "score": -1}
    else:
        status = "ok"

    score = bonus * 1000 + head[[reg_col, target_col]].dropna().shape[0] * 10

    return {
        "path": str(path),
        "kind": "regulator_table",
        "status": status,
        "reg_col": reg_col,
        "target_col": target_col,
        "database": Path(path).stem,
        "filename_bonus": bonus,
        "n_head_rows": int(head.shape[0]),
        "score": score,
    }


def find_external_resources(roots, max_files):
    files = collect_candidate_files(roots, max_files=max_files)

    reg_rows = []
    gmt_rows = []

    for p in files:
        low = str(p).lower()
        if low.endswith(".gmt") or low.endswith(".gmt.gz"):
            gmt_rows.append(inspect_gmt(p))
        elif low.endswith((".csv", ".tsv", ".txt", ".csv.gz", ".tsv.gz", ".txt.gz")):
            reg_rows.append(inspect_reg_table(p))

    reg = pd.DataFrame(reg_rows)
    gmt = pd.DataFrame(gmt_rows)

    if not reg.empty:
        reg = reg.sort_values("score", ascending=False).reset_index(drop=True)
    if not gmt.empty:
        gmt = gmt.sort_values("score", ascending=False).reset_index(drop=True)

    return reg, gmt


def load_reg_table_candidate(row):
    df = read_table(row["path"])
    reg_col = row.get("reg_col")
    target_col = row.get("target_col")

    if reg_col not in df.columns or target_col not in df.columns:
        return pd.DataFrame()

    db = row.get("database") or Path(row["path"]).stem
    rows = []

    for _, r in df[[reg_col, target_col]].dropna().iterrows():
        reg = str(r[reg_col]).strip()
        targets = re.split(r"[;,|]\s*", str(r[target_col]).strip())
        for t in targets:
            tg = gene_upper(t)
            if tg and tg != "NAN":
                rows.append({
                    "regulator": reg,
                    "target": tg,
                    "database": db,
                    "source_file": row["path"],
                })

    return pd.DataFrame(rows).drop_duplicates()


def load_gmt_candidate(path, max_sets=50000):
    rows = []
    n = 0
    with open_text_maybe_gz(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            term = parts[0]
            for g in parts[2:]:
                gu = gene_upper(g)
                if gu and gu != "NAN":
                    rows.append({
                        "regulator": term,
                        "target": gu,
                        "database": "GMT",
                        "source_file": str(path),
                    })
            n += 1
            if n >= max_sets:
                break
    return pd.DataFrame(rows).drop_duplicates()


def build_external_regulator_targets(reg_candidates, gmt_candidates, out_csv, args):
    frames = []

    if args.regulator_table:
        for p in str(args.regulator_table).split(","):
            p = Path(p.strip())
            if p.exists():
                row = inspect_reg_table(p)
                row["path"] = str(p)
                frames.append(load_reg_table_candidate(row))

    if args.gmt:
        for p in str(args.gmt).split(","):
            p = Path(p.strip())
            if p.exists():
                frames.append(load_gmt_candidate(p))

    if reg_candidates is not None and not reg_candidates.empty:
        ok = reg_candidates[reg_candidates["score"] > 0].head(args.max_external_reg_tables)
        for _, row in ok.iterrows():
            try:
                frames.append(load_reg_table_candidate(row))
            except Exception as e:
                log(f"[WARN] failed loading regulator table {row.get('path')}: {e}")

    if gmt_candidates is not None and not gmt_candidates.empty:
        ok = gmt_candidates[gmt_candidates["score"] > 0].head(args.max_external_gmt_files)
        for _, row in ok.iterrows():
            try:
                frames.append(load_gmt_candidate(row["path"]))
            except Exception as e:
                log(f"[WARN] failed loading GMT {row.get('path')}: {e}")

    frames = [x for x in frames if x is not None and not x.empty]

    if not frames:
        return "", {
            "external_db_available": False,
            "reason": "no external TRRUST/DoRothEA/ChEA/ENCODE/MSigDB/GMT resources found",
        }

    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    out = out[out["regulator"].astype(str).ne("")]
    out = out[out["target"].astype(str).ne("")]

    out.to_csv(out_csv, index=False)

    return str(out_csv), {
        "external_db_available": True,
        "path": str(out_csv),
        "n_pairs": int(len(out)),
        "n_regulators": int(out["regulator"].nunique()),
        "n_targets": int(out["target"].nunique()),
        "databases": out["database"].value_counts().head(30).to_dict(),
        "source_files": out["source_file"].drop_duplicates().head(20).tolist(),
    }


# -----------------------------------------------------------------------------
# Running Step65
# -----------------------------------------------------------------------------

def make_unique_outdir(outdir, reuse=False):
    outdir = Path(outdir)
    if reuse:
        outdir.mkdir(parents=True, exist_ok=True)
        return outdir

    if not outdir.exists():
        outdir.mkdir(parents=True, exist_ok=True)
        return outdir

    # Do not overwrite existing finished output.
    if (outdir / "step65_report.json").exists():
        import datetime
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new = Path(str(outdir) + f"_{stamp}")
        new.mkdir(parents=True, exist_ok=True)
        return new

    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def build_step65_cmd(args, outdir, expr_csv, regulator_csv):
    cmd = [
        args.python,
        str(args.step65_script),
        "--in64c", str(args.in64c),
        "--in64b", str(args.in64b),
        "--outdir", str(outdir),
        "--expr_csv", str(expr_csv),
        "--max_genes", str(args.max_clean_genes),
        "--min_cells_per_timepoint", str(args.min_cells_per_timepoint),
        "--min_abs_delta_module", str(args.min_abs_delta_module),
        "--min_abs_delta_gene", str(args.min_abs_delta_gene),
        "--top_n_modules_for_enrichment", str(args.top_n_modules_for_enrichment),
        "--top_n_genes_for_enrichment", str(args.top_n_genes_for_enrichment),
        "--min_overlap", str(args.min_overlap),
        "--top_tracks_fig", str(args.top_tracks_fig),
        "--top_modules_fig", str(args.top_modules_fig),
        "--dpi", str(args.dpi),
    ]

    if regulator_csv:
        cmd.extend(["--regulator_table", str(regulator_csv)])

    return cmd


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--h5ad", default=str(DEFAULT_H5AD))
    ap.add_argument("--in64c", default=str(DEFAULT_IN64C))
    ap.add_argument("--in64b", default=str(DEFAULT_IN64B))
    ap.add_argument("--step65_script", default=str(DEFAULT_STEP65))

    ap.add_argument("--outdir", default=str(DEFAULT_OUT))
    ap.add_argument("--reuse_outdir", action="store_true")

    ap.add_argument("--search_roots", default="/mnt/h/vir/ST,/mnt/h/Data,/mnt/h/vir")
    ap.add_argument("--regulator_table", default="", help="Optional comma-separated external TRRUST/DoRothEA/ChEA/ENCODE table paths.")
    ap.add_argument("--gmt", default="", help="Optional comma-separated GMT paths, e.g. MSigDB Hallmark/Reactome/KEGG/ChEA.")

    ap.add_argument("--stroke_gene_list", default="", help="Optional CSV/TSV with a gene column or first column as genes.")

    ap.add_argument("--max_clean_genes", type=int, default=800)
    ap.add_argument("--drop_predicted_gm", action="store_true", default=True)
    ap.add_argument("--keep_predicted_gm", action="store_false", dest="drop_predicted_gm")

    ap.add_argument("--max_files", type=int, default=200000)
    ap.add_argument("--max_external_reg_tables", type=int, default=10)
    ap.add_argument("--max_external_gmt_files", type=int, default=5)
    ap.add_argument("--require_external_db", action="store_true")

    ap.add_argument("--python", default="/home/shu/miniconda/envs/nicheformer_env/bin/python")

    ap.add_argument("--min_cells_per_timepoint", type=int, default=10)
    ap.add_argument("--min_abs_delta_module", type=float, default=0.03)
    ap.add_argument("--min_abs_delta_gene", type=float, default=0.10)
    ap.add_argument("--top_n_modules_for_enrichment", type=int, default=8)
    ap.add_argument("--top_n_genes_for_enrichment", type=int, default=50)
    ap.add_argument("--min_overlap", type=int, default=1)
    ap.add_argument("--top_tracks_fig", type=int, default=12)
    ap.add_argument("--top_modules_fig", type=int, default=12)
    ap.add_argument("--dpi", type=int, default=600)

    ap.add_argument("--run", action="store_true")

    args = ap.parse_args()

    outdir = make_unique_outdir(args.outdir, reuse=args.reuse_outdir)

    log("=" * 100)
    log("Step 65c: clean stroke-relevant gene-level Step65 with external DB")
    log("=" * 100)
    log(f"h5ad={args.h5ad}")
    log(f"in64c={args.in64c}")
    log(f"in64b={args.in64b}")
    log(f"outdir={outdir}")

    obs_set, assign = load_assignment_obs(args.in64c)

    try:
        import anndata as ad
    except Exception as e:
        raise RuntimeError(f"anndata is required: {e}")

    adata = ad.read_h5ad(args.h5ad, backed=None)

    kept, removed, catalog = select_filtered_genes(adata, obs_set, args)

    expr_csv = outdir / "step65c_filtered_stroke_relevant_expression.csv"
    expr_meta = export_filtered_expression_h5ad(args.h5ad, obs_set, kept, expr_csv)

    kept.to_csv(outdir / "step65c_genes_kept.csv", index=False)
    removed.to_csv(outdir / "step65c_genes_removed_blacklist.csv", index=False)
    catalog.to_csv(outdir / "step65c_stroke_gene_catalog_used.csv", index=False)

    audit = {
        "h5ad": str(args.h5ad),
        "adata_shape": list(adata.shape),
        "assignment_unique_obs": int(len(obs_set)),
        "n_overlapping_obs_exported": expr_meta["n_obs_exported"],
        "n_genes_kept": int(len(kept)),
        "n_genes_removed_blacklist": int(len(removed)),
        "blacklist_reason_counts": removed["reason"].value_counts().to_dict() if not removed.empty else {},
        "kept_category_counts": kept["source_category"].str.split(";").explode().value_counts().to_dict() if not kept.empty else {},
    }

    pd.DataFrame([audit]).to_csv(outdir / "step65c_gene_filter_audit.csv", index=False)

    roots = split_roots(args.search_roots)
    reg_candidates, gmt_candidates = find_external_resources(roots, max_files=args.max_files)

    reg_candidates.to_csv(outdir / "step65c_external_regulator_candidates.csv", index=False)
    gmt_candidates.to_csv(outdir / "step65c_external_gmt_candidates.csv", index=False)

    regulator_csv, reg_meta = build_external_regulator_targets(
        reg_candidates=reg_candidates,
        gmt_candidates=gmt_candidates,
        out_csv=outdir / "step65c_external_regulator_targets.csv",
        args=args,
    )

    if args.require_external_db and not reg_meta.get("external_db_available", False):
        log("[STOP] --require_external_db was set but no true external DB/GMT was found.")
        log("Place TRRUST/DoRothEA/ChEA/ENCODE/MSigDB GMT files under search_roots or pass --regulator_table / --gmt.")
        should_run = False
    else:
        should_run = True

    cmd = build_step65_cmd(args, outdir, expr_csv, regulator_csv if regulator_csv else "")

    run_sh = outdir / "step65c_run_step65_clean_gene_externaldb.sh"
    with open(run_sh, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -euo pipefail\n\n")
        f.write("cd /mnt/h/vir/ST\n\n")
        f.write(" ".join(shell_quote(x) for x in cmd))
        f.write(f" 2>&1 | tee {shell_quote(outdir / 'run_65c_clean_gene_externaldb_step65.log')}\n")

    try:
        os.chmod(run_sh, 0o755)
    except Exception:
        pass

    selected = {
        "status": "prepared",
        "outdir": str(outdir),
        "gene_filter_audit": audit,
        "expression_csv": str(expr_csv),
        "regulator_meta": reg_meta,
        "regulator_csv": regulator_csv,
        "top_external_regulator_candidates": reg_candidates.head(20).to_dict(orient="records") if not reg_candidates.empty else [],
        "top_external_gmt_candidates": gmt_candidates.head(20).to_dict(orient="records") if not gmt_candidates.empty else [],
        "step65_command": cmd,
        "run_script": str(run_sh),
        "note": (
            "This is a clean gene-level Step65 rerun. It keeps only stroke-relevant genes and "
            "removes mitochondrial/ribosomal/hemoglobin genes. It does not overwrite Step65b."
        ),
    }

    (outdir / "step65c_selected_resources.json").write_text(
        json.dumps(selected, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    lines = []
    lines.append("Step65c clean stroke-relevant gene-level external-DB detection report")
    lines.append("=" * 100)
    lines.append(json.dumps(selected, indent=2, ensure_ascii=False, default=str))
    lines.append("")
    lines.append("Gene filter audit:")
    lines.append(json.dumps(audit, indent=2, ensure_ascii=False, default=str))
    lines.append("")
    lines.append("Kept genes:")
    lines.append(kept.head(200).to_string(index=False))
    lines.append("")
    lines.append("Removed blacklist genes:")
    lines.append(removed.head(200).to_string(index=False) if not removed.empty else "None")
    lines.append("")
    lines.append("Top external regulator candidates:")
    lines.append(reg_candidates.head(50).to_string(index=False) if not reg_candidates.empty else "None")
    lines.append("")
    lines.append("Top external GMT candidates:")
    lines.append(gmt_candidates.head(50).to_string(index=False) if not gmt_candidates.empty else "None")
    lines.append("")
    lines.append("Run script:")
    lines.append(str(run_sh))

    (outdir / "step65c_detection_report.txt").write_text("\n".join(lines), encoding="utf-8")

    log("=" * 100)
    log("PREPARED Step65c")
    log("=" * 100)
    log(json.dumps(selected, indent=2, ensure_ascii=False, default=str))

    if args.run and should_run:
        log("=" * 100)
        log("RUNNING clean Step65")
        log("=" * 100)
        subprocess.run(cmd, check=True)
        log("=" * 100)
        log("DONE Step65c clean run")
        log("=" * 100)
    elif args.run and not should_run:
        raise RuntimeError("Requested --run but external DB is required and not available.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
