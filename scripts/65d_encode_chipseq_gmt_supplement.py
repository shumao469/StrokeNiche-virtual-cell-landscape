#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import gzip
import json
import re
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

def open_text(path):
    path = Path(path)
    if str(path).lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")

def gene_upper(x):
    return str(x).strip().upper()

def parse_tf_from_term(term):
    """
    Conservative parser for ChIP-seq GMT names.
    Examples may include:
      TF_ARCHS4_PEARSON
      ENCODE_TF_CHIP_SEQ...
      RELA...
      MAFB_ARCHS4_PEARSON
    Keep first token-like TF if possible.
    """
    s = str(term).strip()
    s = re.sub(r"[^A-Za-z0-9_\\-\\.]+", "_", s)
    parts = re.split(r"[_:;|\\s]+", s)
    for p in parts:
        p2 = p.strip()
        if 2 <= len(p2) <= 20 and re.match(r"^[A-Za-z][A-Za-z0-9\\-\\.]+$", p2):
            bad = {"ENCODE", "CHIP", "CHIPSEQ", "CHIPSE", "SEQ", "TARGETS", "PEARSON", "GMT", "REMAP", "LITERATURE"}
            if p2.upper() not in bad:
                return p2.upper()
    return s[:40].upper()

def load_gmt_files(gmt_files):
    rows = []
    for gmt in gmt_files:
        p = Path(gmt)
        if not p.exists():
            continue
        source_name = p.name
        with open_text(p) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                term = parts[0]
                tf = parse_tf_from_term(term)
                genes = sorted(set(gene_upper(x) for x in parts[2:] if str(x).strip()))
                for gene in genes:
                    rows.append({
                        "regulator": tf,
                        "term": term,
                        "target": gene,
                        "database": "ENCODE_ReMap_Literature_ChIPseq_GMT",
                        "source_file": str(p),
                        "source_name": source_name,
                    })
    return pd.DataFrame(rows).drop_duplicates()

def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return out
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    m = len(ranked)
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out[order] = np.clip(q, 0, 1)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step65_out", required=True)
    ap.add_argument("--gmt", required=True, help="comma-separated GMT files")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--top_n_genes", type=int, default=50)
    ap.add_argument("--min_overlap", type=int, default=2)
    args = ap.parse_args()

    step65 = Path(args.step65_out)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dyn = pd.read_csv(step65 / "step65_track_dynamic_genes.csv", low_memory=False)
    dyn["feature_upper"] = dyn["feature"].map(gene_upper)

    gmt_files = [x for x in str(args.gmt).split(",") if x.strip()]
    edges = load_gmt_files(gmt_files)
    edges.to_csv(outdir / "step65d_encode_chipseq_gmt_edges_parsed.csv", index=False)

    universe = set(dyn["feature_upper"].unique()) | set(edges["target"].unique())
    M = len(universe)

    rows = []
    for track_id, sub in dyn.groupby("track_id"):
        top = sub.sort_values("dynamic_marker_score", ascending=False).head(args.top_n_genes)
        genes = set(top["feature_upper"].unique())
        n = len(genes)
        for (reg, term, source), tg in edges.groupby(["regulator", "term", "source_name"])["target"]:
            targets = set(tg.unique()) & universe
            K = len(targets)
            overlap = genes & targets
            x = len(overlap)
            if x < args.min_overlap:
                continue
            p = float(hypergeom.sf(x - 1, M, K, n))
            rows.append({
                "track_id": track_id,
                "regulator": reg,
                "term": term,
                "source_name": source,
                "overlap_n": x,
                "marker_gene_n": n,
                "target_gene_n": K,
                "universe_n": M,
                "p_value": p,
                "overlap_genes": ";".join(sorted(overlap)),
            })

    enr = pd.DataFrame(rows)
    if not enr.empty:
        enr["fdr_bh"] = bh_fdr(enr["p_value"].values)
        enr["encode_chipseq_score"] = -np.log10(enr["fdr_bh"].clip(lower=1e-300)) * enr["overlap_n"]
        enr = enr.sort_values(["fdr_bh", "p_value", "overlap_n"], ascending=[True, True, False])
    enr.to_csv(outdir / "step65d_encode_chipseq_track_enrichment.csv", index=False)

    summary = []
    if not enr.empty:
        for tid, sub in enr.groupby("track_id"):
            top = sub.sort_values(["fdr_bh", "p_value", "overlap_n"], ascending=[True, True, False]).head(10)
            summary.append({
                "track_id": tid,
                "top_encode_chipseq_terms": ";".join(
                    f"{r.regulator}|{r.source_name}|overlap={r.overlap_n}|FDR={r.fdr_bh:.2g}|genes={r.overlap_genes}"
                    for r in top.itertuples()
                ),
                "n_encode_chipseq_terms_fdr_0_25": int((sub["fdr_bh"] <= 0.25).sum()),
            })
    summary = pd.DataFrame(summary)
    summary.to_csv(outdir / "step65d_encode_chipseq_track_summary.csv", index=False)

    report = {
        "status": "ok",
        "step65_out": str(step65),
        "gmt_files": gmt_files,
        "n_edges": int(len(edges)),
        "n_enrichment_rows": int(len(enr)),
        "n_tracks_with_encode_chipseq_support": int(summary["track_id"].nunique()) if not summary.empty else 0,
        "outputs": {
            "edges": str(outdir / "step65d_encode_chipseq_gmt_edges_parsed.csv"),
            "enrichment": str(outdir / "step65d_encode_chipseq_track_enrichment.csv"),
            "summary": str(outdir / "step65d_encode_chipseq_track_summary.csv"),
            "report": str(outdir / "step65d_encode_chipseq_report.json"),
        }
    }
    (outdir / "step65d_encode_chipseq_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    if not enr.empty:
        print(enr.head(50).to_string(index=False))

if __name__ == "__main__":
    main()
