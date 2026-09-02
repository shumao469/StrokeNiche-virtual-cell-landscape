import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_drug_bridge_retains_all_targets_for_one_effect(tmp_path):
    effects = tmp_path / "effects.csv"
    bridge = tmp_path / "bridge.csv"
    mapping = tmp_path / "mapping.csv"
    output = tmp_path / "combined.csv"
    pd.DataFrame(
        [{"perturbation": "axis", "delta_core": -0.1, "delta_repair": 0.2}]
    ).to_csv(effects, index=False)
    pd.DataFrame(
        [
            {"drug_name": "drug", "target": "G1", "neighbor_bridge_evidence_tier": "tier1"},
            {"drug_name": "drug", "target": "G2", "neighbor_bridge_evidence_tier": "tier2"},
        ]
    ).to_csv(bridge, index=False)
    pd.DataFrame(
        [
            {"target_gene": "G1", "gene_perturbation": "axis"},
            {"target_gene": "G2", "gene_perturbation": "axis"},
        ]
    ).to_csv(mapping, index=False)
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_drug_target_proxy_effects.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--gene-effects", str(effects),
            "--drug-bridge", str(bridge),
            "--mapping", str(mapping),
            "--output", str(output),
            "--target-col", "target",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    result_table = pd.read_csv(output)
    drug = result_table.loc[result_table["modality"] == "drug_proxy"].iloc[0]
    assert drug["target_genes"] == "G1,G2"
    assert "tier1,tier2" in drug["source"]
    assert drug["proxy_from_perturbation"] == "axis"


def test_drug_bridge_rejects_missing_or_unmapped_identifiers(tmp_path):
    effects = tmp_path / "effects.csv"
    mapping = tmp_path / "mapping.csv"
    pd.DataFrame(
        [{"perturbation": "axis", "delta_core": -0.1, "delta_repair": 0.2}]
    ).to_csv(effects, index=False)
    pd.DataFrame(
        [{"target_gene": "G1", "gene_perturbation": "axis"}]
    ).to_csv(mapping, index=False)
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_drug_target_proxy_effects.py"
    for case, bridge_row, expected in [
        ("blank", {"drug_name": "", "target": "G1"}, "missing identifiers"),
        ("unmapped", {"drug_name": "drug", "target": "G9"}, "absent from the mapping"),
    ]:
        bridge = tmp_path / f"bridge_{case}.csv"
        pd.DataFrame([bridge_row]).to_csv(bridge, index=False)
        result = subprocess.run(
            [
                sys.executable, str(script),
                "--gene-effects", str(effects),
                "--drug-bridge", str(bridge),
                "--mapping", str(mapping),
                "--output", str(tmp_path / f"out_{case}.csv"),
                "--target-col", "target",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert expected in result.stderr


def test_drug_bridge_rejects_malformed_manifest_before_writing_output(tmp_path):
    effects = tmp_path / "effects.csv"
    bridge = tmp_path / "bridge.csv"
    mapping = tmp_path / "mapping.csv"
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "combined.csv"
    pd.DataFrame(
        [{"perturbation": "axis", "delta_core": -0.1, "delta_repair": 0.2}]
    ).to_csv(effects, index=False)
    pd.DataFrame([{"drug_name": "drug", "target": "G1"}]).to_csv(bridge, index=False)
    pd.DataFrame(
        [{"target_gene": "G1", "gene_perturbation": "axis"}]
    ).to_csv(mapping, index=False)
    manifest.write_text("{not valid json", encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_drug_target_proxy_effects.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--gene-effects",
            str(effects),
            "--drug-bridge",
            str(bridge),
            "--mapping",
            str(mapping),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
            "--target-col",
            "target",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "valid JSON manifest" in result.stderr
    assert not output.exists()
