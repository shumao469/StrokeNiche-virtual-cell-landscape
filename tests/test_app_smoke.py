from pathlib import Path

import pandas as pd
import pytest


streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from app.perturbation_explorer import (  # noqa: E402
    has_minimum_landscape_support,
    read_csv_upload,
)


def test_small_filtered_subset_skips_landscape_smoothing():
    assert not has_minimum_landscape_support(pd.DataFrame(index=range(2)))
    assert has_minimum_landscape_support(pd.DataFrame(index=range(3)))


def test_upload_reader_preserves_semantic_leading_zero_ids():
    class Upload:
        name = "input.csv"

        def __init__(self, value: bytes):
            self._value = value
            self.size = len(value)

        def getvalue(self) -> bytes:
            return self._value

    cells = read_csv_upload(Upload(b"obs_name,latent1\n001,0.5\n002,1.5\n"))
    effects = read_csv_upload(Upload(b"perturbation,delta_core\n001,-0.1\n"))
    assert cells["obs_name"].tolist() == ["001", "002"]
    assert effects["perturbation"].tolist() == ["001"]


def test_app_runs_in_gene_and_drug_modes():
    app = Path(__file__).resolve().parents[1] / "app" / "perturbation_explorer.py"
    at = AppTest.from_file(str(app), default_timeout=30).run()
    assert not at.exception
    assert any(metric.label == "Virtual cells" for metric in at.metric)

    modality = next(box for box in at.selectbox if box.label == "Modality")
    modality.select("drug_proxy")
    at.run()
    assert not at.exception
    candidate = next(box for box in at.selectbox if box.label == "Candidate")
    assert "proxy" in candidate.value.lower()
