"""Kiểm tra notebook notebooks/raft_stsb_table3.ipynb (chạy trên CPU, không cần mạng).

    pytest -q tests/                      # unit check (~30 s) + chạy toàn bộ SMOKE_TEST (~5 phút)
    RAFT_SKIP_SLOW=1 pytest -q tests/     # chỉ unit check
    STSB_CSV_DIR=/path/to/csv pytest -q tests/test_notebook.py::test_data_pipeline_real_csv
        # tuỳ chọn: kiểm tra tách tập trên dữ liệu STS-B thật (stsb-en-train.csv, stsb-en-dev.csv)
"""
import glob
import json
import os
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "raft_stsb_table3.ipynb"


def _execute(nb, workdir):
    NotebookClient(nb, timeout=3600, kernel_name="python3",
                   resources={"metadata": {"path": str(workdir)}}).execute()
    return nb


def _text(cell):
    out = []
    for o in cell.get("outputs", []):
        if o.get("output_type") == "stream":
            out.append(o["text"])
        elif "text/plain" in o.get("data", {}):
            out.append(o["data"]["text/plain"])
    return "".join(out)


def _cell_index(nb, key):
    return next(i for i, c in enumerate(nb.cells) if key in c.get("id", ""))


def test_unit_checks(tmp_path, monkeypatch):
    monkeypatch.setenv("RAFT_SMOKE_TEST", "1")
    monkeypatch.setenv("RAFT_OUTPUT_DIR", str(tmp_path / "out"))
    nb = nbformat.read(NB_PATH, as_version=4)
    i = _cell_index(nb, "unit-checks")
    nb.cells = nb.cells[:i + 1]
    _execute(nb, tmp_path)
    assert "ALL UNIT CHECKS PASSED" in _text(nb.cells[i])


@pytest.mark.skipif(os.environ.get("RAFT_SKIP_SLOW") == "1", reason="RAFT_SKIP_SLOW=1")
def test_full_smoke_run(tmp_path, monkeypatch):
    monkeypatch.setenv("RAFT_SMOKE_TEST", "1")
    monkeypatch.setenv("RAFT_OUTPUT_DIR", str(tmp_path / "out"))
    nb = _execute(nbformat.read(NB_PATH, as_version=4), tmp_path)
    out = tmp_path / "out" / "smoke"
    zs = json.load(open(out / "zero_shot.json"))
    assert set(zs["test"]) >= {"greedy", "rail_sampling", "rail_grid_digits"}
    runs = [json.load(open(p)) for p in glob.glob(str(out / "runs" / "*.json"))]
    assert len(runs) == 3 * 2 * 2          # {ce, head, raft} x 2 LR x 2 seed trong SMOKE_TEST
    for r in runs:
        assert r["history"][0]["step"] == 0 and r["history"][-1]["step"] == r["max_steps"]
        if r["method"] == "raft":          # RAFT ở bước 0 == RAIL lưới zero-shot (cùng predictor, §E.5)
            assert abs(r["history"][0]["test"]["raft"]["rmse"] - zs["test"]["rail_grid_digits"]["rmse"]) < 1e-6
            assert r["train_loss"][-1][1] < r["train_loss"][0][1]
    table = (out / "table3_stsb.md").read_text()
    assert "RAFT" in table and "Predictive head" in table and "SMOKE_TEST" in table
    assert "ALL UNIT CHECKS PASSED" in _text(nb.cells[_cell_index(nb, "unit-checks")])


@pytest.mark.skipif(not os.environ.get("STSB_CSV_DIR"), reason="cần STSB_CSV_DIR")
def test_data_pipeline_real_csv(tmp_path, monkeypatch):
    monkeypatch.setenv("RAFT_SMOKE_TEST", "0")
    monkeypatch.setenv("RAFT_OUTPUT_DIR", str(tmp_path / "out"))
    nb = nbformat.read(NB_PATH, as_version=4)
    i = _cell_index(nb, "c05-data")
    j = _cell_index(nb, "c02-config")
    cells = nb.cells[:j + 1] + [nbformat.v4.new_code_cell('DATA_SOURCE = "csv"')] + nb.cells[j + 1:i + 1]
    nb.cells = [c for c in cells if "c01-install" not in c.get("id", "")]
    _execute(nb, tmp_path)
    txt = _text(nb.cells[-1])
    assert "train n= 4886" in txt and "val   n=  863" in txt and "test  n= 1500" in txt, txt
    split = json.load(open(glob.glob(str(tmp_path / "out" / "*" / "split_indices.json"))[0]))
    assert len(set(split["train_idx"]) | set(split["val_idx"])) == 5749
