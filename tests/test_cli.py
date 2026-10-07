import json
import subprocess
import sys


def test_demo_and_inspection_work_from_another_directory(tmp_path):
    output = tmp_path / "demo"
    result = subprocess.run([sys.executable, "-m", "lmc_auriga", "demo", "--particles", "300",
                             "--output", str(output)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["input_metadata"]["provenance"] == "synthetic"
    assert 0 < report["selected_count"] < 300
    inspection = subprocess.run([sys.executable, "-m", "lmc_auriga", "inspect", str(output / "synthetic.hdf5")],
                                cwd=tmp_path, capture_output=True, text=True)
    assert inspection.returncode == 0, inspection.stderr
    assert json.loads(inspection.stdout)["n_particles"] == 300
    original = (output / "report.json").read_bytes()
    repeated = subprocess.run([sys.executable, "-m", "lmc_auriga", "demo", "--particles", "300",
                               "--output", str(output)], cwd=tmp_path, capture_output=True, text=True)
    assert repeated.returncode == 2
    assert (output / "report.json").read_bytes() == original


def test_invalid_analysis_returns_readable_cli_error(tmp_path):
    result = subprocess.run([sys.executable, "-m", "lmc_auriga", "analyze", str(tmp_path / "absent.hdf5"),
                             "--output", str(tmp_path / "results"), "--phases", "0"],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "phase_count" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "results").exists()


def test_benchmark_reports_missing_optional_backend(monkeypatch, capsys, tmp_path):
    from lmc_auriga.cli import main
    from lmc_auriga import backends
    monkeypatch.setattr(backends, "available_backends", lambda: ("python", "numpy"))
    assert main(["benchmark", "--backends", "numba", "--output", str(tmp_path / "benchmark")]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "no_available_backends"
    assert result["skipped_backends"]
    assert result["summary"] == []
