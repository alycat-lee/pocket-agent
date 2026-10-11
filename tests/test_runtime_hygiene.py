from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_hygiene():
    path = REPO / "scripts" / "check_runtime_hygiene.py"
    spec = importlib.util.spec_from_file_location("check_runtime_hygiene", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_hygiene_src_is_clean() -> None:
    mod = _load_hygiene()
    assert mod.find_leaks(REPO / "src") == []


def test_hygiene_script_detects_leaks(tmp_path: Path) -> None:
    mod = _load_hygiene()
    src = tmp_path / "src"
    src.mkdir()
    (src / "leak.py").write_text('x = "MockLLM"\n', encoding="utf-8")
    (src / "ok.py").write_text("x = 1\n", encoding="utf-8")
    leaks = mod.find_leaks(src)
    assert any("MockLLM" in item for item in leaks)
    assert mod.main([str(src)]) == 1
    clean = tmp_path / "clean_src"
    clean.mkdir()
    (clean / "ok.py").write_text("x = 1\n", encoding="utf-8")
    assert mod.main([str(clean)]) == 0
