"""Dashboard smoke test (slow; opt-in):  RUN_SLOW=1 pytest tests/test_dashboard_smoke.py"""
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="slow: set RUN_SLOW=1")
@pytest.mark.skipif(not (ROOT / "demo").exists(), reason="no demo bundle")
def test_dashboard_renders_without_exceptions():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=900)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [b.label for b in at.button]
    assert "Verify chain" in labels
    at.button[labels.index("Record this forecast")].click().run()
    at.button[labels.index("Verify chain")].click().run()
    assert any("Chain OK" in s.value for s in at.success)
