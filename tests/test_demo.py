import pytest

from growthlab import config
from growthlab.decision import simulate

pytestmark = pytest.mark.skipif(
    not simulate.decisionspath.exists(), reason="run `make clv uplift decision` first"
)


def test_demo_renders_without_errors():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(config.root / "src" / "growthlab" / "demo.py"), default_timeout=180)
    app.run()
    assert not app.exception
    assert any("Who should get the retention offer?" in m.value for m in app.markdown)
    assert len(app.tabs) == 3

    app.select_slider[0].set_value(20).run()
    assert not app.exception
