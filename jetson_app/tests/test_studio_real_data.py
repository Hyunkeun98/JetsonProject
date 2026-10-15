"""실제 NX5 데이터(JSONData/run3)로 6단계 흐름의 핵심을 확인한다. 데이터가 없는 체크아웃에서는 건너뛴다."""
import shutil
from pathlib import Path

import pytest

from jetson_app.studio.controller import StudioController
from studio_fakes import SourceFactory

RUN3 = Path(__file__).resolve().parents[2] / "JSONData" / "run3"
RUN4 = Path(__file__).resolve().parents[2] / "JSONData" / "run4"
TAGS = [
    "NX5_AxisData:AxX_Act_Pos",
    "NX5_AxisData:AxZ_Act_Pos",
    "NX5_AxisData:AxCV_Act_Pos",
]
EXPECTED_SIGMA = {
    "axx_act_pos": 3.92e-5,
    "axz_act_pos": 3.53e-6,
    "axcv_act_pos": 3.44e-5,
}
CONNECTION = {
    "equipment_id": "nx5", "host": "localhost", "port": 1883,
    "subscribe_topics": ["dx1/AxisData", "dx1/ProcStart"],
}
needs_data = pytest.mark.skipif(not RUN3.is_dir(), reason="JSONData/run3 이 없는 체크아웃에서는 건너뜁니다")
needs_run4 = pytest.mark.skipif(not RUN4.is_dir(), reason="JSONData/run4 가 없는 체크아웃에서는 건너뜁니다")


class FakeIO:
    def connect(self, pipeline, host, port):
        pass

    def disconnect(self, pipeline):
        pass

    def is_connected(self, pipeline):
        return True


@needs_data
def test_run3_recommends_the_position_tags_counts_cycles_and_reproduces_the_noise_levels(tmp_path):
    controller = StudioController(tmp_path, io=FakeIO(), source_factory=SourceFactory(), run_worker=False)
    controller.save_connection(CONNECTION)
    controller.import_folder(str(RUN3))

    analysis = controller.analyze()
    recommended = {t["tag"] for t in analysis["tags"] if t["recommended"]}
    assert set(TAGS) <= recommended
    assert analysis["cycles"]["method"] == "edge" and 8 <= analysis["cycles"]["cycles"] <= 12

    controller.train(TAGS)
    controller.join_training(timeout=300)

    state = controller.state()
    assert state["stage"] == "MONITOR", state["error"]
    sigma = {s["name"]: s["sigma"] for s in state["signals"]}
    for name, expected in EXPECTED_SIGMA.items():
        assert sigma[name] == pytest.approx(expected, rel=0.1), name


@needs_data
def test_without_a_cycle_signal_the_period_estimate_still_finds_about_ten_cycles(tmp_path):
    only_axes = tmp_path / "axes_only"
    only_axes.mkdir()
    shutil.copy2(RUN3 / "DX_AxisData.json", only_axes / "DX_AxisData.json")
    controller = StudioController(tmp_path / "ws", io=FakeIO(), source_factory=SourceFactory(), run_worker=False)
    controller.save_connection(dict(CONNECTION, subscribe_topics=["dx1/AxisData"]))
    controller.import_folder(str(only_axes))

    cycles = controller.analyze(with_tags=False)["cycles"]

    assert cycles["method"] == "period" and cycles["exact"] is False
    assert 6 <= cycles["cycles"] <= 14, cycles["message"]


@needs_run4
def test_a_spurious_period_from_a_single_tag_is_not_reported_as_cycles(tmp_path):
    # run4에서는 AxZ_Act_Vel 한 태그만 3.5초짜리 가짜 주기(약 680사이클)를 냈다.
    only_axes = tmp_path / "axes_only"
    only_axes.mkdir()
    shutil.copy2(RUN4 / "DX_AxisData.json", only_axes / "DX_AxisData.json")
    controller = StudioController(tmp_path / "ws", io=FakeIO(), source_factory=SourceFactory(), run_worker=False)
    controller.save_connection(dict(CONNECTION, subscribe_topics=["dx1/AxisData"]))
    controller.import_folder(str(only_axes))

    cycles = controller.analyze(with_tags=False)["cycles"]

    assert cycles["method"] == "none", cycles["message"]
