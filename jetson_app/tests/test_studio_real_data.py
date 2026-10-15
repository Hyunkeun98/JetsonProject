"""실제 NX5 데이터(JSONData/run3)로 6단계 흐름의 핵심을 확인한다. 데이터가 없는 체크아웃에서는 건너뛴다."""
import shutil
from datetime import datetime
from pathlib import Path

import pytest

from jetson_app.replay import load_records
from jetson_app.segments import Segment
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


@pytest.mark.skipif(not (RUN3.is_dir() and RUN4.is_dir()), reason="JSONData/run3, run4 가 없는 체크아웃에서는 건너뜁니다")
def test_run3_collection_then_run4_monitoring_detects_seven_of_nine_shocks(tmp_path):
    """6단계 흐름으로 학습한 결과가 기존 실험(accept_deviation.py)과 같은 7/9 검출을 낸다.
    기준: 라벨 시각을 -6.1초 보정(PC 시계 차이)하고 ±3초 창에서 해당 축의 알람을 찾는다."""
    controller = StudioController(tmp_path, io=FakeIO(), source_factory=SourceFactory(), run_worker=False)
    controller.save_connection(CONNECTION)
    controller.import_folder(str(RUN3))
    controller.train(TAGS)
    controller.join_training(timeout=300)
    assert controller.state()["stage"] == "MONITOR", controller.state()["error"]
    controller.start()
    everything = Segment(kind="eval", name="all", start_ns=0, end_ns=2**62)
    for n, record in enumerate(load_records(RUN4, tuple(TAGS), (everything,))):
        controller.feed_record(record)
        if n % 200 == 0:
            controller.pump()                      # 처리 큐 한도(10000)를 넘지 않게 계속 비운다
    controller.pump()

    labels = []
    for line in (RUN4 / "batches.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] in ("axisxerror", "axisCVerror"):
            stamp = parts[0]
            labels.append((parts[1], datetime.fromisoformat(stamp[:26] + stamp[-6:]).timestamp() - 6.1))
    axis = {"axisxerror": "axx_act_pos", "axisCVerror": "axcv_act_pos"}
    alarms = [(t / 1000.0, name) for t, groups in controller.pipeline.history._rows
              for name, (_, alarm, _) in groups.items() if alarm]

    detected = sum(any(name == axis[kind] and t0 - 3 <= t <= t0 + 3 for t, name in alarms) for kind, t0 in labels)
    far = [(t, name) for t, name in alarms
           if not any(name == axis[kind] and t0 - 10 <= t <= t0 + 10 for kind, t0 in labels)]
    assert len(labels) == 9
    assert detected >= 7, detected
    assert far == []                               # 라벨과 무관한 알람(정상 오탐)은 없다


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
