from pathlib import Path

import pytest

from jetson_app.replay import load_records
from jetson_app.segments import Segment
from jetson_app.studio.controller import StudioController

RUN3 = Path(__file__).resolve().parents[2] / "JSONData" / "run3"
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


class FakeIO:
    def connect(self, pipeline, host, port):
        pass

    def disconnect(self, pipeline):
        pass

    def is_connected(self, pipeline):
        return True


@pytest.mark.skipif(not RUN3.is_dir(), reason="JSONData/run3 이 없는 체크아웃에서는 건너뜁니다")
def test_run3_training_reproduces_the_known_noise_levels(tmp_path):
    controller = StudioController(tmp_path, io=FakeIO(), run_worker=False)
    controller.save_config({
        "equipment_id": "nx5", "host": "localhost", "port": 1883,
        "subscribe_topics": ["dx1/AxisData"], "tags": TAGS, "min_samples": 3000,
    })
    controller.start()
    everything = Segment(kind="train", name="all", start_ns=0, end_ns=2**62)
    for n, record in enumerate(load_records(RUN3, tuple(TAGS), (everything,))):
        controller.feed_record(record)
        if n % 200 == 0:
            controller.pump()          # 처리 큐는 한도가 있어서, 실제 앱의 워커처럼 계속 비워 준다
    controller.pump()

    controller.train()
    controller.join_training(timeout=120)

    state = controller.state()
    assert state["stage"] == "MONITORING", state["error"]
    sigma = {s["name"]: s["sigma"] for s in state["signals"]}
    for name, expected in EXPECTED_SIGMA.items():
        assert sigma[name] == pytest.approx(expected, rel=0.1), name
