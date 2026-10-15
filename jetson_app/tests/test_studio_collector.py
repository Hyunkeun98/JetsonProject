import json

import pytest

from jetson_app.studio.collector import Collector
from jetson_app.studio.errors import StudioError
from studio_fakes import SourceFactory, payload


def _collector(tmp_path, factory=None, clock=None, **kwargs):
    factory = factory or SourceFactory()
    now = clock if clock is not None else [0.0]
    collector = Collector(tmp_path, factory, clock=lambda: now[0], **kwargs)
    collector.set_equipment("nx5")
    return collector, factory, now


def _cycle_payload(i):
    return payload(i, pos=float(i), U0=1 if i % 100 >= 90 else 0)


def test_payloads_are_stored_as_they_arrive_and_cycles_are_counted(tmp_path):
    collector, factory, now = _collector(tmp_path)
    collector.start("broker", 1883, ["dx1/AxisData"])
    for i in range(1200):
        now[0] = i * 0.1
        factory.last.emit(_cycle_payload(i))
    collector.stop()

    lines = (tmp_path / "collections" / "nx5" / "current" / "capture.jsonl").read_bytes().splitlines()
    state = collector.state()

    assert len(lines) == 1200 and json.loads(lines[5])["records"][0]["pos"] == 5.0
    assert state["active"] is False and state["has_data"] is True and state["ended_reason"] == "user"
    assert state["messages"] == 1200 and state["bytes"] > 0 and state["seconds"] == pytest.approx(119.9, abs=0.2)
    assert state["cycles"]["method"] == "edge" and state["cycles"]["cycles"] == 12
    assert factory.last.connected is False and factory.last.calls == [("broker", 1883, ("dx1/AxisData",))]


def test_a_payload_with_line_breaks_is_stored_on_one_line(tmp_path):
    collector, factory, _ = _collector(tmp_path)
    collector.start("broker", 1883, ["t"])

    factory.last.emit(b'{\n  "records": [\n    {"timestamp": "2026-08-04T00:00:00.000000000+0000", "pos": 1}\n  ]\n}\n')
    collector.stop()

    lines = (tmp_path / "collections" / "nx5" / "current" / "capture.jsonl").read_bytes().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["records"][0]["pos"] == 1


def test_collection_ends_by_itself_at_the_size_limit(tmp_path):
    collector, factory, _ = _collector(tmp_path, max_bytes=3000)
    collector.start("broker", 1883, ["t"])

    for i in range(200):
        factory.last.emit(payload(i, pos=1.0))

    state = collector.state()
    assert state["active"] is False and state["ended_reason"] == "size"
    assert factory.last.connected is False
    assert state["bytes"] < 3000 + 200               # 상한을 크게 넘겨 쓰지 않는다


def test_starting_again_moves_the_previous_collection_to_archive(tmp_path):
    collector, factory, _ = _collector(tmp_path)
    collector.start("broker", 1883, ["t"])
    factory.last.emit(payload(0, pos=111.0))
    collector.stop()

    collector.start("broker", 1883, ["t"])
    factory.last.emit(payload(0, pos=222.0))
    collector.stop()

    root = tmp_path / "collections" / "nx5"
    archived = list((root / "archive").iterdir())
    assert len(archived) == 1 and b"111.0" in (archived[0] / "capture.jsonl").read_bytes()
    assert b"222.0" in (root / "current" / "capture.jsonl").read_bytes()


def test_a_refused_connection_leaves_the_existing_collection_untouched(tmp_path):
    collector, factory, _ = _collector(tmp_path)
    collector.start("broker", 1883, ["t"])
    factory.last.emit(payload(0, pos=111.0))
    collector.stop()
    factory.fail = True

    with pytest.raises(StudioError, match="broker:1883"):
        collector.start("broker", 1883, ["t"])

    state = collector.state()
    assert state["active"] is False and state["has_data"] is True and state["messages"] == 1
    assert not (tmp_path / "collections" / "nx5" / "archive").exists()


def test_start_twice_is_refused(tmp_path):
    collector, _, _ = _collector(tmp_path)
    collector.start("broker", 1883, ["t"])

    with pytest.raises(StudioError, match="이미 수집"):
        collector.start("broker", 1883, ["t"])


def test_discard_removes_the_data_but_not_while_collecting(tmp_path):
    collector, factory, _ = _collector(tmp_path)
    collector.start("broker", 1883, ["t"])
    factory.last.emit(payload(0, pos=1.0))
    with pytest.raises(StudioError, match="수집 중"):
        collector.discard()
    collector.stop()

    collector.discard()

    assert collector.state()["has_data"] is False and not (tmp_path / "collections" / "nx5" / "current").exists()


def test_import_folder_copies_the_data_and_archives_the_previous_collection(tmp_path):
    source_dir = tmp_path / "saved data"
    source_dir.mkdir()
    lines = [_cycle_payload(i) for i in range(1200)]
    (source_dir / "DX_AxisData.json").write_bytes(b"\n".join(lines) + b"\n")
    (source_dir / "notes.txt").write_text("not data", encoding="utf-8")
    collector, factory, _ = _collector(tmp_path / "ws")
    collector.start("broker", 1883, ["t"])
    factory.last.emit(payload(0, pos=1.0))
    collector.stop()

    collector.import_folder(f'  "{source_dir}"  ')

    current = tmp_path / "ws" / "collections" / "nx5" / "current"
    state = collector.state()
    assert (current / "DX_AxisData.json").exists() and not (current / "notes.txt").exists()
    assert state["has_data"] is True and state["ended_reason"] == "import" and state["cycles"]["cycles"] == 12
    assert len(list((current.parent / "archive").iterdir())) == 1


def test_import_folder_rejects_a_missing_or_empty_folder(tmp_path):
    collector, _, _ = _collector(tmp_path / "ws")
    (tmp_path / "empty").mkdir()

    with pytest.raises(StudioError, match="찾을 수 없습니다"):
        collector.import_folder(str(tmp_path / "nope"))
    with pytest.raises(StudioError, match=r"\.json"):
        collector.import_folder(str(tmp_path / "empty"))


def test_a_capture_that_was_never_finished_is_reported_as_interrupted(tmp_path):
    collector, factory, _ = _collector(tmp_path)
    collector.start("broker", 1883, ["t"])
    factory.last.emit(payload(0, pos=1.0))
    # 앱이 꺼진 것과 같다: stop()을 부르지 않고 새 Collector가 같은 작업 폴더를 연다
    reopened, _, _ = _collector(tmp_path)

    state = reopened.state()

    assert state["active"] is False and state["has_data"] is True and state["interrupted"] is True


def test_no_data_flag_after_thirty_silent_seconds_while_collecting(tmp_path):
    collector, factory, now = _collector(tmp_path)
    collector.start("broker", 1883, ["t"])
    assert collector.state()["no_data"] is False

    now[0] = 31.0

    assert collector.state()["no_data"] is True
    factory.last.emit(payload(0, pos=1.0))
    assert collector.state()["no_data"] is False


def test_the_equipment_cannot_change_while_collecting(tmp_path):
    collector, _, _ = _collector(tmp_path)
    collector.start("broker", 1883, ["t"])

    with pytest.raises(StudioError, match="수집 중"):
        collector.set_equipment("other")


def test_state_without_an_equipment_is_empty(tmp_path):
    collector = Collector(tmp_path, SourceFactory())

    state = collector.state()

    assert state["active"] is False and state["has_data"] is False and state["cycles"] is None
