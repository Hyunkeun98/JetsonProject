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
    state = _wait_for_cycles(collector)
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


# --- 리뷰 반영 -----------------------------------------------------------------------------------


def _wait_for_cycles(collector, timeout=10.0):
    import time

    end = time.time() + timeout
    while time.time() < end:
        state = collector.state()
        if state["cycles"] is not None:
            return state
        time.sleep(0.05)
    raise AssertionError("사이클 수가 나오지 않았다")


def test_a_malformed_payload_does_not_stop_collection_or_break_later_state_calls(tmp_path):
    collector, factory, _ = _collector(tmp_path)
    collector.start("broker", 1883, ["t"])

    factory.last.emit(b'{"records":[{"timestamp":"2026-08-04T00:00:00.000000000+0000","x":' + b"9" * 400 + b"}]}")
    factory.last.emit(b"[" * 100000)
    for i in range(1200):
        factory.last.emit(_cycle_payload(i))
    state = collector.state()
    collector.stop()

    assert state["active"] is True and state["messages"] == 1202
    reopened, _, _ = _collector(tmp_path)                     # 다시 열어 처음부터 훑어도 죽지 않는다
    assert _wait_for_cycles(reopened)["cycles"]["cycles"] == 12


def test_a_write_failure_ends_the_collection_with_a_reason(tmp_path):
    collector, factory, _ = _collector(tmp_path)
    collector.start("broker", 1883, ["t"])

    class FullDisk:
        def write(self, data):
            raise OSError("disk full")

        def flush(self):
            pass

        def close(self):
            pass

    collector._file = FullDisk()
    factory.last.emit(payload(0, pos=1.0))

    state = collector.state()
    assert state["active"] is False and state["ended_reason"] == "error" and factory.last.connected is False


def test_start_closes_the_broker_connection_when_the_capture_file_cannot_be_created(tmp_path):
    (tmp_path / "collections").mkdir()
    (tmp_path / "collections" / "nx5").write_text("이 자리가 파일이라 폴더를 만들 수 없다", encoding="utf-8")
    collector, factory, _ = _collector(tmp_path)

    with pytest.raises(StudioError, match="수집 파일"):
        collector.start("broker", 1883, ["t"])

    assert factory.last.connected is False and collector.state()["active"] is False


def test_start_reports_a_locked_previous_collection_and_keeps_it_whole(tmp_path, monkeypatch):
    collector, factory, _ = _collector(tmp_path)
    collector.start("broker", 1883, ["t"])
    factory.last.emit(payload(0, pos=111.0))
    collector.stop()

    def locked(self, target):
        raise PermissionError("다른 프로그램이 쓰는 중")

    monkeypatch.setattr("pathlib.Path.rename", locked)

    with pytest.raises(StudioError, match="옮기지 못했습니다"):
        collector.start("broker", 1883, ["t"])

    assert b"111.0" in (tmp_path / "collections" / "nx5" / "current" / "capture.jsonl").read_bytes()
    archive = tmp_path / "collections" / "nx5" / "archive"
    assert factory.last.connected is False and not (archive.exists() and list(archive.iterdir()))


def test_discard_reports_a_failure_instead_of_silently_deleting_part(tmp_path, monkeypatch):
    collector, factory, _ = _collector(tmp_path)
    collector.start("broker", 1883, ["t"])
    factory.last.emit(payload(0, pos=1.0))
    collector.stop()

    def refuse(path, *args, **kwargs):
        raise PermissionError("사용 중")

    monkeypatch.setattr("shutil.rmtree", refuse)

    with pytest.raises(StudioError, match="지우지 못했습니다"):
        collector.discard()


def test_import_refuses_the_workspace_collection_itself_and_normalizes_file_names(tmp_path):
    collector, factory, _ = _collector(tmp_path / "ws")
    collector.start("broker", 1883, ["t"])
    factory.last.emit(payload(0, pos=1.0))
    collector.stop()
    with pytest.raises(StudioError, match="작업 폴더"):
        collector.import_folder(str(tmp_path / "ws" / "collections" / "nx5" / "current"))
    assert collector.state()["has_data"] is True                # 거절했으니 데이터가 그대로다

    source = tmp_path / "saved"
    source.mkdir()
    (source / "DX_Upper.JSON").write_bytes(payload(0, pos=2.0) + b"\n")
    (source / "folder.json").mkdir()                            # 이름만 .json인 폴더는 건너뛴다
    collector.import_folder(str(source))

    current = tmp_path / "ws" / "collections" / "nx5" / "current"
    assert (current / "DX_Upper.json").exists() and not (current / "folder.json").exists()


def test_the_cycle_count_is_computed_in_the_background_after_a_restart(tmp_path):
    collector, factory, _ = _collector(tmp_path)
    collector.start("broker", 1883, ["t"])
    for i in range(1200):
        factory.last.emit(_cycle_payload(i))
    collector.stop()
    reopened, _, _ = _collector(tmp_path)

    first = reopened.state()
    assert first["cycles"] is None and first["counting"] is True          # 기다리지 않고 바로 돌려준다

    done = _wait_for_cycles(reopened)
    assert done["cycles"]["cycles"] == 12 and done["counting"] is False and done["seconds"] > 0
    reopened.set_equipment("nx5")                                          # 같은 장비를 다시 저장해도 다시 세지 않는다
    assert reopened.state()["cycles"] is not None


def test_a_stale_background_scan_does_not_replace_newer_data(tmp_path):
    collector, factory, _ = _collector(tmp_path)
    collector.start("broker", 1883, ["t"])
    factory.last.emit(payload(0, pos=1.0))
    collector.stop()
    reopened, _, _ = _collector(tmp_path)
    reopened.state()                                           # 훑기 시작
    reopened.discard()                                         # 그 사이 데이터를 지운다

    import time

    time.sleep(0.5)
    assert reopened.state()["has_data"] is False and reopened.state()["cycles"] is None
