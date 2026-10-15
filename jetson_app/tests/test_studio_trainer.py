import pytest

from jetson_app.calibration import CalibrationState, StateStore
from jetson_app.deviation import deviation_artifact_path
from jetson_app.artifact_paths import state_marker_path
from jetson_app.studio import wizard
from jetson_app.studio.errors import StudioError
from jetson_app.studio.trainer import train_from_collection
from studio_fakes import noise, payload


def _config(tmp_path, tags=("pos",)):
    wizard.save_connection(
        tmp_path / "ws",
        {"equipment_id": "nx5", "host": "localhost", "port": 1883, "subscribe_topics": ["dx1/data"]},
    )
    return wizard.save_config(tmp_path / "ws", list(tags))


def _write(directory, build):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "capture.jsonl").write_bytes(b"\n".join(build(i) for i in range(400)) + b"\n")


def test_training_from_a_collection_writes_the_model_and_marks_monitoring(tmp_path):
    config = _config(tmp_path)
    collection = tmp_path / "collection"
    _write(collection, lambda i: payload(i, pos=noise(i), other=5.0))

    steps = train_from_collection(config, collection, tmp_path / "models")

    assert steps > 300
    assert deviation_artifact_path(tmp_path / "models", "nx5").exists()
    assert StateStore(state_marker_path(tmp_path / "models", "nx5")).read() == CalibrationState.MONITORING


def test_training_on_data_that_never_changes_explains_why(tmp_path):
    config = _config(tmp_path)
    collection = tmp_path / "collection"
    _write(collection, lambda i: payload(i, pos=1.0))

    with pytest.raises(ValueError, match="변하지"):
        train_from_collection(config, collection, tmp_path / "models")

    assert not deviation_artifact_path(tmp_path / "models", "nx5").exists()


def test_training_when_the_chosen_tag_is_not_in_the_collection(tmp_path):
    config = _config(tmp_path, tags=("missing",))
    collection = tmp_path / "collection"
    _write(collection, lambda i: payload(i, pos=noise(i)))

    with pytest.raises(StudioError, match="missing"):
        train_from_collection(config, collection, tmp_path / "models")
