from types import SimpleNamespace

from jetson_app.command_subscriber import CommandSubscriber, parse_command


def test_parse_command_valid_train():
    assert parse_command(b'{"command": "train"}') == "train"


def test_parse_command_valid_recalibrate():
    assert parse_command(b'{"command": "recalibrate"}') == "recalibrate"


def test_parse_command_invalid_json_returns_none():
    assert parse_command(b"not json") is None


def test_parse_command_unknown_command_returns_none():
    assert parse_command(b'{"command": "shutdown"}') is None


def test_parse_command_missing_command_key_returns_none():
    assert parse_command(b'{"foo": "bar"}') is None


class _FakeCalibrationManager:
    def __init__(self):
        self.train_calls = 0
        self.recalibrate_calls = 0
        self.raise_on_train = None

    def handle_train_command(self):
        self.train_calls += 1
        if self.raise_on_train is not None:
            raise self.raise_on_train

    def handle_recalibrate_command(self):
        self.recalibrate_calls += 1


def test_command_subscriber_handles_train_message():
    manager = _FakeCalibrationManager()
    subscriber = CommandSubscriber(command_topic="jetson/x/cmd", calibration_manager=manager)
    msg = SimpleNamespace(payload=b'{"command": "train"}')

    subscriber._handle_command_message(None, None, msg)

    assert manager.train_calls == 1
    assert manager.recalibrate_calls == 0


def test_command_subscriber_handles_recalibrate_message():
    manager = _FakeCalibrationManager()
    subscriber = CommandSubscriber(command_topic="jetson/x/cmd", calibration_manager=manager)
    msg = SimpleNamespace(payload=b'{"command": "recalibrate"}')

    subscriber._handle_command_message(None, None, msg)

    assert manager.recalibrate_calls == 1
    assert manager.train_calls == 0


def test_command_subscriber_swallows_calibration_error():
    from jetson_app.calibration import CalibrationError

    manager = _FakeCalibrationManager()
    manager.raise_on_train = CalibrationError("not enough samples")
    subscriber = CommandSubscriber(command_topic="jetson/x/cmd", calibration_manager=manager)
    msg = SimpleNamespace(payload=b'{"command": "train"}')

    subscriber._handle_command_message(None, None, msg)  # 예외가 밖으로 나오면 테스트 실패

    assert manager.train_calls == 1


def test_command_subscriber_ignores_unparseable_payload():
    manager = _FakeCalibrationManager()
    subscriber = CommandSubscriber(command_topic="jetson/x/cmd", calibration_manager=manager)
    msg = SimpleNamespace(payload=b"garbage")

    subscriber._handle_command_message(None, None, msg)

    assert manager.train_calls == 0
    assert manager.recalibrate_calls == 0
