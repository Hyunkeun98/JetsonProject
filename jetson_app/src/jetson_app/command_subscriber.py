from __future__ import annotations

import json

from .calibration import CalibrationManager

_VALID_COMMANDS = {"train", "recalibrate"}


def parse_command(payload: bytes) -> str | None:
    try:
        data = json.loads(payload)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    command = data.get("command")
    if command not in _VALID_COMMANDS:
        return None
    return command


class CommandSubscriber:
    """MQTT의 설비 command_topic에서 train/recalibrate 명령을 받아
    CalibrationManager로 전달한다."""

    def __init__(self, command_topic: str, calibration_manager: CalibrationManager) -> None:
        self._command_topic = command_topic
        self._calibration_manager = calibration_manager

    def attach(self, client) -> None:
        client.message_callback_add(self._command_topic, self._handle_command_message)

    def _handle_command_message(self, client, userdata, msg) -> None:
        command = parse_command(msg.payload)
        if command is None:
            print(f"알 수 없는 명령 페이로드 무시: {msg.payload!r}")
            return
        try:
            if command == "train":
                self._calibration_manager.handle_train_command()
                print("학습 완료 — MONITORING 상태로 전환")
            else:
                self._calibration_manager.handle_recalibrate_command()
                print("재캘리브레이션 — CALIBRATING 상태로 재시작")
        except Exception as e:
            # 어떤 예외도 MQTT 루프를 죽여서는 안 된다.
            print(f"명령 처리 실패: {e}")
