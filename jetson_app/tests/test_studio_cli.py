import socket

import pytest

from jetson_app.studio import cli


def test_default_workspace_is_a_workspace_folder():
    assert cli.default_workspace().name == "workspace"


def test_main_reports_a_port_already_in_use_in_one_line(tmp_path, capsys):
    with socket.socket() as blocker:
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        port = blocker.getsockname()[1]

        with pytest.raises(SystemExit) as exc:
            cli.main(["--workspace", str(tmp_path), "--web-port", str(port), "--no-browser"])

    assert exc.value.code == 2
    assert str(port) in capsys.readouterr().err


def test_control_is_disabled_for_a_remote_host_without_a_token():
    assert cli.control_enabled("0.0.0.0", None) is False
    assert cli.control_enabled("0.0.0.0", "s3cret") is True
    assert cli.control_enabled("127.0.0.1", None) is True
    assert cli.control_enabled("localhost", None) is True


def test_default_port_is_8800(monkeypatch):
    seen = {}

    class Stop(Exception):
        pass

    def fake_server(controller, host, port, token, control_enabled):
        seen["port"] = port
        raise Stop

    monkeypatch.setattr(cli, "StudioServer", fake_server)
    monkeypatch.setattr(cli, "StudioController", lambda workspace: object())
    with pytest.raises(Stop):
        cli.main(["--workspace", "ignored", "--no-browser"])
    assert seen["port"] == 8800


def test_studio_url_percent_encodes_the_token():
    assert cli.studio_url("127.0.0.1", 8080, "a+b#c") == "http://localhost:8080/?token=a%2Bb%23c"
    assert cli.studio_url("0.0.0.0", 9000, None) == "http://0.0.0.0:9000/"
