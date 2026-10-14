import subprocess
import sys

import pytest


def _import_without_torch(*modules):
    code = "import sys\nsys.modules['torch'] = None\n" + "\n".join(f"import {m}" for m in modules)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


@pytest.mark.parametrize(
    "module",
    ["jetson_app.pipeline", "jetson_app.deviation", "jetson_app.deviation_suggest", "jetson_app.replay", "jetson_app.studio.cli"],
)
def test_deviation_chain_imports_without_torch(module):
    result = _import_without_torch(module)

    assert result.returncode == 0, result.stderr


def test_old_import_paths_still_work():
    from jetson_app.inference import ActiveModelHolder, GroupResult
    from jetson_app.training import model_artifact_path, state_marker_path
    from jetson_app import engine_types, artifact_paths

    assert ActiveModelHolder is engine_types.ActiveModelHolder
    assert GroupResult is engine_types.GroupResult
    assert model_artifact_path is artifact_paths.model_artifact_path
    assert state_marker_path is artifact_paths.state_marker_path
