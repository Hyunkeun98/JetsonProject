from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import ConfigError, load_equipment_config
from .timeparse import parse_dx1_timestamp

DEFAULT_RESAMPLE_INTERVAL_MS = 100
DEFAULT_MAX_LATENESS_MS = 500
DEFAULT_WINDOW_SIZE = 30
DEFAULT_MIN_SAMPLES = 3000
GENERAL_GROUP = "general"
_STATE_MARKS = {"y", "yes", "true", "1", "x", "o"}


class ToolError(Exception):
    """사용자 입력/검증 오류. main()이 메시지를 출력하고 종료 코드로 바꾼다."""

    def __init__(self, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


@dataclass(frozen=True)
class Row:
    variable: str
    topic: str
    description: str
    tag: str
    group: str
    state: bool


@dataclass
class Options:
    equipment_id: str
    collector_prefix: str = ""
    group_regex: str | None = None
    group_template: str = "{}"
    state_pattern: str | None = None


@dataclass
class SampleInfo:
    keys: set
    intervals_ms: list
    distinct: dict


# --- 엑셀 읽기 -------------------------------------------------------------


def _norm(header: object) -> str:
    return str(header or "").strip().lower()


def read_rows(path: str | Path) -> list[Row]:
    try:
        import openpyxl
    except ImportError:
        raise ToolError("openpyxl이 필요합니다: `uv sync --extra tools` 로 설치하세요.", 2)
    try:
        workbook = openpyxl.load_workbook(str(path), data_only=True, read_only=True)
    except (OSError, ValueError) as e:
        raise ToolError(f"엑셀을 열 수 없습니다: {path} ({e})", 2)
    try:
        rows_iter = workbook.worksheets[0].iter_rows(values_only=True)
        try:
            header = next(rows_iter)
        except StopIteration:
            raise ToolError("엑셀이 비어 있습니다.", 2)
        index: dict[str, int] = {}
        for position, name in enumerate(header):
            index.setdefault(_norm(name), position)

        def column(*names: str) -> int | None:
            for name in names:
                if name in index:
                    return index[name]
            return None

        c_variable = column("variables", "variable")
        c_topic = column("mqtt topic", "topic")
        if c_variable is None or c_topic is None:
            raise ToolError("필수 열이 없습니다: Variables, MQTT Topic", 2)
        c_description = column("description")
        c_tag = column("tag")
        c_group = column("group")
        c_state = column("statetag", "state tag")

        def cell(raw: tuple, position: int | None) -> str:
            if position is None or position >= len(raw) or raw[position] is None:
                return ""
            return str(raw[position]).strip()

        rows: list[Row] = []
        for raw in rows_iter:
            variable = cell(raw, c_variable)
            if not variable:
                continue  # 빈 행, 변수 이름이 없는 메모 행
            topic = cell(raw, c_topic)
            if not topic:
                raise ToolError(f"토픽이 비어 있습니다: {variable}", 2)
            rows.append(
                Row(
                    variable=variable,
                    topic=topic,
                    description=cell(raw, c_description),
                    tag=cell(raw, c_tag),
                    group=cell(raw, c_group),
                    state=cell(raw, c_state).lower() in _STATE_MARKS,
                )
            )
        return rows
    finally:
        workbook.close()


# --- 태그/그룹 만들기 --------------------------------------------------------


def _clean_variable(variable: str, warnings: list[str]) -> str:
    cleaned = variable.rstrip("[ ")
    if cleaned != variable:
        warnings.append(f"변수 이름 '{variable}' 끝의 잘못된 문자를 제거했습니다 -> '{cleaned}'")
    return cleaned


def _make_tag(row: Row, cleaned_variable: str, prefix: str) -> str:
    if row.tag:
        return row.tag
    suffix = row.topic.rstrip("/").split("/")[-1]
    collector = f"{prefix}_{suffix}" if prefix else suffix
    return f"{collector}:{cleaned_variable.replace('.', '_')}"


def build_groups(
    rows: list[Row], options: Options, warnings: list[str]
) -> tuple[dict[str, dict], list[str]]:
    """그룹 이름 -> {"state_tag": str | None, "tags": [...]} (첫 등장 순서)와 구독 토픽 목록."""
    groups: dict[str, dict] = {}
    topics: list[str] = []
    seen_tags: set[str] = set()
    for row in rows:
        cleaned = _clean_variable(row.variable, warnings)
        tag = _make_tag(row, cleaned, options.collector_prefix)
        if tag in seen_tags:
            raise ToolError(f"태그가 중복됩니다: {tag}")
        seen_tags.add(tag)
        if row.topic not in topics:
            topics.append(row.topic)

        captured: str | None = None
        if row.group:
            name = row.group
        elif options.group_regex:
            match = re.search(options.group_regex, row.description)
            if match:
                captured = match.group(1) if match.groups() else match.group(0)
                name = options.group_template.format(captured)
            else:
                name = GENERAL_GROUP
        else:
            name = GENERAL_GROUP

        is_state = row.state
        if not is_state and options.state_pattern and captured is not None:
            is_state = options.state_pattern.format(captured) == cleaned

        group = groups.setdefault(name, {"state_tag": None, "tags": []})
        group["tags"].append(tag)
        if is_state:
            if group["state_tag"] is not None:
                raise ToolError(
                    f"그룹 '{name}'에 상태 태그가 둘 이상입니다: {group['state_tag']}, {tag}"
                )
            group["state_tag"] = tag
    return groups, topics


# --- 샘플 JSON -----------------------------------------------------------------


def scan_samples(path: str | Path) -> SampleInfo:
    root = Path(path)
    if root.is_dir():
        files = sorted(root.glob("*.json")) + sorted(root.glob("*.jsonl"))
    elif root.is_file():
        files = [root]
    else:
        raise ToolError(f"샘플 경로를 찾을 수 없습니다: {path}", 2)
    if not files:
        raise ToolError(f"샘플 파일(*.json, *.jsonl)이 없습니다: {path}", 2)

    keys: set[str] = set()
    distinct: dict[str, set] = {}
    intervals_ms: list[float] = []
    for file in files:
        previous: int | None = None
        with file.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except ValueError:
                    continue
                records = data.get("records") if isinstance(data, dict) else None
                if not isinstance(records, list):
                    continue
                for record in records:
                    if not isinstance(record, dict):
                        continue
                    for key, value in record.items():
                        if key == "timestamp":
                            continue
                        keys.add(key)
                        seen = distinct.setdefault(key, set())
                        if len(seen) < 2:
                            seen.add(value if isinstance(value, (int, float, str, bool)) else repr(value))
                    stamp = parse_dx1_timestamp(record.get("timestamp"))
                    if stamp is not None:
                        if previous is not None and stamp > previous:
                            intervals_ms.append((stamp - previous) / 1e6)
                        previous = stamp
    return SampleInfo(keys=keys, intervals_ms=intervals_ms, distinct=distinct)


def suggest_interval_ms(intervals_ms: list[float]) -> int | None:
    if not intervals_ms:
        return None
    return max(10, int(round(statistics.median(intervals_ms) / 10.0)) * 10)


# --- 메인 --------------------------------------------------------------------


def _build_document(
    options: Options,
    groups: dict[str, dict],
    topics: list[str],
    resample_interval_ms: int,
    max_lateness_ms: int,
    window_size: int,
    min_samples: int,
) -> dict:
    group_docs: dict[str, dict] = {}
    for name, group in groups.items():
        body: dict = {}
        if group["state_tag"] is not None:
            body["state_tag"] = group["state_tag"]
        body["tags"] = list(group["tags"])
        group_docs[name] = body
    return {
        "equipment_id": options.equipment_id,
        "mqtt": {
            "subscribe_topics": list(topics),
            "publish_topic": f"jetson/{options.equipment_id}/anomaly",
            "command_topic": f"jetson/{options.equipment_id}/cmd",
        },
        "groups": group_docs,
        "resample_interval_ms": resample_interval_ms,
        "max_lateness_ms": max_lateness_ms,
        "window_size": window_size,
        "calibration": {"max_duration": "7d", "min_samples": min_samples},
    }


def _validate_yaml(text: str) -> None:
    """생성한 YAML이 실제 로더로 읽히는지 확인한다(임시 파일 사용)."""
    handle, temp_name = tempfile.mkstemp(suffix=".yaml")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        load_equipment_config(temp_name)
    except ConfigError as e:
        raise ToolError(f"생성한 설정이 올바르지 않습니다: {e}")
    finally:
        os.remove(temp_name)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="jetson-datalist",
        description="사용자가 적은 데이터 목록(엑셀)에서 설비 설정(YAML) 초안을 만든다. "
        "태그는 목록에 적힌 것만 사용하며, 샘플 JSON은 검증과 권장값 계산에만 쓴다.",
    )
    parser.add_argument("datalist", help="데이터 목록 엑셀 경로 (열: Variables, MQTT Topic, ...)")
    parser.add_argument("--equipment-id", required=True)
    parser.add_argument("--collector-prefix", default="", help="Tag 열이 없을 때 태그 이름 접두사(예: NX5)")
    parser.add_argument("--output", help="생성할 YAML 경로(없으면 표준 출력)")
    parser.add_argument("--samples", help="검증에 쓸 샘플 JSON 파일 또는 폴더")
    parser.add_argument("--allow-missing", action="store_true", help="샘플에 없는 태그를 오류 대신 경고로")
    parser.add_argument("--group-regex", help="Group 열이 없을 때 Description에서 그룹을 찾는 정규식(첫 캡처 사용)")
    parser.add_argument("--group-template", default="{}", help="그룹 이름 형식(예: process_{})")
    parser.add_argument("--state-pattern", help="그룹의 상태 태그 변수 이름 형식(예: U{}_ProcStart)")
    parser.add_argument("--resample-interval-ms", type=int)
    parser.add_argument("--max-lateness-ms", type=int, default=DEFAULT_MAX_LATENESS_MS)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    # YAML을 표준 출력으로 내보낼 때는 안내/경고가 섞이지 않도록 표준 오류로 보낸다.
    say_stream = sys.stdout if args.output else sys.stderr

    def say(message: str) -> None:
        print(message, file=say_stream)

    try:
        options = Options(
            equipment_id=args.equipment_id,
            collector_prefix=args.collector_prefix,
            group_regex=args.group_regex,
            group_template=args.group_template,
            state_pattern=args.state_pattern,
        )
        rows = read_rows(args.datalist)
        if not rows:
            raise ToolError("목록에 변수가 없습니다.", 2)
        warnings: list[str] = []
        groups, topics = build_groups(rows, options, warnings)
        tags = [tag for group in groups.values() for tag in group["tags"]]
        for warning in warnings:
            say(f"경고: {warning}")

        suggested: int | None = None
        if args.samples:
            info = scan_samples(args.samples)
            missing = [tag for tag in tags if tag not in info.keys]
            if missing:
                message = f"목록에는 있지만 샘플에 없는 태그 {len(missing)}개: {', '.join(missing)}"
                if not args.allow_missing:
                    raise ToolError(message)
                say(f"경고: {message}")
            extra = sorted(info.keys - set(tags))
            if extra:
                say(
                    f"참고: 샘플에는 있지만 목록에 없는 태그 {len(extra)}개(설정에 넣지 않음): "
                    + ", ".join(extra)
                )
            constant = [t for t in tags if t in info.distinct and len(info.distinct[t]) <= 1]
            if constant:
                say(
                    f"경고: 샘플 동안 값이 변하지 않은 태그 {len(constant)}개"
                    "(정지 상태 샘플일 수 있습니다): " + ", ".join(constant)
                )
            suggested = suggest_interval_ms(info.intervals_ms)

        if args.resample_interval_ms is not None:
            interval = args.resample_interval_ms
        elif suggested is not None:
            interval = suggested
            say(
                f"권장 resample_interval_ms={suggested} (샘플 record 간격의 중앙값) "
                "-> 설정의 기본값으로 사용했습니다. 필요하면 YAML에서 고치세요."
            )
        else:
            interval = DEFAULT_RESAMPLE_INTERVAL_MS

        document = _build_document(
            options, groups, topics, interval, args.max_lateness_ms, args.window_size, args.min_samples
        )
        text = yaml.safe_dump(document, sort_keys=False, allow_unicode=True, default_flow_style=False)
        text = f"# jetson-datalist로 생성한 설정 초안: {Path(args.datalist).name}\n" + text
        _validate_yaml(text)
    except ToolError as e:
        print(f"오류: {e}", file=sys.stderr)
        return e.exit_code

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        say(f"설정을 만들었습니다: {output} (그룹 {len(groups)}개, 태그 {len(tags)}개)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
