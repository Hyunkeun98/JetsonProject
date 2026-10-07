# 이상 점수 그래프 화면(실시간 + 파일) 설계

- 작성일: 2026-10-07
- 상태: 사용자가 대화로 설계 승인, 구현 요청

## 1. 목적

그룹(공정/축)별 이상 점수와 알람 임계값을 선 그래프로 보여 준다. (a) 앱이 돌아가는 동안
localhost 웹 화면으로 실시간으로, (b) 저장된 `scores.csv`를 서버 없이 여는 HTML 한 장으로.
같은 화면 코드를 둘이 공유한다. 읽기 전용이다.

비목표: 로그인, 점수 이력의 디스크 저장, 알람 목록/알림, 화면에서 앱의 임계값을 바꾸는 기능,
외부 라이브러리/CDN 사용(Jetson은 인터넷이 없을 수 있다).

## 2. 구성

| 파일 | 역할 |
|---|---|
| `score_history.py` | `ScoreHistory`(스레드세이프 링 버퍼), 구간 최댓값 줄이기, `HistoryRecordingPublisher` |
| `web_server.py` | `ScoreWebServer`(표준 `http.server`, 읽기 전용 JSON API + 화면 제공) |
| `web/dashboard.html` | 화면 한 장(순수 JS + canvas). 실시간/파일 두 모드 |
| `plot_cli.py` | `jetson-plot`: 점수 CSV → 데이터를 내장한 독립 HTML |
| `pipeline.py`, `subscriber_cli.py` | 이력 기록 연결, `--web-port`/`--web-host` |

## 3. 실시간(앱 안)

- `jetson-app ... --web-port 8080`을 줄 때만 이력과 서버가 켜진다. 안 주면 코드 경로가 전혀 달라지지
  않는다(`build_pipeline(history_seconds=0)`이 기본).
- **이력**: 스텝마다 `(epoch_ms, 그룹별 (점수, 알람, 원인 태그))`를 링 버퍼에 쌓는다. 용량은
  `3600초 / resample_interval_ms`행. 재시작하면 사라진다.
- **기록 위치**: `SnapshotProcessor`에 넘기는 발행기를 `HistoryRecordingPublisher`로 감싼다. 원래
  발행을 먼저 하고, 이력 기록은 예외를 삼켜 점수/발행 경로에 영향을 주지 않는다. 그룹이 하나뿐인
  평면 설정도 그룹 `all`로 기록한다.
- **API**(GET만, 그 외 405):
  - `/` 화면, `/healthz`
  - `/api/meta`: `equipment_id`, `groups`, `threshold`, `confirm_steps`, `interval_ms`,
    `history_seconds`, `state`(`CALIBRATING`/`MONITORING`)
  - `/api/scores?seconds=S&max_points=N`: 최근 S초(10~이력 길이로 제한)를 최대 N점(50~5000)으로
    줄여서 반환. 구간(bucket)마다 그룹별 **최댓값 점수**를 남겨 짧은 튐이 사라지지 않게 하고, 구간
    안에 알람이 있었으면 알람 표시. 응답: `{now_ms, bucket_ms, points:[{t, g:{그룹:[점수, 알람, 태그]}}]}`.
- 화면은 1~2초마다 이 API를 다시 불러 전체를 다시 그린다(증분 상태가 없어 단순하고 견고하다).
- **바인딩**: 기본 `127.0.0.1`. `--web-host 0.0.0.0`이면 "인증이 없다"는 경고를 출력한다.
  응답에 `Cache-Control: no-store`, `X-Content-Type-Options: nosniff`.
- 포트가 이미 쓰이거나 열 수 없으면 메시지를 출력하고 앱은 웹 없이 계속 돈다.

## 4. 파일 보기(`jetson-plot`)

`jetson-plot --scores scores.csv [--config cfg.yaml] [--threshold T] [--confirm N] --out plot.html`.

- 점수 CSV를 읽어 HTML에 JSON으로 내장한다(`</`는 `<\/`로 이스케이프). 더블클릭으로 열린다.
- x축은 행 순서(구간을 이어 붙임)이고 구간 사이는 선을 끊고 구간 이름/종류를 배경색으로 표시한다
  (정상 검증 / 이상). 마우스를 올리면 실제 시각, 점수, 원인 태그가 보인다.
- **임계값/confirm 슬라이더**: 바꾸면 알람 구간(그룹별)과 요약("정상 알람 N건, 이상 검출 x/y")을
  즉시 다시 계산한다. 계산 규칙은 앱의 디바운서와 같다(그룹별 `점수 ≥ T`가 `confirm`번 연속,
  `reset` 행에서 초기화, 전체 알람은 그룹 알람의 OR, 연속된 알람은 1건).
- 초기값은 `--threshold`, 없으면 `--config`의 `alarm.threshold`, 없으면 3.0. 임계값 추천
  (`threshold.analyze`)이 가능하면 "추천 N 적용" 버튼을 만든다.

## 5. 화면

- 그룹마다 카드 한 칸: 제목, 현재 점수/원인 태그/알람 배지, 점수 선, 임계값 점선(값 라벨), 알람
  구간 음영, 호버 툴팁. 알람은 색과 함께 "ALARM" 글자로도 표시한다(색만으로 구분하지 않는다).
- 실시간 모드: 범위 선택(1/5/30/60분), 상단에 설비 이름·앱 상태·마지막 갱신 시각. 점수가 없으면
  상태별 안내("학습 데이터 수집 중", "연결 대기").
- 밝은/어두운 테마(`prefers-color-scheme`), 폭 좁으면 한 열. 선 끊김은 점 사이 간격이
  `max(3×bucket_ms, 2s)`를 넘거나(실시간) reset 행(파일)일 때.
- y축은 `min(0, 최소 점수)`부터 `max(임계값×1.2, 최대 점수)`까지 자동.

## 6. 테스트

- `ScoreHistory`: 용량 초과 시 오래된 것 제거, 구간 최댓값 줄이기(튐 보존, 알람 전파, 빈 구간),
  스레드 동시 기록.
- `HistoryRecordingPublisher`: 원래 발행 인자 그대로 전달, 이력 예외가 발행을 막지 않음.
- 웹 서버: 포트 0으로 띄워 `/`, `/healthz`, `/api/meta`, `/api/scores`(범위/상한 제한), 404/405,
  포트 충돌 처리.
- 파이프라인: `history_seconds=0`이면 이력 없음, 양수면 점수 스텝이 이력에 쌓임.
- `jetson-plot`: 데이터 내장, 이스케이프, 설정/추천 반영.
- **JS와 파이썬 일치**: 화면의 알람 계산 함수를 Node로 실행해 `threshold._simulate_segment`와 무작위
  입력 40건 이상에서 알람 배열·건수를 비교한다(Node가 없으면 건너뜀).
- 화면 모양은 헤드리스 브라우저가 가능하면 스크린샷으로, 아니면 사용자가 직접 열어 확인한다.
