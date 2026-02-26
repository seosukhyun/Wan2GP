# Wan2GP Worker 테스트 가이드

---

## 목차

1. [사전 준비](#1-사전-준비)
2. [단계별 테스트](#2-단계별-테스트)
   - 2.1 개별 모듈 테스트 (GPU 불필요)
   - 2.2 wgp_bridge 부트스트랩 테스트 (GPU 필요)
   - 2.3 로컬 통합 테스트 (GPU + Redis + MinIO)
   - 2.4 Docker 통합 테스트
3. [Job 제출 방법](#3-job-제출-방법)
4. [검증 체크리스트](#4-검증-체크리스트)
5. [트러블슈팅](#5-트러블슈팅)

---

## 1. 사전 준비

### 필수 패키지 설치

```bash
pip install redis boto3 requests
```

### 로컬 Redis 실행 (Docker)

```bash
docker run -d --name redis-test -p 6379:6379 redis:7-alpine
```

### 로컬 MinIO 실행 (Docker)

```bash
docker run -d --name minio-test \
  -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio server /data --console-address ":9001"
```

MinIO 콘솔 확인: http://localhost:9001 (minioadmin / minioadmin)

---

## 2. 단계별 테스트

### 2.1 개별 모듈 테스트 (GPU 불필요)

GPU 없이도 Redis, S3, Webhook, Job 파싱 모듈을 각각 검증할 수 있다.

#### config.py 테스트

```python
import os
os.environ["REDIS_URL"] = "redis://localhost:6379/0"
os.environ["MODEL_TYPE"] = "i2v_2_2"

from worker.config import WorkerConfig
config = WorkerConfig()
print(f"worker_id: {config.worker_id}")
print(f"model_type: {config.model_type}")
print(f"redis_url: {config.redis_url}")
```

#### job.py 테스트

```python
from worker.job import parse_job, load_input_image, JobError

# 정상 케이스
job = parse_job('{"job_id": "test-1", "prompt": "a cat", "image_url": "https://picsum.photos/832/480"}')
print(f"Parsed: {job['job_id']}")

# 이미지 로드
img = load_input_image(job)
print(f"Image size: {img.size}, mode: {img.mode}")

# 에러 케이스 — job_id 누락
try:
    parse_job('{"prompt": "hello"}')
except JobError as e:
    print(f"Expected error: {e}")

# 에러 케이스 — 잘못된 JSON
try:
    parse_job("not json")
except JobError as e:
    print(f"Expected error: {e}")
```

#### redis_queue.py 테스트

```bash
# 터미널 1: Redis가 실행중인지 확인
redis-cli ping
# → PONG
```

```python
from worker.config import WorkerConfig
from worker.redis_queue import RedisQueue

config = WorkerConfig()
queue = RedisQueue(config)

# 연결 테스트
print(f"Ping: {queue.ping()}")

# 하트비트 시작/중지
queue.start_heartbeat()
import time; time.sleep(3)
queue.stop_heartbeat()

# 결과 저장 테스트
queue.report_result("test-job", {"status": "completed", "video_url": "http://example.com"})
```

```bash
# 터미널에서 결과 확인
redis-cli GET wan2gp:results:test-job
```

#### storage.py 테스트

```python
import tempfile, os
from worker.config import WorkerConfig
from worker.storage import S3Storage

os.environ["S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["S3_ACCESS_KEY"] = "minioadmin"
os.environ["S3_SECRET_KEY"] = "minioadmin"

config = WorkerConfig()
storage = S3Storage(config)

# 테스트 파일 업로드
with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
    f.write(b"\x00" * 1024)  # 더미 파일
    tmp_path = f.name

s3_key = storage.upload_video(tmp_path, "test-job-123")
url = storage.get_public_url(s3_key)
print(f"S3 key: {s3_key}")
print(f"URL: {url}")
os.unlink(tmp_path)
```

MinIO 콘솔(http://localhost:9001)에서 `wan2gp-outputs` 버킷 → `videos/test-job-123/` 확인.

#### webhook.py 테스트

https://webhook.site 에 접속하여 고유 URL을 받은 뒤:

```python
from worker.webhook import send_webhook

url = "https://webhook.site/YOUR-UNIQUE-URL"  # ← 교체
ok = send_webhook(url, {
    "job_id": "test-1",
    "status": "completed",
    "video_url": "https://example.com/video.mp4",
})
print(f"Webhook delivered: {ok}")
```

webhook.site 페이지에서 수신된 POST 요청 확인.

---

### 2.2 wgp_bridge 부트스트랩 테스트 (GPU 필요)

이 테스트는 실제 GPU가 있는 환경에서 실행해야 한다.

```python
import os
os.environ["WAN2GP_ROOT"] = "/workspace"          # Wan2GP 프로젝트 경로
os.environ["OUTPUT_DIR"] = "/workspace/outputs"
os.environ["MMGP_PROFILE"] = "3"
os.environ["ATTENTION_MODE"] = "sage2"

from worker.config import WorkerConfig
from worker.wgp_bridge import bootstrap_wgp, preload_model, create_state, build_task

config = WorkerConfig()

# 1단계: 부트스트랩 (wgp.py import)
bootstrap_wgp(config)
print("Bootstrap OK")

# 2단계: 모델 프리로드
preload_model("i2v_2_2")
print("Model preloaded OK")

# 3단계: state 생성
state = create_state()
print(f"State keys: {list(state.keys())}")

# 4단계: task 빌드 (더미 이미지)
from PIL import Image
dummy_img = Image.new("RGB", (832, 480), color=(128, 128, 128))
task = build_task(
    job_params={"video_length": 81, "num_inference_steps": 2},  # steps=2 로 빠른 테스트
    prompt="a gray screen test",
    image_start=dummy_img,
)
print(f"Task prompt: {task['prompt']}")
print(f"Task model_type: {task['params']['model_type']}")
```

`num_inference_steps=2`로 설정하면 실제 영상 품질은 의미없지만 파이프라인 전체가 동작하는지 빠르게 확인 가능.

---

### 2.3 로컬 통합 테스트 (GPU + Redis + MinIO)

Redis와 MinIO가 실행중인 GPU 머신에서 워커를 직접 실행한다.

#### 워커 시작

```bash
cd /workspace   # Wan2GP 루트

export REDIS_URL=redis://localhost:6379/0
export S3_ENDPOINT_URL=http://localhost:9000
export S3_BUCKET=wan2gp-outputs
export S3_ACCESS_KEY=minioadmin
export S3_SECRET_KEY=minioadmin
export WAN2GP_ROOT=/workspace
export OUTPUT_DIR=/workspace/outputs
export MODEL_TYPE=i2v_2_2
export MMGP_PROFILE=3

python -m worker.main
```

모델 로딩이 끝나면 `Listening on queue: wan2gp:jobs` 메시지가 출력된다.

#### Job 제출 (별도 터미널)

```bash
redis-cli LPUSH wan2gp:jobs '{
  "job_id": "test-001",
  "prompt": "a cat walking slowly in the rain, cinematic lighting",
  "image_url": "https://picsum.photos/832/480",
  "webhook_url": "",
  "params": {
    "resolution": "832x480",
    "video_length": 81,
    "num_inference_steps": 30,
    "seed": 42
  }
}'
```

#### 결과 확인

```bash
# 처리 완료 대기 후...
redis-cli GET wan2gp:results:test-001

# 출력 파일 확인
ls -la /workspace/outputs/

# S3 업로드 확인 (MinIO CLI)
docker exec minio-test mc alias set local http://localhost:9000 minioadmin minioadmin
docker exec minio-test mc ls local/wan2gp-outputs/videos/test-001/
```

---

### 2.4 Docker 통합 테스트

#### 사전 조건

- NVIDIA Container Toolkit 설치 (`nvidia-docker`)
- Wan2GP 기본 Docker 이미지 빌드 완료 (`wan2gp:latest`)

#### 실행

```bash
cd worker/
docker compose up --build
```

세 서비스가 순서대로 시작된다:
1. `redis` — 즉시 준비
2. `minio` — 즉시 준비
3. `worker` — 모델 로딩 후 대기 (수 분 소요)

#### Job 제출

```bash
# 호스트에서 Redis에 직접 제출
redis-cli -p 6379 LPUSH wan2gp:jobs '{
  "job_id": "docker-test-001",
  "prompt": "a serene mountain lake at sunset",
  "image_url": "https://picsum.photos/832/480",
  "params": {
    "video_length": 81,
    "num_inference_steps": 30
  }
}'
```

#### 로그 확인

```bash
docker compose logs -f worker
```

#### 결과 확인

```bash
redis-cli -p 6379 GET wan2gp:results:docker-test-001
```

MinIO 콘솔: http://localhost:9001

#### Graceful Shutdown 테스트

```bash
# Job 처리 중에 실행
docker compose stop worker

# 로그에서 "Shutting down after current job..." 확인
# 현재 Job 완료 후 종료되는지 확인
```

---

## 3. Job 제출 방법

### Python 클라이언트 예시

```python
import json
import uuid
import redis

r = redis.Redis(host="localhost", port=6379, db=0)

job = {
    "job_id": str(uuid.uuid4()),
    "prompt": "a dog playing in the park, golden hour",
    "image_url": "https://picsum.photos/832/480",
    "webhook_url": "https://webhook.site/YOUR-URL",   # 선택
    "params": {
        "resolution": "832x480",
        "video_length": 81,
        "num_inference_steps": 30,
        "guidance_scale": 3.5,
        "seed": -1,                                    # -1 = 랜덤
    }
}

r.lpush("wan2gp:jobs", json.dumps(job))
print(f"Submitted: {job['job_id']}")

# 결과 폴링
import time
for _ in range(300):  # 최대 5분 대기
    result = r.get(f"wan2gp:results:{job['job_id']}")
    if result:
        print(json.loads(result))
        break
    time.sleep(1)
```

### base64 이미지로 제출

```python
import base64

with open("input_image.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

job = {
    "job_id": str(uuid.uuid4()),
    "prompt": "the person starts to dance",
    "image_base64": b64,
    "params": {"video_length": 81}
}

r.lpush("wan2gp:jobs", json.dumps(job))
```

### 진행상황 구독 (PubSub)

```python
import redis, json

r = redis.Redis(host="localhost", port=6379, db=0)
ps = r.pubsub()
ps.subscribe(f"wan2gp:progress:{job_id}")

for msg in ps.listen():
    if msg["type"] == "message":
        data = json.loads(msg["data"])
        print(f"Status: {data['status']}")
        if data["status"] in ("completed", "failed"):
            break
```

---

## 4. 검증 체크리스트

| # | 항목 | 확인 방법 |
|---|------|-----------|
| 1 | 워커 시작 시 모델 로딩 완료 | 로그에 `Model preloaded successfully` |
| 2 | Redis 연결 | 로그에 `Redis connected` |
| 3 | 하트비트 동작 | `redis-cli GET wan2gp:workers:{worker_id}` → `alive` |
| 4 | Job 수신 및 처리 | `LPUSH` 후 로그에 `Processing job` |
| 5 | 영상 생성 | `outputs/` 폴더에 `.mp4` 파일 생성 |
| 6 | S3 업로드 | MinIO 콘솔에서 파일 확인 |
| 7 | 결과 저장 | `redis-cli GET wan2gp:results:{job_id}` |
| 8 | Webhook 발송 | webhook.site에서 POST 수신 확인 |
| 9 | 잘못된 Job 에러 처리 | `LPUSH` 잘못된 JSON → 워커가 크래시 없이 에러 로그 |
| 10 | 잘못된 이미지 URL | 존재하지 않는 URL → `status: failed` 결과 |
| 11 | Graceful shutdown | `Ctrl+C` 또는 `SIGTERM` → 현재 Job 완료 후 종료 |
| 12 | CUDA OOM 복구 | (해상도를 극단적으로 높여서 재현) → `empty_cache` 후 다음 Job 계속 |

### 에러 케이스 테스트

```bash
# 빈 JSON
redis-cli LPUSH wan2gp:jobs '{}'

# job_id 누락
redis-cli LPUSH wan2gp:jobs '{"prompt": "test"}'

# 잘못된 이미지 URL
redis-cli LPUSH wan2gp:jobs '{
  "job_id": "err-1",
  "prompt": "test",
  "image_url": "https://invalid.example.com/nothing.jpg"
}'

# JSON이 아닌 문자열
redis-cli LPUSH wan2gp:jobs 'this is not json'
```

모든 케이스에서 워커가 에러를 로그에 남기고 다음 Job을 계속 처리하는지 확인.

---

## 5. 트러블슈팅

### `ModuleNotFoundError: No module named 'wgp'`

`WAN2GP_ROOT` 환경변수가 `wgp.py`가 있는 디렉토리를 가리키는지 확인.
워커는 반드시 해당 디렉토리에서 실행되어야 한다.

### `CUDA out of memory`

- `MMGP_PROFILE`을 더 높은 값으로 조정 (3 = LowRAM_LowVRAM)
- 해상도를 낮추거나 `video_length`를 줄인다
- 워커는 OOM 시 `torch.cuda.empty_cache()`를 호출하고 자동 복구한다

### `redis.ConnectionError`

- Redis가 실행중인지: `redis-cli ping`
- `REDIS_URL` 환경변수 확인
- Docker 네트워크에서는 호스트명이 `redis`인지 확인

### S3 업로드 실패

- MinIO가 실행중인지: `curl http://localhost:9000/minio/health/live`
- `S3_ENDPOINT_URL`, `S3_ACCESS_KEY`, `S3_SECRET_KEY` 확인
- Docker 네트워크에서는 엔드포인트가 `http://minio:9000`인지 확인

### 워커가 Job을 받지 못함

```bash
# 큐에 Job이 있는지 확인
redis-cli LLEN wan2gp:jobs

# 큐 키 이름이 일치하는지 확인 (기본: wan2gp:jobs)
redis-cli KEYS "wan2gp:*"
```

### 생성된 영상이 S3에 없음

```bash
# 로컬 출력 파일 확인
ls -la /workspace/outputs/

# 워커 로그에서 upload 성공/실패 확인
# "Uploading ... -> s3://..." 메시지 확인
```
