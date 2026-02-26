import os
import uuid
from dataclasses import dataclass, field


@dataclass
class WorkerConfig:
    # Redis
    redis_url: str = field(default_factory=lambda: os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    redis_queue_key: str = field(default_factory=lambda: os.environ.get("REDIS_QUEUE_KEY", "wan2gp:jobs"))

    # S3
    s3_endpoint_url: str = field(default_factory=lambda: os.environ.get("S3_ENDPOINT_URL", ""))
    s3_bucket: str = field(default_factory=lambda: os.environ.get("S3_BUCKET", "wan2gp-outputs"))
    s3_access_key: str = field(default_factory=lambda: os.environ.get("S3_ACCESS_KEY", ""))
    s3_secret_key: str = field(default_factory=lambda: os.environ.get("S3_SECRET_KEY", ""))
    s3_region: str = field(default_factory=lambda: os.environ.get("S3_REGION", "us-east-1"))
    s3_prefix: str = field(default_factory=lambda: os.environ.get("S3_PREFIX", "videos/"))

    # Wan2GP
    wan2gp_root: str = field(default_factory=lambda: os.environ.get("WAN2GP_ROOT", "/workspace"))
    output_dir: str = field(default_factory=lambda: os.environ.get("OUTPUT_DIR", "/workspace/outputs"))
    model_type: str = field(default_factory=lambda: os.environ.get("MODEL_TYPE", "i2v_2_2"))
    mmgp_profile: str = field(default_factory=lambda: os.environ.get("MMGP_PROFILE", "3"))
    attention_mode: str = field(default_factory=lambda: os.environ.get("ATTENTION_MODE", "sage2"))
    quantization: str = field(default_factory=lambda: os.environ.get("QUANTIZATION", "int8"))

    # Worker
    worker_id: str = field(default_factory=lambda: os.environ.get("WORKER_ID", f"worker-{uuid.uuid4().hex[:8]}"))
    heartbeat_interval: int = field(default_factory=lambda: int(os.environ.get("HEARTBEAT_INTERVAL", "15")))
    brpop_timeout: int = field(default_factory=lambda: int(os.environ.get("BRPOP_TIMEOUT", "1")))
