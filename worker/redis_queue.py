import json
import logging
import threading
import time
from typing import Optional

import redis

from .config import WorkerConfig

logger = logging.getLogger(__name__)


class RedisQueue:
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.client = redis.Redis.from_url(
            config.redis_url,
            decode_responses=True,
            retry_on_timeout=True,
        )
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_stop = threading.Event()

    def ping(self) -> bool:
        try:
            return self.client.ping()
        except redis.ConnectionError:
            return False

    def poll_job(self, timeout: int = 1) -> Optional[str]:
        """Block-pop a job from the queue. Returns raw JSON string or None on timeout."""
        try:
            result = self.client.brpop(self.config.redis_queue_key, timeout=timeout)
            if result is None:
                return None
            # brpop returns (key, value)
            return result[1]
        except redis.ConnectionError:
            logger.warning("Redis connection lost during poll, will retry...")
            time.sleep(1)
            return None

    def report_progress(self, job_id: str, status: str, data: Optional[dict] = None):
        """Publish progress update to a PubSub channel."""
        payload = {"job_id": job_id, "status": status}
        if data:
            payload.update(data)
        try:
            self.client.publish(f"wan2gp:progress:{job_id}", json.dumps(payload))
        except redis.ConnectionError:
            logger.warning("Failed to publish progress for job %s", job_id)

    def report_result(self, job_id: str, result: dict, ttl: int = 3600):
        """Store the job result with a TTL."""
        try:
            self.client.setex(f"wan2gp:results:{job_id}", ttl, json.dumps(result))
        except redis.ConnectionError:
            logger.warning("Failed to store result for job %s", job_id)

    def start_heartbeat(self):
        """Start a daemon thread that periodically updates a heartbeat key."""
        self._heartbeat_stop.clear()

        def _heartbeat_loop():
            key = f"wan2gp:workers:{self.config.worker_id}"
            ttl = self.config.heartbeat_interval * 3
            while not self._heartbeat_stop.is_set():
                try:
                    self.client.setex(key, ttl, "alive")
                except redis.ConnectionError:
                    logger.warning("Heartbeat: Redis connection lost")
                self._heartbeat_stop.wait(self.config.heartbeat_interval)

        self._heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
        logger.info("Heartbeat started (interval=%ds)", self.config.heartbeat_interval)

    def stop_heartbeat(self):
        self._heartbeat_stop.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)
        try:
            self.client.delete(f"wan2gp:workers:{self.config.worker_id}")
        except redis.ConnectionError:
            pass
