"""Wan2GP Worker — Redis queue consumer for I2V video generation."""

import logging
import signal
import sys
import time

import torch

from .config import WorkerConfig
from .job import JobError, load_input_image, parse_job
from .redis_queue import RedisQueue
from .storage import S3Storage
from .webhook import send_webhook
from .wgp_bridge import bootstrap_wgp, build_task, create_state, preload_model, run_generation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("worker")

shutdown_requested = False


def _handle_signal(signum, _frame):
    global shutdown_requested
    sig_name = signal.Signals(signum).name
    logger.info("Received %s — shutting down after current job...", sig_name)
    shutdown_requested = True


def main():
    global shutdown_requested

    config = WorkerConfig()
    logger.info("Worker %s starting", config.worker_id)
    logger.info("Model: %s | Profile: %s | Attention: %s", config.model_type, config.mmgp_profile, config.attention_mode)

    # --- Bootstrap wgp.py ---
    bootstrap_wgp(config)

    # --- Preload model ---
    preload_model(config.model_type)

    # --- Init external services ---
    queue = RedisQueue(config)
    storage = S3Storage(config)

    if not queue.ping():
        logger.error("Cannot connect to Redis at %s", config.redis_url)
        sys.exit(1)
    logger.info("Redis connected: %s", config.redis_url)

    queue.start_heartbeat()

    # --- Signal handlers ---
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    logger.info("Listening on queue: %s", config.redis_queue_key)

    # --- Main loop ---
    while not shutdown_requested:
        raw = queue.poll_job(timeout=config.brpop_timeout)
        if raw is None:
            continue

        job_id = None
        start_time = time.time()

        try:
            job = parse_job(raw)
            job_id = job["job_id"]
            webhook_url = job.get("webhook_url")
            job_params = job.get("params", {})

            logger.info("[%s] Processing job — prompt: %.60s", job_id, job["prompt"])
            queue.report_progress(job_id, "processing")

            # Load input image
            image = load_input_image(job)

            # Build task and state
            task = build_task(job_params, job["prompt"], image)
            state = create_state()

            # Generate
            output_files, success = run_generation(task, state)

            duration = time.time() - start_time

            if success and output_files:
                # Upload to S3
                s3_key = storage.upload_video(output_files[0], job_id)
                video_url = storage.get_public_url(s3_key)

                result = {
                    "job_id": job_id,
                    "status": "completed",
                    "video_url": video_url,
                    "s3_key": s3_key,
                    "duration_seconds": round(duration, 2),
                    "error": None,
                }
                logger.info("[%s] Completed in %.1fs — %s", job_id, duration, s3_key)
            else:
                result = {
                    "job_id": job_id,
                    "status": "failed",
                    "video_url": None,
                    "s3_key": None,
                    "duration_seconds": round(duration, 2),
                    "error": "Generation returned no output",
                }
                logger.warning("[%s] Generation failed (no output)", job_id)

            queue.report_result(job_id, result)
            queue.report_progress(job_id, result["status"])
            send_webhook(webhook_url, result)

        except JobError as e:
            duration = time.time() - start_time
            logger.warning("[%s] Job error: %s", job_id or "?", e)
            _report_failure(queue, job_id, str(e), duration, raw)

        except torch.cuda.OutOfMemoryError as e:
            duration = time.time() - start_time
            logger.error("[%s] CUDA OOM: %s", job_id or "?", e)
            torch.cuda.empty_cache()
            _report_failure(queue, job_id, str(e), duration, raw)

        except Exception as e:
            duration = time.time() - start_time
            logger.exception("[%s] Unexpected error", job_id or "?")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            _report_failure(queue, job_id, str(e), duration, raw)

    # --- Shutdown ---
    logger.info("Shutting down...")
    queue.stop_heartbeat()
    logger.info("Worker %s stopped", config.worker_id)


def _report_failure(queue: RedisQueue, job_id, error_msg: str, duration: float, raw: str):
    """Best-effort failure reporting — must never raise."""
    result = {
        "job_id": job_id,
        "status": "failed",
        "video_url": None,
        "s3_key": None,
        "duration_seconds": round(duration, 2),
        "error": error_msg,
    }
    try:
        if job_id:
            queue.report_result(job_id, result)
            queue.report_progress(job_id, "failed")
        # Try to extract webhook_url from raw message for error callback
        import json
        job_data = json.loads(raw) if isinstance(raw, str) else {}
        webhook_url = job_data.get("webhook_url")
        if webhook_url:
            send_webhook(webhook_url, result)
    except Exception:
        pass


if __name__ == "__main__":
    main()
