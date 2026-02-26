import base64
import io
import json
import logging

import requests
from PIL import Image

logger = logging.getLogger(__name__)


class JobError(Exception):
    pass


def parse_job(raw: str) -> dict:
    """Parse and validate a raw JSON job message from the queue."""
    try:
        job = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        raise JobError(f"Invalid JSON: {e}")

    if not isinstance(job, dict):
        raise JobError("Job must be a JSON object")

    job_id = job.get("job_id")
    if not job_id:
        raise JobError("Missing required field: job_id")

    prompt = job.get("prompt")
    if not prompt:
        raise JobError("Missing required field: prompt")

    return job


def load_input_image(job: dict) -> Image.Image:
    """Load the input image from URL or base64. Returns an RGB PIL Image."""
    image_url = job.get("image_url")
    image_base64 = job.get("image_base64")

    if not image_url and not image_base64:
        raise JobError("I2V requires either 'image_url' or 'image_base64'")

    if image_url:
        try:
            resp = requests.get(image_url, timeout=60)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content))
        except requests.RequestException as e:
            raise JobError(f"Failed to download image from URL: {e}")
        except Exception as e:
            raise JobError(f"Failed to open downloaded image: {e}")
    else:
        try:
            data = base64.b64decode(image_base64)
            img = Image.open(io.BytesIO(data))
        except Exception as e:
            raise JobError(f"Failed to decode base64 image: {e}")

    return img.convert("RGB")
