import logging
import time

import requests

logger = logging.getLogger(__name__)

RETRY_DELAYS = [1, 5, 15]


def send_webhook(url: str, payload: dict) -> bool:
    """POST a JSON payload to the webhook URL with retries.

    Returns True if the webhook was delivered successfully, False otherwise.
    Failures are logged but never raise — the worker must keep running.
    """
    if not url:
        return False

    for attempt, delay in enumerate(RETRY_DELAYS):
        try:
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code < 400:
                logger.info("Webhook delivered to %s (status=%d)", url, resp.status_code)
                return True
            logger.warning(
                "Webhook attempt %d/%d failed: HTTP %d",
                attempt + 1, len(RETRY_DELAYS), resp.status_code,
            )
        except requests.RequestException as e:
            logger.warning(
                "Webhook attempt %d/%d failed: %s",
                attempt + 1, len(RETRY_DELAYS), e,
            )

        if attempt < len(RETRY_DELAYS) - 1:
            time.sleep(delay)

    logger.error("Webhook delivery failed after %d attempts: %s", len(RETRY_DELAYS), url)
    return False
