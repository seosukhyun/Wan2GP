import logging
import os

import boto3
from botocore.config import Config as BotoConfig

from .config import WorkerConfig

logger = logging.getLogger(__name__)


class S3Storage:
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.bucket = config.s3_bucket
        self.prefix = config.s3_prefix

        client_kwargs = {
            "service_name": "s3",
            "aws_access_key_id": config.s3_access_key or None,
            "aws_secret_access_key": config.s3_secret_key or None,
            "region_name": config.s3_region,
            "config": BotoConfig(retries={"max_attempts": 3, "mode": "adaptive"}),
        }
        if config.s3_endpoint_url:
            client_kwargs["endpoint_url"] = config.s3_endpoint_url

        self.client = boto3.client(**client_kwargs)
        self._ensure_bucket()

    def _ensure_bucket(self):
        """Create the bucket if it doesn't exist (useful for MinIO dev)."""
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except Exception:
            try:
                self.client.create_bucket(Bucket=self.bucket)
                logger.info("Created bucket: %s", self.bucket)
            except Exception as e:
                logger.warning("Could not create bucket %s: %s", self.bucket, e)

    def upload_video(self, local_path: str, job_id: str) -> str:
        """Upload a video file to S3. Returns the S3 key."""
        filename = os.path.basename(local_path)
        s3_key = f"{self.prefix}{job_id}/{filename}"

        logger.info("Uploading %s -> s3://%s/%s", local_path, self.bucket, s3_key)
        self.client.upload_file(
            local_path,
            self.bucket,
            s3_key,
            ExtraArgs={"ContentType": "video/mp4"},
        )
        return s3_key

    def get_public_url(self, s3_key: str, expires_in: int = 3600) -> str:
        """Generate a presigned URL for the uploaded object."""
        url = self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": s3_key},
            ExpiresIn=expires_in,
        )
        return url
