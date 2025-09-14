from typing import Optional, Tuple

import boto3

from src.helper.logging import logger


def get_boto_session(
    profile_name: Optional[str] = None,
    region_name: Optional[str] = None,
) -> boto3.Session:
    """Create and return a boto3 session with the given profile and region."""
    return boto3.Session(profile_name=profile_name, region_name=region_name)


def is_s3_path(path: str) -> bool:
    """Check if a path is an S3 path."""
    return isinstance(path, str) and path.startswith("s3://")


def parse_s3_path(s3_path: str) -> Tuple[str, str]:
    """Parse an S3 path into bucket and key."""
    path = s3_path.replace("s3://", "")
    path = path.rstrip("/")
    parts = path.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


def upload_to_s3(s3_client, local_file: str, bucket: str, s3_key: str) -> None:
    """Upload a local file to S3."""
    logger.debug(f"Uploading {local_file} to s3://{bucket}/{s3_key}")
    s3_client.upload_file(local_file, bucket, s3_key)


def download_s3_file(s3_client, bucket: str, key: str, local_path: str) -> None:
    """Download a file from S3 to a local path."""
    logger.debug(f"Downloading s3://{bucket}/{key} to {local_path}")
    s3_client.download_file(bucket, key, local_path)
