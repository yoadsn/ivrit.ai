"""S3 helpers for Knesset committee downloads.

Wraps :mod:`boto3` with an optional credential override so callers can
either pass explicit AWS credentials on the CLI or fall back to the
default boto3 credential resolution (env vars, ``~/.aws/credentials``,
IAM roles, ...).
"""

import logging
import pathlib
from typing import Optional, Tuple
from urllib.parse import unquote, urlparse

import boto3

logger = logging.getLogger(__name__)


def make_s3_client(
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_region: Optional[str] = None,
):
    kwargs = {}
    if aws_access_key_id:
        kwargs["aws_access_key_id"] = aws_access_key_id
    if aws_secret_access_key:
        kwargs["aws_secret_access_key"] = aws_secret_access_key
    if aws_region:
        kwargs["region_name"] = aws_region
    return boto3.client("s3", **kwargs)


def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Not an s3:// URI: {s3_uri}")
    bucket = parsed.netloc
    key = unquote(parsed.path.lstrip("/"))
    if not bucket or not key:
        raise ValueError(f"Malformed s3:// URI (missing bucket or key): {s3_uri}")
    return bucket, key


def s3_uri_filename(s3_uri: str) -> str:
    _, key = parse_s3_uri(s3_uri)
    return pathlib.Path(key).name


def s3_download(s3_client, s3_uri: str, dest: pathlib.Path) -> None:
    bucket, key = parse_s3_uri(s3_uri)
    logger.info("S3 download s3://%s/%s -> %s", bucket, key, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        s3_client.download_file(bucket, key, str(dest))
    except Exception:
        if dest.exists():
            dest.unlink()
        raise
