import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from pydoc import html
from typing import Any, Dict, Iterable, Optional
import io
# For pandas DataFrame type hint
import pyarrow
import gzip
from typing import Union

import boto3
from pyparsing import line
import html

class StorageError(RuntimeError):
    """Raised when the storage backend is misconfigured or fails in a non-recoverable way."""


class Storage(ABC):
    """
    Storage interface used by ingestion code.

    The ingestion layer writes data using logical keys (S3-style paths), e.g.:
        bronze/offers/dt=2026-02-12/run_id=20260212T171402Z/code_rome=C1504/segment=global/part-000001.jsonl

    The same key must work for:
    - Local filesystem (mapped to <FT_DATA_DIR>/<key>)
    - S3-compatible object storage (MinIO) (mapped to s3://<bucket>/<prefix>/<key>)

    This is an abstraction layer:
    - Do not implement logic here other than method signatures.
    - Concrete implementations are LocalStorage and S3Storage.
    """

    @abstractmethod
    def write_bytes(self, key: str, payload: bytes, content_type: str = "application/octet-stream") -> None:
        """Write raw bytes at key (useful for gz/html, images, etc.)."""
        raise NotImplementedError

    @abstractmethod
    def read_bytes(self, key: str) -> bytes:
        """Read raw bytes from key."""
        raise NotImplementedError

    # Helpers très pratiques pour les fichiers texte et JSON
    def write_gzip_text(self, key: str, text: str, encoding: str = "utf-8") -> None:
        """
        Write a .gz file containing UTF-8 text.
        Key should typically end with .gz (ex: page.html.gz).
        """
        gz_bytes = gzip.compress(text.encode(encoding))
        self.write_bytes(key, gz_bytes, content_type="application/gzip")

    def read_gzip_text(self, key: str, encoding: str = "utf-8") -> str:
        gz_bytes = self.read_bytes(key)
        return gzip.decompress(gz_bytes).decode(encoding, errors="replace")

    @abstractmethod
    def list_keys(self, prefix: str) -> Iterable[str]:
        """
        List logical keys under a given prefix.
        Example: prefix="bronze/offers/" returns keys like ".../part-000001.jsonl"
        """
        raise NotImplementedError

    @abstractmethod
    def read_jsonl(self, key: str) -> Iterable[Dict[str, Any]]:
        """
        Read an NDJSON/JSONL document from storage and yield dict records.
        """
        raise NotImplementedError

    @abstractmethod
    def get_object_jsonl(self, key: str) -> Iterable[Dict[str, Any]]:
        """
        Read an NDJSON/JSONL document from storage and yield dict records.
        """
        raise NotImplementedError


    @abstractmethod
    def write_parquet(self, key: str, df) -> None:
        """Write a parquet file at key."""
        raise NotImplementedError

    @abstractmethod
    def write_json(self, key: str, payload: Dict[str, Any]) -> None:
        """
        Write a single JSON document at 'key'.

        Typical usage:
        - run metadata: bronze/metadata/runs/run_id=.../run.json

        The write is expected to be idempotent (overwrite if existing).
        """
        raise NotImplementedError

    @abstractmethod
    def write_jsonl(self, key: str, records: Iterable[Dict[str, Any]]) -> int:
        """
        Write an NDJSON/JSONL document at 'key'.

        NDJSON = Newline Delimited JSON:
        - One JSON object per line
        - Easy to stream, easy to process in Spark / DuckDB / Pandas

        S3 object storage does not support "append" efficiently.
        The ingestion code should therefore write immutable parts:
        - part-000001.jsonl
        - part-000002.jsonl
        etc.

        Returns:
            Number of records written.
        """
        raise NotImplementedError


class LocalStorage(Storage):
    """
    Local filesystem storage.

    Keys are interpreted as relative paths under a root directory, typically:
        FT_DATA_DIR=data/france_travail

    Example:
        key = "bronze/offers/dt=.../part-000001.jsonl"
        -> data/france_travail/bronze/offers/dt=.../part-000001.jsonl

    Notes:
    - Keys are normalized to use forward slashes internally.
    - Parent directories are created automatically.
    """

    def __init__(self, root: Path, prefix: Optional[str] = None) -> None:
        self.root = root
        self.prefix = (prefix).strip("/") if prefix else None

    def _resolve(self, key: str) -> Path:
        """
        Convert a logical key to a local filesystem path.

        - Strips leading "/" to avoid absolute paths.
        - Normalizes backslashes to forward slashes to keep keys portable.
        - Creates parent directories.
        """
        normalized = key.lstrip("/").replace("\\", "/")
        path = self.root / Path(normalized)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_bytes(self, key: str, payload: bytes, content_type: str = "application/octet-stream") -> None:
        path = self._resolve(key)
        path.write_bytes(payload)

    def read_bytes(self, key: str) -> bytes:
        path = self._resolve(key)
        if not path.exists():
            raise FileNotFoundError(f"Key not found: {key}")
        return path.read_bytes()

    def list_keys(self, prefix: str) -> Iterable[str]:
        normalized = prefix.lstrip("/").replace("\\", "/")
        base = (self.root / Path(normalized))
        if not base.exists():
            return []
        keys = []
        for p in base.rglob("*"):
            if p.is_file():
                # convert absolute path -> logical key
                rel = p.relative_to(self.root).as_posix()
                keys.append(rel)
        return keys

    def read_jsonl(self, key: str) -> Iterable[Dict[str, Any]]:
        path = self._resolve(key)
        # _resolve crée les parents; ici on veut juste lire, mais ça ne gêne pas.
        if not path.exists():
            return []
        def gen():
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        #yield json.loads(line)
                        # AJOUTE unescape
                        clean_line = html.unescape(str(line))
                        
                        yield json.loads(clean_line)

                    except json.JSONDecodeError:
                        continue
        return gen()

    def write_parquet(self, key: str, df) -> None:
        path = self._resolve(key)
        df.to_parquet(path, index=False)

    def write_json(self, key: str, payload: Dict[str, Any]) -> None:
        """
        Write a JSON document to disk.

        Uses UTF-8 and ensure_ascii=True for consistent encoding.
        """
        path = self._resolve(key)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def write_jsonl(self, key: str, records: Iterable[Dict[str, Any]]) -> int:
        """
        Write NDJSON to disk.

        The file is written in one pass.
        """
        path = self._resolve(key)
        count = 0
        with path.open("w", encoding="utf-8") as f:
            for rec in records:
                #f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.write(json.dumps(rec, ensure_ascii=True) + "\n") # True = ASCII safe
                count += 1
        return count


class S3Storage(Storage):
    """
    S3-compatible storage for MinIO (or AWS S3).

    Uses boto3 with a custom endpoint:
        S3_ENDPOINT_URL=http://localhost:9000

    Objects are written using put_object:
    - write_json  -> application/json
    - write_jsonl -> application/x-ndjson

    Important constraints vs filesystem:
    - There is no cheap "append" in S3.
      Each object should be written as a full immutable file (part-* pattern).
    - Listing and partitioning is done by key prefixes, not directories.
      Using dt=..., run_id=..., code_rome=... enables efficient prefix filtering.
    """

    def __init__(
        self,
        *,
        endpoint_url: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        bucket: Optional[str] = None,
        prefix: Optional[str] = None,
        region: Optional[str] = None,
    ) -> None:
        # Bucket is mandatory when STORAGE_BACKEND=s3
        self.bucket = bucket or os.getenv("S3_BUCKET")
        if not self.bucket:
            raise StorageError("S3_BUCKET is required when STORAGE_BACKEND=s3")

        # Prefix is optional; it acts like a "root folder" in the bucket.
        # Example: prefix=france_travail
        # Key "bronze/offers/..." -> "france_travail/bronze/offers/..."
        self.prefix = (prefix).strip("/") if prefix else None

        # boto3 S3 client configured for MinIO via endpoint_url
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url or os.getenv("S3_ENDPOINT_URL"),
            aws_access_key_id=access_key or os.getenv("S3_ACCESS_KEY"),
            aws_secret_access_key=secret_key or os.getenv("S3_SECRET_KEY"),
            region_name=region or os.getenv("S3_REGION", "us-east-1"),
        )

    def _full_key(self, key: str) -> str:
        """
        Apply prefix and normalize separators.

        - Strips leading "/" so keys are always relative.
        - Replaces backslashes by forward slashes to keep keys portable.
        """
        normalized = key.lstrip("/").replace("\\", "/")
        if self.prefix:
            return f"{self.prefix}/{normalized}"
        return normalized

    def write_bytes(self, key: str, payload: bytes, content_type: str = "application/octet-stream") -> None:
        self.client.put_object(
            Bucket=self.bucket,
            Key=self._full_key(key),
            Body=payload,
            ContentType=content_type,
        )

    def read_bytes(self, key: str) -> bytes:
        resp = self.client.get_object(Bucket=self.bucket, Key=self._full_key(key))
        return resp["Body"].read()

    def list_keys(self, prefix: str) -> Iterable[str]:
        normalized = prefix.lstrip("/").replace("\\", "/")
        full_prefix = self._full_key(normalized)
        paginator = self.client.get_paginator("list_objects_v2")

        keys = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                k = obj["Key"]
                # strip prefix root (S3_PREFIX) to return logical key
                if self.prefix:
                    # remove "<prefix>/"
                    logical = k[len(self.prefix) + 1 :]
                else:
                    logical = k
                keys.append(logical)
        return keys

    def get_object_jsonl(self, key: str) -> Iterable[Dict[str, Any]]:
        full_key = self._full_key(key)
        resp = self.client.get_object(Bucket=self.bucket, Key=full_key)
        return resp 


    def read_jsonl(self, key: str) -> Iterable[Dict[str, Any]]:
        full_key = self._full_key(key)
        resp = self.client.get_object(Bucket=self.bucket, Key=full_key)
        body = resp["Body"].read().decode("utf-8", errors="replace")

        def gen():
            for line in body.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    #yield json.loads(line)
                    # AJOUTE unescape
                    clean_line = html.unescape(str(line))
                    yield json.loads(clean_line)                    
                except json.JSONDecodeError:
                    print(f"⚠️ JSON decode error in {key}: {line[:100]}...")
                    continue
        return gen()

    def write_parquet(self, key: str, df) -> None:
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)

        self.client.put_object(
            Bucket=self.bucket,
            Key=self._full_key(key),
            Body=buffer.getvalue(),
            ContentType="application/octet-stream",
        )

    def write_json(self, key: str, payload: Dict[str, Any]) -> None:
        """
        Write a JSON object as a single S3 object (overwrite if exists).

        ContentType is set for clarity and downstream consumers.
        """
        body = json.dumps(payload, ensure_ascii=True, indent=2).encode("utf-8")
        self.client.put_object(
            Bucket=self.bucket,
            Key=self._full_key(key),
            Body=body,
            ContentType="application/json; charset=utf-8",
        )

    def write_jsonl(self, key: str, records: Iterable[Dict[str, Any]]) -> int:
        """
        Write NDJSON as a single S3 object.

        This method builds a complete body in memory and uploads it.
        For very large payloads, write smaller parts (part-xxxxxx.jsonl)
        to keep memory usage bounded.
        """
        lines = []
        count = 0
        for rec in records:
            #lines.append(json.dumps(rec, ensure_ascii=False))
            lines.append(json.dumps(rec, ensure_ascii=True))  # True = ASCII safe
            count += 1

        body = ("\n".join(lines) + "\n").encode("utf-8")
        self.client.put_object(
            Bucket=self.bucket,
            Key=self._full_key(key),
            Body=body,
            ContentType="application/x-ndjson; charset=utf-8",
        )
        return count


def get_storage_from_env(local_root, s3_prefix) -> Storage:
    """
    Factory that selects the storage backend based on environment variables.

    Env variables:
    - STORAGE_BACKEND=local|s3
      - local: uses FT_DATA_DIR as root folder
      - s3:    uses MinIO/S3 settings

    Local:
    - FT_DATA_DIR=data/france_travail

    S3/MinIO:
    - S3_ENDPOINT_URL=http://localhost:9000
    - S3_ACCESS_KEY=...
    - S3_SECRET_KEY=...
    - S3_BUCKET=jobmarket
    - S3_PREFIX=france_travail (optional)
    - S3_REGION=us-east-1
    """
    backend = os.getenv("STORAGE_BACKEND", "local").lower().strip()

    if backend == "s3":
        return S3Storage(prefix=s3_prefix)

    root = Path(local_root)
    return LocalStorage(root)
