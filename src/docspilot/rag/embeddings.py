"""Embedders. `HashEmbedder` is an offline stand-in (feature hashing): good enough to
develop and test the pipeline, not to judge retrieval quality. Use Voyage in prod."""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import math
import random
import re
from typing import Any, Literal, Protocol

import httpx

from docspilot.logging import get_logger

log = get_logger("embeddings")
InputType = Literal["document", "query"]
DIM = 1024
_TOKEN = re.compile(r"[a-z0-9$%]+")


class Embedder(Protocol):
    name: str  # recorded per document: a model change triggers re-embedding
    dim: int

    async def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]: ...


class HashEmbedder:
    name = "hash-1024"
    dim = DIM

    async def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = _TOKEN.findall(text.lower())
        grams = [*tokens, *(f"{a}_{b}" for a, b in itertools.pairwise(tokens))]
        for gram in grams:
            h = int.from_bytes(hashlib.blake2b(gram.encode(), digest_size=8).digest(), "big")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class VoyageError(Exception):
    pass


async def _post_with_retries(
    client: httpx.AsyncClient, url: str, payload: dict[str, Any], *, attempts: int = 4
) -> dict[str, Any]:
    delay = 0.5
    for attempt in range(1, attempts + 1):
        try:
            res = await client.post(url, json=payload)
            if res.status_code in (429, 500, 502, 503, 504) and attempt < attempts:
                raise httpx.HTTPStatusError("retryable", request=res.request, response=res)
            res.raise_for_status()
            data: dict[str, Any] = res.json()
            return data
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            retryable = isinstance(exc, httpx.TransportError) or (
                isinstance(exc, httpx.HTTPStatusError)
                and exc.response.status_code in (429, 500, 502, 503, 504)
            )
            if not retryable or attempt == attempts:
                raise VoyageError(f"{url} failed: {exc}") from exc
            await asyncio.sleep(delay + random.uniform(0, delay / 2))  # noqa: S311
            delay *= 2
    raise VoyageError("unreachable")  # pragma: no cover


class VoyageEmbedder:
    dim = DIM

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "voyage-3.5",
        base_url: str = "https://api.voyageai.com/v1",
        client: httpx.AsyncClient | None = None,
        batch_size: int = 96,
    ) -> None:
        self.name = f"voyage:{model}"
        self._model = model
        self._batch = batch_size
        self._client = client or httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(30.0, connect=5.0),
        )

    async def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            batch = texts[i : i + self._batch]
            data = await _post_with_retries(
                self._client,
                "/embeddings",
                {"input": batch, "model": self._model, "input_type": input_type},
            )
            rows = sorted(data["data"], key=lambda r: r["index"])
            vectors = [r["embedding"] for r in rows]
            if len(vectors) != len(batch) or any(len(v) != self.dim for v in vectors):
                raise VoyageError("unexpected embedding shape from provider")
            out.extend(vectors)
        return out

    async def aclose(self) -> None:
        await self._client.aclose()


class CloudflareEmbedder:
    """Workers AI embeddings. bge-m3 returns 1024 dims, matching the `vector(1024)` schema;
    a model with another dimension needs a new migration and a full re-ingest."""

    dim = DIM

    def __init__(
        self,
        account_id: str,
        api_token: str,
        *,
        model: str = "@cf/baai/bge-m3",
        client: httpx.AsyncClient | None = None,
        batch_size: int = 50,
    ) -> None:
        self.name = f"cloudflare:{model}"
        self._model = model
        self._batch = batch_size
        self._client = client or httpx.AsyncClient(
            base_url=f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run",
            headers={"Authorization": f"Bearer {api_token}"},
            timeout=httpx.Timeout(30.0, connect=5.0),
        )

    async def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            batch = texts[i : i + self._batch]
            data = await _post_with_retries(self._client, f"/{self._model}", {"text": batch})
            vectors = (data.get("result") or {}).get("data") or []
            if len(vectors) != len(batch) or any(len(v) != self.dim for v in vectors):
                raise VoyageError("unexpected embedding shape from Workers AI")
            out.extend(vectors)
        return out

    async def aclose(self) -> None:
        await self._client.aclose()
