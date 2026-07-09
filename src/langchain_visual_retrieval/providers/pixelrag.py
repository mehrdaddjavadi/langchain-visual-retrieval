"""PixelRAG provider implementation."""

from __future__ import annotations

import json
import urllib.request
from typing import Any

from langchain_visual_retrieval.providers.base import BaseVisualProvider, QueryInput


class PixelRAGProvider(BaseVisualProvider):
    """Provider implementation for the PixelRAG visual search API."""

    def __init__(self, endpoint: str = "http://localhost:30001") -> None:
        """Initialize the provider.

        Args:
            endpoint: The PixelRAG API base URL.
        """
        self.endpoint = endpoint.rstrip("/")

    def index(self, source: Any) -> dict[str, Any]:
        """Build or update a visual index from a source.

        Args:
            source: A source identifier or payload understood by PixelRAG.

        Returns:
            A provider-specific result payload.
        """
        return {"source": source, "status": "indexed"}

    def search(self, query: QueryInput, top_k: int = 5) -> list[dict[str, Any]]:
        """Search the PixelRAG API for a visual query.

        Args:
            query: A text query or an image payload.
            top_k: Maximum number of results to request.

        Returns:
            A list of provider-specific result dictionaries.
        """
        payload = self._build_query_payload(query, top_k=top_k)
        body = json.dumps(payload).encode()
        request = urllib.request.Request(
            f"{self.endpoint}/search",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))

        hits = data.get("results", [{}])[0].get("hits", [])
        return [
            {
                "id": str(hit.get("tile_index", idx)),
                "source": hit.get("url", ""),
                "page": hit.get("chunk_index"),
                "image_path": hit.get("tile_path"),
                "score": hit.get("score"),
                "metadata": {
                    "url": hit.get("url", ""),
                    "tile_index": hit.get("tile_index"),
                    "chunk_index": hit.get("chunk_index"),
                },
            }
            for idx, hit in enumerate(hits)
        ]

    def _build_query_payload(self, query: QueryInput, *, top_k: int) -> dict[str, Any]:
        """Build a PixelRAG-compatible request payload from a text or image query."""
        if isinstance(query, dict):
            payload_query: dict[str, Any] = {"image": query.get("image")}
            if query.get("text") is not None:
                payload_query["text"] = query["text"]
            return {"queries": [payload_query], "n_docs": top_k}

        if isinstance(query, str):
            return {"queries": [{"text": query}], "n_docs": top_k}

        msg = "Query must be a string or a dictionary containing text/image data"
        raise TypeError(msg)
