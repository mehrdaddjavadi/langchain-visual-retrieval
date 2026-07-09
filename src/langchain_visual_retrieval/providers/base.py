"""Base interfaces for visual retrieval providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

QueryInput = str | dict[str, Any]


class BaseVisualProvider(ABC):
    """Base interface for vision-native retrieval providers.

    Future providers may include PixelRAG, ColPali, Qwen-VL, Florence, or
    InternVL without changing the public API.
    """

    @abstractmethod
    def index(self, source: Any) -> Any:
        """Build or update a visual index from a source.

        Args:
            source: An input source for indexing.

        Returns:
            Provider-specific index information.
        """

    @abstractmethod
    def search(self, query: QueryInput, top_k: int = 5) -> list[Any]:
        """Search the visual index for a query.

        Args:
            query: A text query or a provider-supported image payload.
            top_k: The maximum number of results to return.

        Returns:
            Ranked provider results.
        """
