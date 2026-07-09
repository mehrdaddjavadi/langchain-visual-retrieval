"""Visual retriever implementation."""

from __future__ import annotations

from typing import Any

from langchain_visual_retrieval.adapters.document_adapter import DocumentAdapter
from langchain_visual_retrieval.documents.visual_document import VisualDocument
from langchain_visual_retrieval.providers.base import BaseVisualProvider, QueryInput


class VisualRetriever:
    """Retrieve visual results through a provider-agnostic interface."""

    def __init__(self, provider: BaseVisualProvider) -> None:
        """Initialize the retriever with a provider.

        Args:
            provider: A visual retrieval provider implementation.
        """
        self.provider = provider

    def search(self, query: QueryInput, top_k: int = 5) -> list[VisualDocument]:
        """Search the underlying provider and return visual documents.

        Args:
            query: A text query or a provider-supported image payload.
            top_k: Maximum number of results to return.

        Returns:
            A list of visual documents.
        """
        raw_results = self.provider.search(query, top_k=top_k)
        return [self._normalize_result(result) for result in raw_results]

    def as_documents(self, query: QueryInput, top_k: int = 5) -> list[Any]:
        """Return visual search results converted to LangChain documents."""
        visual_documents = self.search(query, top_k=top_k)
        return DocumentAdapter.to_documents(visual_documents)

    def _normalize_result(self, result: dict[str, Any]) -> VisualDocument:
        """Normalize provider-specific result payloads into a visual document."""
        return VisualDocument(
            id=result.get("id"),
            source=result.get("source"),
            page=result.get("page"),
            image_path=result.get("image_path"),
            score=result.get("score"),
            metadata=result.get("metadata", {}),
        )
