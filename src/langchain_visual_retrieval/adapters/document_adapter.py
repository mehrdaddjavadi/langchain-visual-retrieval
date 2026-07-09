"""Compatibility adapter for converting visual documents to LangChain documents."""

from __future__ import annotations

from langchain_core.documents import Document

from langchain_visual_retrieval.documents.visual_document import VisualDocument


class DocumentAdapter:
    """Convert internal visual results into LangChain documents."""

    @staticmethod
    def to_document(visual_document: VisualDocument) -> Document:
        """Convert a visual document into a LangChain document.

        Args:
            visual_document: The internal visual document.

        Returns:
            A LangChain document with compatibility metadata.
        """
        metadata = dict(visual_document.metadata)
        if visual_document.source is not None:
            metadata.setdefault("source", visual_document.source)
        if visual_document.page is not None:
            metadata.setdefault("page", visual_document.page)
        if visual_document.image_path is not None:
            metadata.setdefault("image_path", visual_document.image_path)
        if visual_document.score is not None:
            metadata.setdefault("score", visual_document.score)
        return Document(page_content="", metadata=metadata)

    @staticmethod
    def to_documents(visual_documents: list[VisualDocument]) -> list[Document]:
        """Convert a list of visual documents into LangChain documents."""
        return [DocumentAdapter.to_document(document) for document in visual_documents]
