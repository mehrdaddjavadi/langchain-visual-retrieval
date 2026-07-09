from langchain_core.documents import Document

from langchain_visual_retrieval.adapters.document_adapter import DocumentAdapter
from langchain_visual_retrieval.documents.visual_document import VisualDocument
from langchain_visual_retrieval.providers.base import BaseVisualProvider
from langchain_visual_retrieval.providers.pixelrag import PixelRAGProvider
from langchain_visual_retrieval.retrievers.visual_retriever import VisualRetriever


class StubProvider(BaseVisualProvider):
    def index(self, source):
        return {"source": source, "status": "indexed"}

    def search(self, query, top_k=5):
        return [
            {
                "id": "1",
                "source": "https://example.com",
                "page": 3,
                "image_path": "/tmp/a.png",
                "score": 0.91,
                "metadata": {"url": "https://example.com"},
            }
        ]


def test_visual_retriever_returns_visual_documents() -> None:
    retriever = VisualRetriever(StubProvider())

    results = retriever.search("kubernetes deployment")

    assert len(results) == 1
    assert isinstance(results[0], VisualDocument)
    assert results[0].source == "https://example.com"
    assert results[0].score == 0.91


def test_document_adapter_converts_to_langchain_document() -> None:
    visual_document = VisualDocument(
        id="1",
        source="https://example.com",
        page=3,
        image_path="/tmp/a.png",
        score=0.91,
        metadata={"url": "https://example.com"},
    )

    document = DocumentAdapter.to_document(visual_document)

    assert isinstance(document, Document)
    assert document.metadata["source"] == "https://example.com"
    assert document.metadata["page"] == 3


def test_pixelrag_provider_search_uses_provider_interface() -> None:
    provider = PixelRAGProvider(endpoint="http://localhost:30001")

    assert isinstance(provider, BaseVisualProvider)
    assert provider.endpoint == "http://localhost:30001"
