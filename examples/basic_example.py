from langchain_visual_retrieval.providers.pixelrag import PixelRAGProvider
from langchain_visual_retrieval.retrievers.visual_retriever import VisualRetriever


def main() -> None:
    provider = PixelRAGProvider(endpoint="http://localhost:30001")
    retriever = VisualRetriever(provider)

    results = retriever.search("Find the Kubernetes deployment screenshot")
    for result in results:
        print(result.model_dump())


if __name__ == "__main__":
    main()
