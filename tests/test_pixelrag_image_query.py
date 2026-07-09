import base64
from pathlib import Path

from langchain_visual_retrieval.providers.pixelrag import PixelRAGProvider


def test_pixelrag_provider_accepts_image_payload() -> None:
    provider = PixelRAGProvider(endpoint="http://localhost:30001")

    image_path = Path(__file__).with_name("sample.png")
    if not image_path.exists():
        image_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    with image_path.open("rb") as handle:
        image_bytes = handle.read()

    payload = provider._build_query_payload(
        {"image": base64.b64encode(image_bytes).decode("utf-8")},
        top_k=3,
    )

    assert payload["queries"][0]["image"] == base64.b64encode(image_bytes).decode("utf-8")
    assert payload["n_docs"] == 3
