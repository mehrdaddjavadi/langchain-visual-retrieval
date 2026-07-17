"""Visual retrieval integration for LangChain."""

from langchain_visual_retrieval.adapters.document_adapter import DocumentAdapter
from langchain_visual_retrieval.documents.visual_document import VisualDocument
from langchain_visual_retrieval.providers.base import BaseVisualProvider
from langchain_visual_retrieval.providers.pixelrag import PixelRAGProvider
from langchain_visual_retrieval.retrievers.visual_retriever import VisualRetriever

# Geospatial v0.1.3
from langchain_visual_retrieval.geospatial.feature import (
    GeoFeature,
    GeoFeatureMetadata,
    IndexMetadata,
)
from langchain_visual_retrieval.geospatial.provider import GeoVectorProvider
from langchain_visual_retrieval.geospatial.query import GeoQuery
from langchain_visual_retrieval.geospatial.retriever import GeoRetriever
from langchain_visual_retrieval.geospatial.exceptions import (
    EmptyDatasetError,
    GeoIndexBuildError,
    InvalidGeometryError,
    MissingCRSError,
    SpatialQueryError,
    UnsupportedGeoFormatError,
)

__all__ = [
    # Existing
    "BaseVisualProvider",
    "DocumentAdapter",
    "PixelRAGProvider",
    "VisualDocument",
    "VisualRetriever",
    # Geospatial v0.1.3
    "GeoFeature",
    "GeoFeatureMetadata",
    "IndexMetadata",
    "GeoQuery",
    "GeoVectorProvider",
    "GeoRetriever",
    "EmptyDatasetError",
    "GeoIndexBuildError",
    "InvalidGeometryError",
    "MissingCRSError",
    "SpatialQueryError",
    "UnsupportedGeoFormatError",
]
