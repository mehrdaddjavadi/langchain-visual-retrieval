"""GeoRetriever: LangChain wrapper for geospatial retrieval.

Thin wrapper that orchestrates the provider and converts results to LangChain Documents.
Never parses files, builds indexes, or implements spatial logic; all in provider.
"""

from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from .feature import GeoFeature
from .provider import GeoVectorProvider
from .query import GeoQuery


class GeoRetriever(BaseRetriever):
    """LangChain retriever for geospatial vector data.

    Thin wrapper around GeoVectorProvider. Orchestrates provider queries
    and converts GeoFeatures to LangChain Documents with complete metadata.
    Never parses files, builds indexes, or implements spatial logic.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider: GeoVectorProvider
    source: str
    layer: Optional[str] = None

    def __init__(
        self, provider: GeoVectorProvider, source: str, layer: Optional[str] = None
    ):
        """Initialize retriever with provider.

        Args:
            provider: GeoVectorProvider instance
            source: Path to source file
            layer: Optional layer name

        Raises:
            Any exception from provider.index()
        """
        super().__init__(provider=provider, source=source, layer=layer)

        # Index the source (provider handles everything)
        self.provider.index(source, layer)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Execute query and return LangChain Documents.

        This method is required by BaseRetriever but not used in practice.
        Use search_bbox(), search_intersects(), etc. instead.

        Args:
            query: Query string (not supported for spatial retrieval)

        Raises:
            NotImplementedError: Always; use specific search methods instead
        """
        raise NotImplementedError(
            "Use search_bbox(), search_intersects(), search_contains(), "
            "or search_nearest() instead. String queries not supported for geospatial retrieval."
        )

    def search_bbox(
        self,
        bbox: tuple,
        limit: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Search features in bounding box.

        Args:
            bbox: (minx, miny, maxx, maxy)
            limit: Maximum results
            where: Optional attribute filters (e.g., {"field": [">", 100]})

        Returns:
            List of LangChain Documents
        """
        query = GeoQuery(bbox=bbox, where=where, limit=limit)
        features = self.provider.search(query)
        return self._to_documents(features)

    def search_intersects(
        self,
        geometry: object,
        limit: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Search features intersecting geometry.

        Args:
            geometry: Shapely geometry to intersect
            limit: Maximum results
            where: Optional attribute filters

        Returns:
            List of LangChain Documents
        """
        query = GeoQuery(intersects=geometry, where=where, limit=limit)
        features = self.provider.search(query)
        return self._to_documents(features)

    def search_contains(
        self,
        geometry: object,
        limit: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Search features containing geometry.

        Args:
            geometry: Shapely geometry to contain
            limit: Maximum results
            where: Optional attribute filters

        Returns:
            List of LangChain Documents
        """
        query = GeoQuery(contains=geometry, where=where, limit=limit)
        features = self.provider.search(query)
        return self._to_documents(features)

    def search_nearest(
        self,
        geometry: object,
        k: int = 5,
    ) -> List[Document]:
        """Search k nearest features to geometry.

        Args:
            geometry: Reference geometry
            k: Number of nearest features

        Returns:
            List of LangChain Documents
        """
        query = GeoQuery(nearest_to=(geometry, k), limit=k)
        features = self.provider.search(query)
        return self._to_documents(features)

    def _to_documents(self, features: List[GeoFeature]) -> List[Document]:
        """Convert GeoFeatures to LangChain Documents with metadata preservation.

        Metadata is merged from:
        - Feature.properties: Dataset attributes (from file)
        - Feature.metadata: Framework information (source, layer, provider, driver)

        Args:
            features: List of GeoFeature objects

        Returns:
            List of LangChain Document objects
        """
        documents = []
        for feature in features:
            # Build metadata: properties (dataset attrs) + framework attrs
            metadata = {
                **feature.properties,
                "feature_id": feature.feature_id,
                "geometry_type": feature.geometry.geom_type,
                "crs": feature.crs,
            }

            # Add framework metadata if available
            if feature.metadata:
                metadata.update(
                    {
                        "source": feature.metadata.source,
                        "layer": feature.metadata.layer,
                        "provider": feature.metadata.provider,
                        "driver": feature.metadata.driver,
                    }
                )

            # Convert geometry to WKT for Document.page_content
            page_content = feature.geometry.wkt

            doc = Document(page_content=page_content, metadata=metadata)
            documents.append(doc)

        return documents
