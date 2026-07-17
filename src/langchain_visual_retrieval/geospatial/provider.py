"""GeoVectorProvider: Core geospatial retrieval provider.

All geospatial logic (file I/O, CRS normalization, feature construction,
spatial indexing, query execution) is in the provider. Zero LangChain logic.
The provider is the single owner of all spatial operations.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional

import geopandas as gpd

from .exceptions import EmptyDatasetError, GeoIndexBuildError, SpatialQueryError
from .feature import GeoFeature, GeoFeatureMetadata, IndexMetadata
from .index import GeoSpatialIndex
from .io import load_dataframe
from .query import GeoQuery


class GeoVectorProvider:
    """Geospatial vector retrieval provider for Shapefile and GeoPackage.

    Orchestrates file I/O, CRS normalization, GeoFeature construction,
    and spatial index management. All spatial queries execute via the index.
    No LangChain knowledge; pure geospatial logic.
    """

    def __init__(self):
        """Initialize provider."""
        self.source: Optional[Path] = None
        self.layer: Optional[str] = None
        self.gdf: Optional[gpd.GeoDataFrame] = None
        self.spatial_index: Optional[GeoSpatialIndex] = None
        self.index_metadata: Optional[IndexMetadata] = None
        self.features: List[GeoFeature] = []

    def index(
        self,
        source: str | Path,
        layer: Optional[str] = None,
        bbox: Optional[tuple] = None,
        mask: Optional[object] = None,
        where: Optional[str] = None,
    ) -> IndexMetadata:
        """Index a Shapefile or GeoPackage.

        Creates canonical GeoFeature records and builds spatial index (STRtree).
        Index is never serialized; it is rebuilt in-memory from canonical
        feature records on every load.

        Args:
            source: Path to .shp or .gpkg file
            layer: Layer name (for GeoPackage with multiple layers)
            bbox: Optional spatial filter during loading
            mask: Optional geometry mask
            where: Optional attribute filter

        Returns:
            IndexMetadata with index information

        Raises:
            UnsupportedGeoFormatError: If format not supported
            MissingCRSError: If CRS cannot be determined
            EmptyDatasetError: If no features loaded
            GeoIndexBuildError: If indexing fails
        """
        source = Path(source)
        self.source = source
        self.layer = layer

        # Load GeoDataFrame
        self.gdf = load_dataframe(source, layer)

        # Build canonical GeoFeatures (dataset properties + framework metadata)
        self._build_features()

        # Build spatial index (STRtree, never serialized)
        self.spatial_index = GeoSpatialIndex(self.features)

        # Create metadata for persistence/versioning
        self.index_metadata = IndexMetadata(
            index_version=1,
            framework_version="1.4.7+",  # langchain-core version
            provider_name="GeoVectorProvider",
            created_at=self._get_timestamp(),
            source_path=str(source),
            layer_name=layer or "default",
            crs=str(self.gdf.crs),
            feature_count=len(self.features),
        )

        return self.index_metadata

    def _build_features(self):
        """Build canonical GeoFeature objects from GeoDataFrame.

        Separates dataset properties (from file) from framework metadata.
        """
        if self.gdf is None:
            raise GeoIndexBuildError("GeoDataFrame not loaded")

        self.features = []

        for idx, row in self.gdf.iterrows():
            feature_id = f"{self.source.stem}_{idx}"

            # Separate properties (dataset attrs) from metadata (framework info)
            properties = {k: v for k, v in row.items() if k != "geometry"}

            # Framework metadata
            feature_metadata = GeoFeatureMetadata(
                source=str(self.source),
                layer=self.layer or "default",
                provider="GeoVectorProvider",
                driver=self._get_driver(),
                index_version=1,
            )

            # Build GeoFeature with separated concerns
            feature = GeoFeature(
                feature_id=feature_id,
                geometry=row.geometry,
                properties=properties,
                metadata=feature_metadata,
                crs=str(self.gdf.crs),
            )

            self.features.append(feature)

    def _get_driver(self) -> str:
        """Determine driver from source file extension.

        Returns:
            "SHAPEFILE" for .shp, "GPKG" for .gpkg, "UNKNOWN" otherwise
        """
        if self.source is None:
            return "UNKNOWN"

        if self.source.suffix.lower() == ".shp":
            return "SHAPEFILE"
        elif self.source.suffix.lower() == ".gpkg":
            return "GPKG"
        return "UNKNOWN"

    @staticmethod
    def _get_timestamp() -> str:
        """Get ISO 8601 timestamp.

        Returns:
            ISO 8601 formatted timestamp with Z suffix
        """
        return datetime.utcnow().isoformat() + "Z"

    def search(self, query: GeoQuery) -> List[GeoFeature]:
        """Execute spatial + attribute query.

        All spatial queries use STRtree index; no full-dataset scans.
        Attribute filtering applied post-spatial on candidates.

        Args:
            query: GeoQuery object with spatial/attribute filters

        Returns:
            List of matching GeoFeature objects (up to query.limit)

        Raises:
            RuntimeError: If index not built
            SpatialQueryError: If query fails
        """
        if self.spatial_index is None:
            raise RuntimeError("Index not built. Call index() first.")

        return self.spatial_index.execute_query(query)

    def get_metadata(self) -> Optional[IndexMetadata]:
        """Get index metadata.

        Returns:
            IndexMetadata if index has been built, None otherwise
        """
        return self.index_metadata

    def get_feature_count(self) -> int:
        """Get total number of features indexed.

        Returns:
            Number of features, 0 if not indexed
        """
        return len(self.features)
