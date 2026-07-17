"""Data models for geospatial features and metadata."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from shapely.geometry.base import BaseGeometry


@dataclass
class GeoFeatureMetadata:
    """Framework-level metadata for a geospatial feature.
    
    This separates framework concerns (source, layer, provider, driver)
    from dataset attributes (stored in GeoFeature.properties).
    """

    source: str  # File path or identifier
    layer: str  # Layer name (Shapefile layer or GeoPackage layer)
    provider: str  # "GeoVectorProvider"
    driver: str  # "SHAPEFILE" or "GPKG"
    index_version: int  # Index version for rebuild detection


@dataclass
class GeoFeature:
    """Canonical geospatial feature with separation of dataset and framework metadata.
    
    Properties are dataset attributes from the source file.
    Metadata are framework-level attributes managed by the provider.
    """

    feature_id: str  # Unique identifier
    geometry: BaseGeometry  # Shapely geometry object
    properties: Dict[str, Any] = field(default_factory=dict)  # Dataset attributes (from file)
    metadata: Optional[GeoFeatureMetadata] = None  # Framework metadata
    crs: Optional[str] = None  # CRS string (EPSG or WKT)


@dataclass
class IndexMetadata:
    """Metadata for persisted indexes and versioning.
    
    Used to track index state, schema version, and rebuild information.
    STRtree spatial index is never persisted; it is rebuilt in-memory
    from canonical feature records when an index is loaded.
    """

    index_version: int  # Schema version (for rebuild detection)
    framework_version: str  # langchain_core version used
    provider_name: str  # "GeoVectorProvider"
    created_at: str  # ISO 8601 timestamp
    source_path: str  # Path to source file
    layer_name: str  # Layer name
    crs: Optional[str]  # CRS string
    feature_count: int  # Number of features indexed
