"""File I/O for geospatial vector data using pyogrio and geopandas."""

from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd

from .exceptions import (
    EmptyDatasetError,
    GeoIndexBuildError,
    MissingCRSError,
    SpatialQueryError,
    UnsupportedGeoFormatError,
)


def load_dataframe(
    source: str | Path, layer: Optional[str] = None
) -> gpd.GeoDataFrame:
    """Load entire GeoDataFrame from Shapefile or GeoPackage.

    Args:
        source: Path to .shp or .gpkg file
        layer: Layer name (for GeoPackage with multiple layers)

    Returns:
        GeoDataFrame with all features

    Raises:
        UnsupportedGeoFormatError: If format is not .shp or .gpkg
        MissingCRSError: If CRS cannot be determined
        EmptyDatasetError: If dataset contains no features
        GeoIndexBuildError: If file read fails
    """
    source = Path(source)

    # Validate format
    if source.suffix.lower() not in [".shp", ".gpkg"]:
        raise UnsupportedGeoFormatError(
            f"Unsupported format: {source.suffix}. Must be .shp or .gpkg"
        )

    # Read with pyogrio (faster than fiona)
    try:
        gdf = gpd.read_file(source, layer=layer, engine="pyogrio")
    except Exception as e:
        raise GeoIndexBuildError(f"Failed to read {source}: {e}")

    # Validate CRS
    if gdf.crs is None:
        raise MissingCRSError(f"No CRS found in {source}")

    # Check not empty
    if len(gdf) == 0:
        raise EmptyDatasetError(f"Dataset {source} contains no features")

    return gdf


def load_bounds(
    source: str | Path, layer: Optional[str] = None
) -> Tuple[float, float, float, float]:
    """Get bounding box of dataset.

    Args:
        source: Path to .shp or .gpkg file
        layer: Layer name (for GeoPackage)

    Returns:
        Bounding box tuple (minx, miny, maxx, maxy)
    """
    gdf = load_dataframe(source, layer)
    bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
    return tuple(bounds)


def load_subset(
    source: str | Path,
    layer: Optional[str] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    mask: Optional[object] = None,
    where: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Load subset of dataset with spatial/attribute filtering.

    Args:
        source: Path to .shp or .gpkg
        layer: Layer name (for GeoPackage)
        bbox: Bounding box filter (minx, miny, maxx, maxy)
        mask: Geometry mask
        where: Attribute filter string (SQL-like)

    Returns:
        Filtered GeoDataFrame

    Raises:
        SpatialQueryError: If subset query fails
        EmptyDatasetError: If subset query returns no features
    """
    # Use pyogrio's read_file with spatial filters for efficiency
    try:
        gdf = gpd.read_file(
            source,
            layer=layer,
            engine="pyogrio",
            bbox=bbox,
            mask=mask,
            where=where,
        )
    except Exception as e:
        raise SpatialQueryError(f"Subset query failed: {e}")

    if len(gdf) == 0:
        raise EmptyDatasetError("Subset query returned no features")

    return gdf
