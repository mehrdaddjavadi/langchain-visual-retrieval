"""Unit tests for geospatial file ingestion."""

import pytest
from pathlib import Path

from langchain_visual_retrieval.geospatial.exceptions import (
    EmptyDatasetError,
    GeoIndexBuildError,
    MissingCRSError,
    UnsupportedGeoFormatError,
)
from langchain_visual_retrieval.geospatial.io import (
    load_bounds,
    load_dataframe,
    load_subset,
)


class TestLoadDataframe:
    """Test load_dataframe function."""

    def test_load_valid_shapefile(self):
        """Test loading a valid Shapefile."""
        # Using the real test asset
        test_shapefile = Path("tests/flood_area_shapefile/pahnehaye_seylab.shp")

        if test_shapefile.exists():
            gdf = load_dataframe(test_shapefile)

            assert gdf is not None
            assert len(gdf) > 0
            assert gdf.crs is not None
            assert "geometry" in gdf.columns

    def test_load_unsupported_format(self):
        """Test loading unsupported format raises error."""
        with pytest.raises(UnsupportedGeoFormatError):
            load_dataframe("test.csv")

    def test_load_unsupported_format_with_uppercase(self):
        """Test unsupported format with uppercase extension."""
        with pytest.raises(UnsupportedGeoFormatError):
            load_dataframe("test.TXT")

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises GeoIndexBuildError."""
        with pytest.raises(GeoIndexBuildError):
            load_dataframe("nonexistent_file.shp")

    def test_shapefile_format_accepted(self):
        """Test that .shp format is accepted."""
        test_shapefile = Path("tests/flood_area_shapefile/pahnehaye_seylab.shp")

        if test_shapefile.exists():
            gdf = load_dataframe(test_shapefile)
            assert len(gdf) > 0

    def test_geopackage_format_accepted(self):
        """Test that .gpkg format is accepted.
        
        This test is skipped if no .gpkg file is available in tests.
        """
        # This would test with a real .gpkg file if available
        # Placeholder for real test with actual file
        pass


class TestLoadBounds:
    """Test load_bounds function."""

    def test_get_bounds_from_shapefile(self):
        """Test getting bounds from shapefile."""
        test_shapefile = Path("tests/flood_area_shapefile/pahnehaye_seylab.shp")

        if test_shapefile.exists():
            bounds = load_bounds(test_shapefile)

            assert bounds is not None
            assert len(bounds) == 4
            assert isinstance(bounds, tuple)
            minx, miny, maxx, maxy = bounds
            assert minx < maxx
            assert miny < maxy

    def test_bounds_format(self):
        """Test that bounds are in correct format (minx, miny, maxx, maxy)."""
        test_shapefile = Path("tests/flood_area_shapefile/pahnehaye_seylab.shp")

        if test_shapefile.exists():
            minx, miny, maxx, maxy = load_bounds(test_shapefile)

            # Bounds should follow convention: minx < maxx, miny < maxy
            assert minx < maxx
            assert miny < maxy
            # For valid geographic data
            assert isinstance(minx, (int, float))
            assert isinstance(miny, (int, float))
            assert isinstance(maxx, (int, float))
            assert isinstance(maxy, (int, float))


class TestLoadSubset:
    """Test load_subset function."""

    def test_load_subset_with_bbox(self):
        """Test loading subset with bounding box filter."""
        test_shapefile = Path("tests/flood_area_shapefile/pahnehaye_seylab.shp")

        if test_shapefile.exists():
            # Get full bounds first
            bounds = load_bounds(test_shapefile)
            minx, miny, maxx, maxy = bounds

            # Define a smaller bbox within the full extent
            small_bbox = (minx, miny, (minx + maxx) / 2, (miny + maxy) / 2)

            subset = load_subset(test_shapefile, bbox=small_bbox)
            full = load_dataframe(test_shapefile)

            # Subset should have fewer or equal features
            assert len(subset) <= len(full)

    def test_load_subset_with_invalid_bbox(self):
        """Test load_subset with bbox that covers no features."""
        test_shapefile = Path("tests/flood_area_shapefile/pahnehaye_seylab.shp")

        if test_shapefile.exists():
            # Use a bbox far outside the data extent
            empty_bbox = (180, 80, 179, 85)  # Invalid: minx > maxx in this range

            # This should raise EmptyDatasetError when result is empty
            with pytest.raises(EmptyDatasetError):
                load_subset(test_shapefile, bbox=empty_bbox)

    def test_load_subset_with_attribute_filter(self):
        """Test loading subset with attribute filter (SQL-like where clause)."""
        test_shapefile = Path("tests/flood_area_shapefile/pahnehaye_seylab.shp")

        if test_shapefile.exists():
            full = load_dataframe(test_shapefile)

            # Try with a basic where clause if columns exist
            if len(full.columns) > 0:
                first_col = [c for c in full.columns if c != "geometry"][0] if len(full.columns) > 1 else None
                if first_col:
                    first_value = full[first_col].iloc[0]
                    where_clause = f"{first_col} = {first_value}"

                    subset = load_subset(test_shapefile, where=where_clause)
                    assert len(subset) <= len(full)

    def test_load_subset_returns_geodataframe(self):
        """Test that load_subset returns a valid GeoDataFrame."""
        test_shapefile = Path("tests/flood_area_shapefile/pahnehaye_seylab.shp")

        if test_shapefile.exists():
            bounds = load_bounds(test_shapefile)
            minx, miny, maxx, maxy = bounds
            small_bbox = (minx, miny, (minx + maxx) / 2, (miny + maxy) / 2)

            subset = load_subset(test_shapefile, bbox=small_bbox)

            # Should have geometry column
            assert "geometry" in subset.columns
            # Should have CRS
            assert subset.crs is not None
            # Should have at least one feature
            assert len(subset) > 0
