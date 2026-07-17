"""Unit tests for geospatial data models and exceptions."""

import pytest
from shapely.geometry import Point, Polygon

from langchain_visual_retrieval.geospatial.exceptions import (
    EmptyDatasetError,
    GeoIndexBuildError,
    InvalidGeometryError,
    MissingCRSError,
    SpatialQueryError,
    UnsupportedGeoFormatError,
)
from langchain_visual_retrieval.geospatial.feature import (
    GeoFeature,
    GeoFeatureMetadata,
    IndexMetadata,
)
from langchain_visual_retrieval.geospatial.query import GeoQuery


class TestGeoFeatureMetadata:
    """Test GeoFeatureMetadata dataclass."""

    def test_creation_with_all_fields(self):
        """Test creating GeoFeatureMetadata with all required fields."""
        metadata = GeoFeatureMetadata(
            source="/path/to/file.shp",
            layer="default",
            provider="GeoVectorProvider",
            driver="SHAPEFILE",
            index_version=1,
        )

        assert metadata.source == "/path/to/file.shp"
        assert metadata.layer == "default"
        assert metadata.provider == "GeoVectorProvider"
        assert metadata.driver == "SHAPEFILE"
        assert metadata.index_version == 1

    def test_driver_field_types(self):
        """Test GeoFeatureMetadata with different driver types."""
        shp_metadata = GeoFeatureMetadata(
            source="data.shp", layer="layer1", provider="GeoVectorProvider",
            driver="SHAPEFILE", index_version=1
        )
        assert shp_metadata.driver == "SHAPEFILE"

        gpkg_metadata = GeoFeatureMetadata(
            source="data.gpkg", layer="layer1", provider="GeoVectorProvider",
            driver="GPKG", index_version=1
        )
        assert gpkg_metadata.driver == "GPKG"


class TestGeoFeature:
    """Test GeoFeature dataclass."""

    def test_creation_with_all_fields(self):
        """Test creating GeoFeature with all fields."""
        metadata = GeoFeatureMetadata(
            source="test.shp", layer="default", provider="GeoVectorProvider",
            driver="SHAPEFILE", index_version=1
        )
        geometry = Point(45.0, 35.0)
        properties = {"name": "feature1", "area": 100}

        feature = GeoFeature(
            feature_id="feat_0",
            geometry=geometry,
            properties=properties,
            metadata=metadata,
            crs="EPSG:4326",
        )

        assert feature.feature_id == "feat_0"
        assert feature.geometry == geometry
        assert feature.properties == properties
        assert feature.metadata == metadata
        assert feature.crs == "EPSG:4326"

    def test_creation_with_empty_properties(self):
        """Test GeoFeature with empty properties dict."""
        feature = GeoFeature(
            feature_id="feat_1",
            geometry=Point(0, 0),
            properties={},
        )

        assert feature.properties == {}
        assert feature.metadata is None

    def test_geometry_types(self):
        """Test GeoFeature with different geometry types."""
        point = Point(45.0, 35.0)
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])

        point_feature = GeoFeature(
            feature_id="point",
            geometry=point,
        )
        assert point_feature.geometry.geom_type == "Point"

        poly_feature = GeoFeature(
            feature_id="polygon",
            geometry=polygon,
        )
        assert poly_feature.geometry.geom_type == "Polygon"


class TestIndexMetadata:
    """Test IndexMetadata dataclass."""

    def test_creation_with_all_fields(self):
        """Test creating IndexMetadata with all required fields."""
        metadata = IndexMetadata(
            index_version=1,
            framework_version="1.4.7+",
            provider_name="GeoVectorProvider",
            created_at="2024-01-15T10:30:00Z",
            source_path="/path/to/file.shp",
            layer_name="default",
            crs="EPSG:4326",
            feature_count=42,
        )

        assert metadata.index_version == 1
        assert metadata.framework_version == "1.4.7+"
        assert metadata.provider_name == "GeoVectorProvider"
        assert metadata.created_at == "2024-01-15T10:30:00Z"
        assert metadata.source_path == "/path/to/file.shp"
        assert metadata.layer_name == "default"
        assert metadata.crs == "EPSG:4326"
        assert metadata.feature_count == 42

    def test_iso8601_timestamp_format(self):
        """Test that created_at uses ISO 8601 format."""
        metadata = IndexMetadata(
            index_version=1,
            framework_version="1.4.7+",
            provider_name="GeoVectorProvider",
            created_at="2024-01-15T10:30:00Z",
            source_path="test.shp",
            layer_name="default",
            crs="EPSG:4326",
            feature_count=10,
        )

        # Should be ISO 8601 format with Z suffix
        assert metadata.created_at.endswith("Z")
        assert "T" in metadata.created_at


class TestGeoQuery:
    """Test GeoQuery dataclass."""

    def test_query_with_no_filters(self):
        """Test GeoQuery with no filters returns default."""
        query = GeoQuery()

        assert query.bbox is None
        assert query.intersects is None
        assert query.contains is None
        assert query.nearest_to is None
        assert query.where is None
        assert query.limit == 5

    def test_query_with_bbox(self):
        """Test GeoQuery with bbox filter."""
        bbox = (40.0, 30.0, 50.0, 40.0)
        query = GeoQuery(bbox=bbox, limit=10)

        assert query.bbox == bbox
        assert query.limit == 10

    def test_query_with_intersects(self):
        """Test GeoQuery with geometry intersection."""
        geom = Point(45.0, 35.0)
        query = GeoQuery(intersects=geom, limit=5)

        assert query.intersects == geom

    def test_query_with_contains(self):
        """Test GeoQuery with geometry containment."""
        geom = Point(45.0, 35.0)
        query = GeoQuery(contains=geom, limit=5)

        assert query.contains == geom

    def test_query_with_nearest_to(self):
        """Test GeoQuery with nearest-neighbor search."""
        geom = Point(45.0, 35.0)
        query = GeoQuery(nearest_to=(geom, 3), limit=3)

        assert query.nearest_to == (geom, 3)

    def test_query_with_attribute_filter(self):
        """Test GeoQuery with attribute filtering."""
        where = {"area": [">", 1000], "name": "test"}
        query = GeoQuery(bbox=(0, 0, 10, 10), where=where, limit=5)

        assert query.where == where

    def test_query_with_combined_filters(self):
        """Test GeoQuery with spatial and attribute filters."""
        bbox = (40.0, 30.0, 50.0, 40.0)
        where = {"area": [">=", 500]}
        query = GeoQuery(bbox=bbox, where=where, limit=15)

        assert query.bbox == bbox
        assert query.where == where
        assert query.limit == 15


class TestExceptions:
    """Test custom exception classes."""

    def test_unsupported_geo_format_error(self):
        """Test UnsupportedGeoFormatError."""
        with pytest.raises(UnsupportedGeoFormatError):
            raise UnsupportedGeoFormatError("Format .csv not supported")

    def test_missing_crs_error(self):
        """Test MissingCRSError."""
        with pytest.raises(MissingCRSError):
            raise MissingCRSError("No CRS found in file")

    def test_invalid_geometry_error(self):
        """Test InvalidGeometryError."""
        with pytest.raises(InvalidGeometryError):
            raise InvalidGeometryError("Geometry is invalid or empty")

    def test_spatial_query_error(self):
        """Test SpatialQueryError."""
        with pytest.raises(SpatialQueryError):
            raise SpatialQueryError("Query failed: invalid parameters")

    def test_geo_index_build_error(self):
        """Test GeoIndexBuildError."""
        with pytest.raises(GeoIndexBuildError):
            raise GeoIndexBuildError("STRtree construction failed")

    def test_empty_dataset_error(self):
        """Test EmptyDatasetError."""
        with pytest.raises(EmptyDatasetError):
            raise EmptyDatasetError("Dataset contains no features")

    def test_exception_messages(self):
        """Test that exception messages are preserved."""
        msg = "Test error message"
        exc = UnsupportedGeoFormatError(msg)
        assert str(exc) == msg
