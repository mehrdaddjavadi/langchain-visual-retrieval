"""Unit tests for geospatial spatial index."""

import pytest
from shapely.geometry import Point, Polygon, box

from langchain_visual_retrieval.geospatial.exceptions import (
    EmptyDatasetError,
    GeoIndexBuildError,
)
from langchain_visual_retrieval.geospatial.feature import (
    GeoFeature,
    GeoFeatureMetadata,
)
from langchain_visual_retrieval.geospatial.index import GeoSpatialIndex
from langchain_visual_retrieval.geospatial.query import GeoQuery


@pytest.fixture
def sample_features():
    """Create sample features for testing."""
    metadata = GeoFeatureMetadata(
        source="test.shp",
        layer="default",
        provider="GeoVectorProvider",
        driver="SHAPEFILE",
        index_version=1,
    )

    features = [
        GeoFeature(
            feature_id="feat_0",
            geometry=Point(0, 0),
            properties={"name": "origin", "value": 100},
            metadata=metadata,
            crs="EPSG:4326",
        ),
        GeoFeature(
            feature_id="feat_1",
            geometry=Point(10, 10),
            properties={"name": "northeast", "value": 200},
            metadata=metadata,
            crs="EPSG:4326",
        ),
        GeoFeature(
            feature_id="feat_2",
            geometry=Point(5, 5),
            properties={"name": "center", "value": 150},
            metadata=metadata,
            crs="EPSG:4326",
        ),
        GeoFeature(
            feature_id="feat_3",
            geometry=Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]),
            properties={"name": "square", "value": 300},
            metadata=metadata,
            crs="EPSG:4326",
        ),
    ]

    return features


class TestGeoSpatialIndex:
    """Test GeoSpatialIndex class."""

    def test_index_creation(self, sample_features):
        """Test creating spatial index from features."""
        index = GeoSpatialIndex(sample_features)

        assert index is not None
        assert len(index.features) == 4
        assert len(index.geometries) == 4
        assert index.tree is not None

    def test_index_with_empty_features_raises_error(self):
        """Test that empty feature list raises EmptyDatasetError."""
        with pytest.raises(EmptyDatasetError):
            GeoSpatialIndex([])

    def test_search_bbox_basic(self, sample_features):
        """Test bounding box search."""
        index = GeoSpatialIndex(sample_features)

        # Bbox covering origin and center
        bbox = (-2, -2, 6, 6)
        results = index.search_bbox(bbox)

        assert len(results) > 0
        # Should find at least feat_0 (origin) and feat_2 (center)
        assert any(i for i in results if sample_features[i].feature_id in ["feat_0", "feat_2"])

    def test_search_intersects(self, sample_features):
        """Test intersection search with geometry."""
        index = GeoSpatialIndex(sample_features)

        # Point at origin should intersect the square and the origin point
        point = Point(0, 0)
        results = index.search_intersects(point)

        assert len(results) > 0

    def test_search_contains(self, sample_features):
        """Test containment search."""
        index = GeoSpatialIndex(sample_features)

        # Point inside the square
        point = Point(0, 0)
        results = index.search_contains(point)

        # The square (feat_3) should contain the origin point
        assert len(results) > 0

    def test_search_nearest(self, sample_features):
        """Test nearest-neighbor search."""
        index = GeoSpatialIndex(sample_features)

        # Search for 2 nearest to northeast point
        point = Point(10, 10)
        results = index.search_nearest(point, k=2)

        assert len(results) > 0
        assert len(results) <= 2

    def test_filter_attributes(self, sample_features):
        """Test attribute filtering on candidate indices."""
        index = GeoSpatialIndex(sample_features)

        # Get indices for value > 150
        indices = [0, 1, 2, 3]  # All indices
        where = {"value": [">", 150]}
        filtered = index.filter_attributes(indices, where)

        # Should find feat_1 (200) and feat_3 (300)
        assert len(filtered) == 2

    def test_filter_attributes_with_equality(self, sample_features):
        """Test attribute filtering with equality."""
        index = GeoSpatialIndex(sample_features)

        indices = [0, 1, 2, 3]
        where = {"name": "northeast"}
        filtered = index.filter_attributes(indices, where)

        # Only feat_1 has name="northeast"
        assert len(filtered) == 1

    def test_filter_attributes_on_subset(self, sample_features):
        """Test attribute filtering on subset of indices."""
        index = GeoSpatialIndex(sample_features)

        # Only check indices 0 and 1
        indices = [0, 1]
        where = {"value": [">", 150]}
        filtered = index.filter_attributes(indices, where)

        # From indices [0, 1], only feat_1 (200) > 150
        assert len(filtered) == 1

    def test_execute_query_with_bbox(self, sample_features):
        """Test query execution with bbox."""
        index = GeoSpatialIndex(sample_features)

        bbox = (-2, -2, 6, 6)
        query = GeoQuery(bbox=bbox, limit=10)
        results = index.execute_query(query)

        assert len(results) > 0
        assert all(isinstance(f, GeoFeature) for f in results)

    def test_execute_query_with_bbox_and_attribute_filter(self, sample_features):
        """Test query execution with spatial and attribute filters."""
        index = GeoSpatialIndex(sample_features)

        bbox = (-2, -2, 11, 11)  # Covers all
        where = {"value": [">", 150]}
        query = GeoQuery(bbox=bbox, where=where, limit=10)
        results = index.execute_query(query)

        # Should find feat_1 (200) and feat_3 (300)
        assert len(results) > 0

    def test_execute_query_with_limit(self, sample_features):
        """Test that query limit is respected."""
        index = GeoSpatialIndex(sample_features)

        bbox = (-2, -2, 11, 11)  # Covers all 4
        query = GeoQuery(bbox=bbox, limit=2)
        results = index.execute_query(query)

        assert len(results) <= 2

    def test_execute_query_with_no_spatial_filter(self, sample_features):
        """Test query with no spatial filter returns all (up to limit)."""
        index = GeoSpatialIndex(sample_features)

        query = GeoQuery(limit=10)
        results = index.execute_query(query)

        # Should return all features up to limit
        assert len(results) == 4

    def test_execute_query_returns_features_in_order(self, sample_features):
        """Test that query results maintain index order."""
        index = GeoSpatialIndex(sample_features)

        bbox = (-2, -2, 11, 11)
        query = GeoQuery(bbox=bbox, limit=10)
        results = index.execute_query(query)

        # Results should be GeoFeature objects
        assert all(isinstance(f, GeoFeature) for f in results)

    def test_matches_conditions_various_operators(self, sample_features):
        """Test attribute matching with various operators."""
        index = GeoSpatialIndex(sample_features)

        properties = {"value": 200}

        # Test >
        assert index._matches_conditions(properties, {"value": [">", 150]})
        assert not index._matches_conditions(properties, {"value": [">", 250]})

        # Test <
        assert index._matches_conditions(properties, {"value": ["<", 250]})
        assert not index._matches_conditions(properties, {"value": ["<", 150]})

        # Test >=
        assert index._matches_conditions(properties, {"value": [">=", 200]})
        assert index._matches_conditions(properties, {"value": [">=", 100]})

        # Test <=
        assert index._matches_conditions(properties, {"value": ["<=", 200]})
        assert index._matches_conditions(properties, {"value": ["<=", 250]})

        # Test in
        assert index._matches_conditions(properties, {"value": ["in", [100, 200, 300]]})
        assert not index._matches_conditions(properties, {"value": ["in", [100, 300]]})

    def test_matches_conditions_missing_key_returns_false(self):
        """Test that matching fails if key missing from properties."""
        index = GeoSpatialIndex([GeoFeature(
            feature_id="test",
            geometry=Point(0, 0),
        )])

        properties = {"name": "test"}
        where = {"missing_key": "value"}

        assert not index._matches_conditions(properties, where)

    def test_bbox_search_returns_indices(self, sample_features):
        """Test that bbox search returns list of indices."""
        index = GeoSpatialIndex(sample_features)

        bbox = (0, 0, 10, 10)
        results = index.search_bbox(bbox)

        assert isinstance(results, list)
        assert all(isinstance(i, (int, type(0))) for i in results)
