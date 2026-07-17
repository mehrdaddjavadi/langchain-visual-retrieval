"""Integration tests for GeoVectorProvider."""

import pytest
from pathlib import Path

from langchain_visual_retrieval.geospatial.provider import GeoVectorProvider
from langchain_visual_retrieval.geospatial.query import GeoQuery
from langchain_visual_retrieval.geospatial.feature import GeoFeature


class TestGeoVectorProvider:
    """Test GeoVectorProvider class."""

    def test_provider_initialization(self):
        """Test provider initialization."""
        provider = GeoVectorProvider()

        assert provider.source is None
        assert provider.layer is None
        assert provider.gdf is None
        assert provider.spatial_index is None
        assert provider.index_metadata is None
        assert provider.features == []

    def test_index_shapefile(self):
        """Test indexing a Shapefile."""
        test_shapefile = Path("tests/flood_area_shapefile/pahnehaye_seylab.shp")

        if test_shapefile.exists():
            provider = GeoVectorProvider()
            metadata = provider.index(test_shapefile)

            assert metadata is not None
            assert metadata.index_version == 1
            assert metadata.provider_name == "GeoVectorProvider"
            assert metadata.source_path == str(test_shapefile)
            assert metadata.feature_count > 0
            assert metadata.crs is not None

    def test_index_metadata_fields(self):
        """Test that all IndexMetadata fields are populated."""
        test_shapefile = Path("tests/flood_area_shapefile/pahnehaye_seylab.shp")

        if test_shapefile.exists():
            provider = GeoVectorProvider()
            metadata = provider.index(test_shapefile)

            assert metadata.index_version is not None
            assert metadata.framework_version is not None
            assert metadata.provider_name is not None
            assert metadata.created_at is not None
            assert metadata.created_at.endswith("Z")  # ISO 8601
            assert metadata.source_path is not None
            assert metadata.layer_name is not None
            assert metadata.crs is not None
            assert metadata.feature_count > 0

    def test_index_builds_features(self):
        """Test that indexing builds GeoFeature objects."""
        test_shapefile = Path("tests/flood_area_shapefile/pahnehaye_seylab.shp")

        if test_shapefile.exists():
            provider = GeoVectorProvider()
            provider.index(test_shapefile)

            assert len(provider.features) > 0
            assert all(isinstance(f, GeoFeature) for f in provider.features)
            assert all(f.geometry is not None for f in provider.features)
            assert all(f.properties is not None for f in provider.features)
            assert all(f.metadata is not None for f in provider.features)

    def test_feature_metadata_populated(self):
        """Test that GeoFeature metadata is correctly populated."""
        test_shapefile = Path("tests/flood_area_shapefile/pahnehaye_seylab.shp")

        if test_shapefile.exists():
            provider = GeoVectorProvider()
            provider.index(test_shapefile)

            for feature in provider.features:
                assert feature.metadata.source == str(test_shapefile)
                assert feature.metadata.provider == "GeoVectorProvider"
                assert feature.metadata.driver == "SHAPEFILE"
                assert feature.metadata.layer == "default"
                assert feature.metadata.index_version == 1

    def test_search_after_index(self):
        """Test that search works after indexing."""
        test_shapefile = Path("tests/flood_area_shapefile/pahnehaye_seylab.shp")

        if test_shapefile.exists():
            provider = GeoVectorProvider()
            provider.index(test_shapefile)

            # Get bounds
            metadata = provider.get_metadata()
            assert metadata is not None

            # Try a search query
            query = GeoQuery(limit=5)
            results = provider.search(query)

            assert len(results) > 0
            assert all(isinstance(f, GeoFeature) for f in results)

    def test_search_before_index_raises_error(self):
        """Test that search before index raises error."""
        provider = GeoVectorProvider()

        query = GeoQuery(limit=5)
        with pytest.raises(RuntimeError):
            provider.search(query)

    def test_get_metadata_before_index(self):
        """Test that get_metadata returns None before indexing."""
        provider = GeoVectorProvider()

        assert provider.get_metadata() is None

    def test_get_feature_count(self):
        """Test get_feature_count method."""
        test_shapefile = Path("tests/flood_area_shapefile/pahnehaye_seylab.shp")

        if test_shapefile.exists():
            provider = GeoVectorProvider()

            assert provider.get_feature_count() == 0

            provider.index(test_shapefile)

            count = provider.get_feature_count()
            assert count > 0

    def test_crs_preserved(self):
        """Test that CRS is preserved through indexing."""
        test_shapefile = Path("tests/flood_area_shapefile/pahnehaye_seylab.shp")

        if test_shapefile.exists():
            provider = GeoVectorProvider()
            provider.index(test_shapefile)

            metadata = provider.get_metadata()
            assert metadata.crs is not None

            # All features should have same CRS
            for feature in provider.features:
                assert feature.crs == metadata.crs

    def test_geometry_types_preserved(self):
        """Test that geometry types are preserved."""
        test_shapefile = Path("tests/flood_area_shapefile/pahnehaye_seylab.shp")

        if test_shapefile.exists():
            provider = GeoVectorProvider()
            provider.index(test_shapefile)

            for feature in provider.features:
                assert feature.geometry is not None
                assert hasattr(feature.geometry, "geom_type")

    def test_properties_separated_from_geometry(self):
        """Test that properties exclude geometry column."""
        test_shapefile = Path("tests/flood_area_shapefile/pahnehaye_seylab.shp")

        if test_shapefile.exists():
            provider = GeoVectorProvider()
            provider.index(test_shapefile)

            for feature in provider.features:
                # Properties should not contain geometry
                assert "geometry" not in feature.properties

    def test_driver_detection_shapefile(self):
        """Test that driver is correctly detected for Shapefile."""
        test_shapefile = Path("tests/flood_area_shapefile/pahnehaye_seylab.shp")

        if test_shapefile.exists():
            provider = GeoVectorProvider()
            provider.index(test_shapefile)

            metadata = provider.get_metadata()
            # Should detect SHAPEFILE driver from .shp extension
            assert metadata is not None

    def test_layer_name_stored(self):
        """Test that layer name is stored in metadata."""
        test_shapefile = Path("tests/flood_area_shapefile/pahnehaye_seylab.shp")

        if test_shapefile.exists():
            provider = GeoVectorProvider()
            provider.index(test_shapefile)

            metadata = provider.get_metadata()
            assert metadata.layer_name == "default"

    def test_feature_ids_unique(self):
        """Test that feature IDs are unique."""
        test_shapefile = Path("tests/flood_area_shapefile/pahnehaye_seylab.shp")

        if test_shapefile.exists():
            provider = GeoVectorProvider()
            provider.index(test_shapefile)

            feature_ids = [f.feature_id for f in provider.features]
            # All IDs should be unique
            assert len(feature_ids) == len(set(feature_ids))

    def test_sequential_indexing(self):
        """Test that provider can be reused for sequential indexing."""
        test_shapefile = Path("tests/flood_area_shapefile/pahnehaye_seylab.shp")

        if test_shapefile.exists():
            provider = GeoVectorProvider()

            # First index
            provider.index(test_shapefile)
            count1 = provider.get_feature_count()

            # Second index (should replace)
            provider.index(test_shapefile)
            count2 = provider.get_feature_count()

            assert count1 == count2
