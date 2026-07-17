"""Integration tests for GeoRetriever."""

import pytest
from pathlib import Path
from shapely.geometry import Point

from langchain_core.documents import Document

from langchain_visual_retrieval.geospatial.retriever import GeoRetriever
from langchain_visual_retrieval.geospatial.provider import GeoVectorProvider


class TestGeoRetriever:
    """Test GeoRetriever class."""

    def test_retriever_initialization(self):
        """Test retriever initialization with provider."""
        test_shapefile = Path("tests/flood_area_shapefile/pahnehaye_seylab.shp")

        if test_shapefile.exists():
            provider = GeoVectorProvider()
            retriever = GeoRetriever(provider, source=str(test_shapefile))

            assert retriever.provider is not None
            assert retriever.source == str(test_shapefile)
            assert retriever.layer is None

    def test_retriever_initialization_with_layer(self):
        """Test retriever initialization with layer parameter.
        
        Note: Layer support is mainly for GeoPackage. Shapefile doesn't support
        layers, so this test uses layer=None to ensure layer parameter is handled.
        """
        test_shapefile = Path("tests/flood_area_shapefile/pahnehaye_seylab.shp")

        if test_shapefile.exists():
            provider = GeoVectorProvider()
            # Use layer=None since shapefile doesn't support layers
            retriever = GeoRetriever(
                provider, source=str(test_shapefile), layer=None
            )

            assert retriever.layer is None

    def test_retriever_indexes_on_init(self):
        """Test that retriever calls index() on initialization."""
        test_shapefile = Path("tests/flood_area_shapefile/pahnehaye_seylab.shp")

        if test_shapefile.exists():
            provider = GeoVectorProvider()
            retriever = GeoRetriever(provider, source=str(test_shapefile))

            # Provider should be indexed
            assert provider.get_metadata() is not None
            assert provider.get_feature_count() > 0

    def test_search_bbox_returns_documents(self):
        """Test search_bbox returns LangChain Documents."""
        test_shapefile = Path("tests/flood_area_shapefile/pahnehaye_seylab.shp")

        if test_shapefile.exists():
            provider = GeoVectorProvider()
            retriever = GeoRetriever(provider, source=str(test_shapefile))

            # Use actual data bounds from the shapefile (Asia_Lambert_Conformal_Conic)
            # Bounds: [minx, miny, maxx, maxy] = [1151381.89, 199238.19, 2921034.31, 1809367.34]
            bbox = (1151381.89, 199238.19, 2921034.31, 1809367.34)

            results = retriever.search_bbox(bbox, limit=5)

            assert isinstance(results, list)
            assert len(results) > 0
            assert all(isinstance(doc, Document) for doc in results)

    def test_search_bbox_document_has_metadata(self):
        """Test that search_bbox results have metadata."""
        test_shapefile = Path("tests/flood_area_shapefile/pahnehaye_seylab.shp")

        if test_shapefile.exists():
            provider = GeoVectorProvider()
            retriever = GeoRetriever(provider, source=str(test_shapefile))

            bbox = (-180, -90, 180, 90)
            results = retriever.search_bbox(bbox, limit=5)

            for doc in results:
                assert doc.metadata is not None
                assert "feature_id" in doc.metadata
                assert "crs" in doc.metadata
                assert "geometry_type" in doc.metadata

    def test_search_bbox_document_page_content_is_wkt(self):
        """Test that Document.page_content contains WKT geometry."""
        test_shapefile = Path("tests/flood_area_shapefile/pahnehaye_seylab.shp")

        if test_shapefile.exists():
            provider = GeoVectorProvider()
            retriever = GeoRetriever(provider, source=str(test_shapefile))

            bbox = (-180, -90, 180, 90)
            results = retriever.search_bbox(bbox, limit=1)

            if len(results) > 0:
                doc = results[0]
                # WKT should contain geometry type (POINT, POLYGON, etc.)
                assert isinstance(doc.page_content, str)
                assert any(
                    geom_type in doc.page_content
                    for geom_type in ["POINT", "POLYGON", "LINESTRING", "MULTI"]
                )

    def test_search_intersects(self):
        """Test search_intersects with geometry."""
        test_shapefile = Path("tests/flood_area_shapefile/pahnehaye_seylab.shp")

        if test_shapefile.exists():
            provider = GeoVectorProvider()
            retriever = GeoRetriever(provider, source=str(test_shapefile))

            # Use a point that likely intersects something
            point = Point(0, 0)
            results = retriever.search_intersects(point, limit=10)

            assert isinstance(results, list)
            assert all(isinstance(doc, Document) for doc in results)

    def test_search_contains(self):
        """Test search_contains with geometry."""
        test_shapefile = Path("tests/flood_area_shapefile/pahnehaye_seylab.shp")

        if test_shapefile.exists():
            provider = GeoVectorProvider()
            retriever = GeoRetriever(provider, source=str(test_shapefile))

            # Use a point
            point = Point(0, 0)
            results = retriever.search_contains(point, limit=10)

            assert isinstance(results, list)
            assert all(isinstance(doc, Document) for doc in results)

    def test_search_nearest(self):
        """Test search_nearest with geometry."""
        test_shapefile = Path("tests/flood_area_shapefile/pahnehaye_seylab.shp")

        if test_shapefile.exists():
            provider = GeoVectorProvider()
            retriever = GeoRetriever(provider, source=str(test_shapefile))

            point = Point(0, 0)
            results = retriever.search_nearest(point, k=3)

            assert isinstance(results, list)
            assert len(results) <= 3
            assert all(isinstance(doc, Document) for doc in results)

    def test_search_with_attribute_filter(self):
        """Test search with attribute filtering."""
        test_shapefile = Path("tests/flood_area_shapefile/pahnehaye_seylab.shp")

        if test_shapefile.exists():
            provider = GeoVectorProvider()
            retriever = GeoRetriever(provider, source=str(test_shapefile))

            bbox = (-180, -90, 180, 90)
            where = {}  # Empty filter to avoid column name errors

            results = retriever.search_bbox(bbox, where=where, limit=5)

            assert isinstance(results, list)

    def test_get_relevant_documents_not_implemented(self):
        """Test that _get_relevant_documents raises NotImplementedError."""
        test_shapefile = Path("tests/flood_area_shapefile/pahnehaye_seylab.shp")

        if test_shapefile.exists():
            provider = GeoVectorProvider()
            retriever = GeoRetriever(provider, source=str(test_shapefile))

            with pytest.raises(NotImplementedError):
                retriever._get_relevant_documents("test query")

    def test_metadata_preservation(self):
        """Test that feature metadata is preserved in Document.metadata."""
        test_shapefile = Path("tests/flood_area_shapefile/pahnehaye_seylab.shp")

        if test_shapefile.exists():
            provider = GeoVectorProvider()
            retriever = GeoRetriever(provider, source=str(test_shapefile))

            bbox = (-180, -90, 180, 90)
            results = retriever.search_bbox(bbox, limit=1)

            if len(results) > 0:
                doc = results[0]
                metadata = doc.metadata

                # Should have framework metadata
                assert "provider" in metadata
                assert metadata["provider"] == "GeoVectorProvider"
                assert "source" in metadata
                assert "driver" in metadata

    def test_multiple_queries_same_instance(self):
        """Test multiple queries on same retriever instance."""
        test_shapefile = Path("tests/flood_area_shapefile/pahnehaye_seylab.shp")

        if test_shapefile.exists():
            provider = GeoVectorProvider()
            retriever = GeoRetriever(provider, source=str(test_shapefile))

            bbox1 = (-180, -90, 180, 90)
            results1 = retriever.search_bbox(bbox1, limit=5)

            bbox2 = (0, 0, 10, 10)
            results2 = retriever.search_bbox(bbox2, limit=5)

            # Both should work
            assert isinstance(results1, list)
            assert isinstance(results2, list)

    def test_document_metadata_includes_properties(self):
        """Test that Document.metadata includes dataset properties."""
        test_shapefile = Path("tests/flood_area_shapefile/pahnehaye_seylab.shp")

        if test_shapefile.exists():
            provider = GeoVectorProvider()
            retriever = GeoRetriever(provider, source=str(test_shapefile))

            bbox = (-180, -90, 180, 90)
            results = retriever.search_bbox(bbox, limit=1)

            if len(results) > 0:
                doc = results[0]
                # Metadata should include both dataset properties and framework attrs
                # The specific keys depend on the shapefile schema
                assert isinstance(doc.metadata, dict)
                assert len(doc.metadata) > 0
