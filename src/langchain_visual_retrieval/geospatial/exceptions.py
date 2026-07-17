"""Custom exceptions for geospatial retrieval."""


class UnsupportedGeoFormatError(Exception):
    """Raised when file format is not Shapefile or GeoPackage."""

    pass


class MissingCRSError(Exception):
    """Raised when CRS cannot be determined from source."""

    pass


class InvalidGeometryError(Exception):
    """Raised when geometry is invalid, corrupt, or empty."""

    pass


class SpatialQueryError(Exception):
    """Raised when spatial query fails (invalid parameters, etc.)."""

    pass


class GeoIndexBuildError(Exception):
    """Raised when spatial index construction fails."""

    pass


class EmptyDatasetError(Exception):
    """Raised when dataset has no features."""

    pass
