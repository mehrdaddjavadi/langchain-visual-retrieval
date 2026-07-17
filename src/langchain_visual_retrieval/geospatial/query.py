"""Query models for geospatial retrieval."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union
from shapely.geometry.base import BaseGeometry


@dataclass
class GeoQuery:
    """Spatial and attribute filtering query.
    
    Supports spatial predicates (bbox, intersects, contains, nearest_to)
    combined with attribute filtering (where).
    At most one spatial predicate should be set per query.
    """

    bbox: Optional[Tuple[float, float, float, float]] = None  # (minx, miny, maxx, maxy)
    intersects: Optional[BaseGeometry] = None  # Geometry to intersect
    contains: Optional[BaseGeometry] = None  # Geometry to contain
    nearest_to: Optional[Tuple[BaseGeometry, int]] = None  # (geometry, k) for k-nearest
    where: Optional[Union[Dict[str, Any], str]] = None  # Attribute filter: dict or string expression
    limit: int = 5  # Result limit
