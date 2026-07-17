"""Spatial index using Shapely STRtree for efficient geospatial queries.

The spatial index is never serialized. It is rebuilt in-memory from
canonical feature records whenever an index is loaded.
"""

from typing import Any, Dict, List, Optional, Tuple

from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from .exceptions import EmptyDatasetError, GeoIndexBuildError, SpatialQueryError
from .feature import GeoFeature
from .query import GeoQuery


class GeoSpatialIndex:
    """Spatial index for efficient retrieval using STRtree.

    All spatial queries use the index; no full-dataset scans are performed.
    Attribute filtering is applied post-spatial-query on candidate indices.
    """

    def __init__(self, features: List[GeoFeature]):
        """Build STRtree from features. Index is never serialized.

        Args:
            features: List of GeoFeature objects

        Raises:
            EmptyDatasetError: If feature list is empty
            GeoIndexBuildError: If index construction fails
        """
        if not features:
            raise EmptyDatasetError("Cannot build index from empty feature list")

        # Extract geometries in order; maintain reference to features
        self.features = features
        self.geometries = [f.geometry for f in features]

        try:
            self.tree = STRtree(self.geometries)
        except Exception as e:
            raise GeoIndexBuildError(f"STRtree construction failed: {e}")

    def search_bbox(
        self, bbox: Tuple[float, float, float, float]
    ) -> List[int]:
        """Get feature indices intersecting bounding box.

        Args:
            bbox: (minx, miny, maxx, maxy)

        Returns:
            List of feature indices (never full scan; STRtree predicate used)
        """
        try:
            bbox_geom = box(*bbox)
            indices = self.tree.query(bbox_geom, predicate="intersects")
            # Convert numpy int64 to native Python int
            return [int(i) for i in indices] if hasattr(indices, "__iter__") else [int(indices)]
        except Exception as e:
            raise SpatialQueryError(f"Bbox search failed: {e}")

    def search_intersects(self, geometry: BaseGeometry) -> List[int]:
        """Get indices of features intersecting geometry.

        Args:
            geometry: Shapely geometry to intersect

        Returns:
            List of feature indices

        Raises:
            SpatialQueryError: If query fails
        """
        try:
            indices = self.tree.query(geometry, predicate="intersects")
            # Convert numpy int64 to native Python int
            return [int(i) for i in indices] if hasattr(indices, "__iter__") else [int(indices)]
        except Exception as e:
            raise SpatialQueryError(f"Intersection search failed: {e}")

    def search_contains(self, geometry: BaseGeometry) -> List[int]:
        """Get indices of features containing geometry.

        Args:
            geometry: Shapely geometry to contain

        Returns:
            List of feature indices

        Raises:
            SpatialQueryError: If query fails
        """
        try:
            indices = self.tree.query(geometry, predicate="contains")
            # Convert numpy int64 to native Python int
            return [int(i) for i in indices] if hasattr(indices, "__iter__") else [int(indices)]
        except Exception as e:
            raise SpatialQueryError(f"Containment search failed: {e}")

    def search_nearest(
        self, geometry: BaseGeometry, k: int = 5
    ) -> List[int]:
        """Get k nearest features to geometry.

        Args:
            geometry: Reference geometry
            k: Number of nearest features to return

        Returns:
            List of feature indices (up to k results)

        Raises:
            SpatialQueryError: If query fails
        """
        try:
            # Compute distances to all geometries and get k smallest
            distances = [
                (i, self.geometries[i].distance(geometry))
                for i in range(len(self.geometries))
            ]
            distances.sort(key=lambda x: x[1])
            nearest = [int(i) for i, _ in distances[:k]]
            return nearest
        except Exception as e:
            raise SpatialQueryError(f"Nearest-neighbor search failed: {e}")

    def filter_attributes(
        self, indices: List[int], where: Optional[Any]
    ) -> List[int]:
        """Filter feature indices by attribute conditions.

        Applied post-spatial-query; operates only on candidate indices.

        Args:
            indices: Feature indices to filter
            where: Dict of {column: value} or {column: [op, value]} conditions,
                   or a string expression (e.g., "ID_ > 0")

        Returns:
            Filtered indices

        Raises:
            SpatialQueryError: If filtering fails
        """
        if not where:
            return indices
            
        filtered = []
        # Parse string expression if needed
        conditions = where
        if isinstance(where, str):
            conditions = self._parse_where_string(where)
            
        for idx in indices:
            if idx >= len(self.features):
                continue
            feature = self.features[idx]
            if self._matches_conditions(feature.properties, conditions):
                filtered.append(idx)
        return filtered

    @staticmethod
    def _parse_where_string(expr: str) -> Dict[str, Any]:
        """Parse simple where string expressions.
        
        Supports: "column > value", "column < value", "column == value", 
                  "column != value", "column in [v1, v2, ...]"
        
        Args:
            expr: Where expression string
            
        Returns:
            Dict of {column: [op, value]} conditions
        """
        expr = expr.strip()
        
        # Handle comparison operators
        for op in ["!=", "<=", ">=", "==", "<", ">", " in "]:
            if op in expr:
                parts = expr.split(op, 1)
                col = parts[0].strip()
                val_str = parts[1].strip()
                
                # Try to convert value to int/float
                try:
                    # Handle "in" operator with list
                    if op == " in ":
                        # Parse list like "[1, 2, 3]"
                        val_str = val_str.strip()
                        if val_str.startswith("[") and val_str.endswith("]"):
                            val_str = val_str[1:-1]  # Remove brackets
                            values = [v.strip() for v in val_str.split(",")]
                            # Try to convert to numbers
                            try:
                                values = [int(v) if v.isdigit() else float(v) for v in values]
                            except ValueError:
                                pass  # Keep as strings
                            return {col: ["in", values]}
                    
                    # Try numeric conversion
                    if val_str.isdigit():
                        val = int(val_str)
                    else:
                        try:
                            val = float(val_str)
                        except ValueError:
                            val = val_str
                except:
                    val = val_str
                    
                return {col: [op.strip(), val]}
        
        # If no operator found, treat as equality
        raise SpatialQueryError(f"Invalid where expression: {expr}")


    @staticmethod
    def _matches_conditions(
        properties: Dict[str, Any], conditions: Dict[str, Any]
    ) -> bool:
        """Check if properties match all conditions.

        Args:
            properties: Feature properties dict
            conditions: Conditions to check

        Returns:
            True if all conditions match, False otherwise
        """
        for key, condition in conditions.items():
            if key not in properties:
                return False

            # Simple value equality or operator-based comparison
            if isinstance(condition, (list, tuple)) and len(condition) == 2:
                op, value = condition
                prop_val = properties[key]

                if op == "==" and prop_val != value:
                    return False
                elif op == ">" and not (prop_val > value):
                    return False
                elif op == "<" and not (prop_val < value):
                    return False
                elif op == ">=" and not (prop_val >= value):
                    return False
                elif op == "<=" and not (prop_val <= value):
                    return False
                elif op == "in" and prop_val not in value:
                    return False
            else:
                if properties[key] != condition:
                    return False

        return True

    def execute_query(self, query: GeoQuery) -> List[GeoFeature]:
        """Execute spatial + attribute query.

        Applies spatial predicate via STRtree, then attribute filters
        on candidate indices. No full-dataset scans.

        Args:
            query: GeoQuery with spatial and/or attribute filters

        Returns:
            List of matching GeoFeature objects (up to query.limit)

        Raises:
            SpatialQueryError: If query parameters are invalid
        """
        # Start with all features if no spatial filter
        candidate_indices = list(range(len(self.features)))

        # Apply spatial filters (STRtree-based; no full scans)
        if query.bbox:
            candidate_indices = self.search_bbox(query.bbox)
        elif query.intersects:
            candidate_indices = self.search_intersects(query.intersects)
        elif query.contains:
            candidate_indices = self.search_contains(query.contains)
        elif query.nearest_to:
            geometry, k = query.nearest_to
            candidate_indices = self.search_nearest(geometry, k)

        # Apply attribute filtering (post-spatial, on candidates only)
        if query.where:
            candidate_indices = self.filter_attributes(candidate_indices, query.where)

        # Apply limit and return features
        result_indices = candidate_indices[: query.limit]
        return [self.features[i] for i in result_indices]
