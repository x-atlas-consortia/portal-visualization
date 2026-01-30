"""
View layout configuration for portal-visualization builders.

This module centralizes the complex nested conditional logic for view layouts
that appears in multiple builders, reducing code duplication and improving maintainability.
"""

from dataclasses import dataclass


@dataclass
class ViewLayoutConfig:
    """Configuration for Vitessce view layouts.

    Encapsulates the layout decision logic based on:
    - minimal vs full mode
    - spatial vs non-spatial
    - presence of heatmap

    This replaces the 4-level nested conditionals that appear in multiple builders.

    Attributes:
        minimal: Whether to use minimal layout (fewer views)
        is_spatial: Whether spatial view is present
        include_heatmap: Whether heatmap should be included
    """

    minimal: bool = False
    is_spatial: bool = False
    include_heatmap: bool = False

    def get_minimal_spatial_dimensions(self) -> dict:
        """Get view dimensions for minimal spatial layout.

        Returns:
            Dictionary mapping view names to (x, y, w, h) tuples

        >>> config = ViewLayoutConfig(minimal=True, is_spatial=True)
        >>> dims = config.get_minimal_spatial_dimensions()
        >>> dims['scatterplot']
        {'x': 0, 'y': 0, 'w': 6, 'h': 6}
        >>> dims['spatial']
        {'x': 0, 'y': 6, 'w': 6, 'h': 6}
        """
        return {
            "scatterplot": {"x": 0, "y": 0, "w": 6, "h": 6},
            "spatial": {"x": 0, "y": 6, "w": 6, "h": 6},
            "cell_sets": {"x": 6, "y": 0, "w": 6, "h": 4},
            "cell_sets_expr": {"x": 6, "y": 4, "w": 6, "h": 8},
        }

    def get_minimal_nonspatial_dimensions(self) -> dict:
        """Get view dimensions for minimal non-spatial layout.

        Returns:
            Dictionary mapping view names to (x, y, w, h) tuples

        >>> config = ViewLayoutConfig(minimal=True, is_spatial=False)
        >>> dims = config.get_minimal_nonspatial_dimensions()
        >>> dims['scatterplot']
        {'x': 0, 'y': 0, 'w': 6, 'h': 12}
        """
        return {
            "scatterplot": {"x": 0, "y": 0, "w": 6, "h": 12},
            "cell_sets": {"x": 6, "y": 0, "w": 6, "h": 4},
            "cell_sets_expr": {"x": 6, "y": 4, "w": 6, "h": 8},
        }

    def get_minimal_views(self) -> list[str]:
        """Get list of view names to include in minimal mode.

        Returns:
            List of view names

        >>> config = ViewLayoutConfig(minimal=True, is_spatial=True)
        >>> config.get_minimal_views()
        ['scatterplot', 'spatial', 'cell_sets_expr']
        >>> config = ViewLayoutConfig(minimal=True, is_spatial=False)
        >>> config.get_minimal_views()
        ['scatterplot', 'cell_sets_expr']
        """
        if self.is_spatial:
            return ["scatterplot", "spatial", "cell_sets_expr"]
        else:
            return ["scatterplot", "cell_sets_expr"]

    def apply_minimal_layout(self, views_dict: dict) -> list:
        """Apply minimal layout to view objects.

        Args:
            views_dict: Dictionary mapping view names to view objects

        Returns:
            List of views to display in minimal mode

        >>> from unittest.mock import Mock
        >>> scatterplot = Mock()
        >>> spatial = Mock()
        >>> cell_sets_expr = Mock()
        >>> views = {'scatterplot': scatterplot, 'spatial': spatial, 'cell_sets_expr': cell_sets_expr}
        >>> config = ViewLayoutConfig(minimal=True, is_spatial=True)
        >>> result = config.apply_minimal_layout(views)
        >>> len(result)
        3
        >>> scatterplot.set_xywh.assert_called_once_with(x=0, y=0, w=6, h=6)
        """
        dims = self.get_minimal_spatial_dimensions() if self.is_spatial else self.get_minimal_nonspatial_dimensions()

        # Apply dimensions to views
        for view_name, dim in dims.items():
            if view_name in views_dict and views_dict[view_name] is not None:
                views_dict[view_name].set_xywh(**dim)

        # Return selected views
        view_names = self.get_minimal_views()
        return [views_dict[name] for name in view_names if name in views_dict and views_dict[name] is not None]

    def get_full_layout_expression(self) -> str:
        """Get layout expression for full (non-minimal) mode.

        Returns:
            String describing the layout (for documentation/debugging)

        >>> config = ViewLayoutConfig(minimal=False, is_spatial=True, include_heatmap=True)
        >>> config.get_full_layout_expression()
        '((scatterplot | spatial) / heatmap) | ((cell_sets | gene_list) / cell_sets_expr)'
        """
        if self.is_spatial:
            if self.include_heatmap:
                return "((scatterplot | spatial) / heatmap) | ((cell_sets | gene_list) / cell_sets_expr)"
            else:
                return "(scatterplot | spatial) | ((cell_sets | gene_list) / cell_sets_expr)"
        else:
            if self.include_heatmap:
                return "(scatterplot / heatmap) | ((cell_sets | gene_list) / cell_sets_expr)"
            else:
                return "scatterplot | ((cell_sets | gene_list) / cell_sets_expr)"

    def apply_full_layout(self, vc, views_dict: dict) -> None:
        """Apply full layout to Vitessce config.

        Args:
            vc: Vitessce config object
            views_dict: Dictionary mapping view names to view objects

        Example usage:
            from unittest.mock import Mock
            vc = Mock()
            # Create view mocks with proper __or__ and __truediv__ support
            views = {...}  # Dictionary of view objects
            config = ViewLayoutConfig(minimal=False, is_spatial=True, include_heatmap=True)
            config.apply_full_layout(vc, views)
            # vc.layout will be called with appropriate layout expression
        """
        scatterplot = views_dict.get("scatterplot")
        spatial = views_dict.get("spatial")
        heatmap = views_dict.get("heatmap")
        cell_sets = views_dict.get("cell_sets")
        gene_list = views_dict.get("gene_list")
        cell_sets_expr = views_dict.get("cell_sets_expr")

        if self.is_spatial:
            if self.include_heatmap and heatmap is not None:
                vc.layout(((scatterplot | spatial) / heatmap) | ((cell_sets | gene_list) / cell_sets_expr))
            else:
                # When heatmap is hidden, expand scatterplot/spatial vertically to fill the space
                vc.layout((scatterplot | spatial) | ((cell_sets | gene_list) / cell_sets_expr))
        else:
            if self.include_heatmap and heatmap is not None:
                vc.layout((scatterplot / heatmap) | ((cell_sets | gene_list) / cell_sets_expr))
            else:
                # When heatmap is hidden, expand scatterplot vertically to fill the space
                vc.layout(scatterplot | ((cell_sets | gene_list) / cell_sets_expr))
                if scatterplot is not None:
                    scatterplot.set_xywh(x=0, y=0, w=6, h=12)


def create_layout_config(minimal: bool, is_spatial: bool, include_heatmap: bool) -> ViewLayoutConfig:
    """Factory function for creating ViewLayoutConfig instances.

    Args:
        minimal: Whether to use minimal layout
        is_spatial: Whether spatial view is present
        include_heatmap: Whether to include heatmap

    Returns:
        Configured ViewLayoutConfig instance

    >>> config = create_layout_config(minimal=True, is_spatial=False, include_heatmap=False)
    >>> config.minimal
    True
    >>> config.is_spatial
    False
    """
    return ViewLayoutConfig(minimal=minimal, is_spatial=is_spatial, include_heatmap=include_heatmap)


# Standard dimension constants used across multiple builders
class ViewDimensions:
    """Standard view dimensions for common layouts.

    Centralizes hardcoded w=6, h=12 values that appear throughout builders.
    """

    # Full width views
    FULL_WIDTH_HALF_HEIGHT = {"w": 12, "h": 6}
    FULL_WIDTH_FULL_HEIGHT = {"w": 12, "h": 12}

    # Half width views
    HALF_WIDTH_HALF_HEIGHT = {"w": 6, "h": 6}
    HALF_WIDTH_FULL_HEIGHT = {"w": 6, "h": 12}
    HALF_WIDTH_THIRD_HEIGHT = {"w": 6, "h": 4}
    HALF_WIDTH_TWO_THIRDS_HEIGHT = {"w": 6, "h": 8}

    # Quarter width views
    QUARTER_WIDTH_FULL_HEIGHT = {"w": 3, "h": 12}

    @staticmethod
    def at_position(x: int, y: int, w: int, h: int) -> dict:
        """Create dimension dict with position.

        Args:
            x: X coordinate
            y: Y coordinate
            w: Width
            h: Height

        Returns:
            Dictionary with x, y, w, h keys

        >>> ViewDimensions.at_position(0, 0, 6, 12)
        {'x': 0, 'y': 0, 'w': 6, 'h': 12}
        """
        return {"x": x, "y": y, "w": w, "h": h}
