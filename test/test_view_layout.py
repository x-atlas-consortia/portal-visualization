"""Tests for view_layout module."""

from unittest.mock import Mock

import pytest


@pytest.mark.requires_full
class TestViewLayoutConfig:
    """Test ViewLayoutConfig class."""

    def test_get_full_layout_expression_spatial_with_heatmap(self):
        """Test get_full_layout_expression for spatial with heatmap."""
        from portal_visualization.view_layout import ViewLayoutConfig

        config = ViewLayoutConfig(minimal=False, is_spatial=True, include_heatmap=True)

        # Lines 132-137: Test the expression generation
        expression = config.get_full_layout_expression()
        assert expression == "((scatterplot | spatial) / heatmap) | ((cell_sets | gene_list) / cell_sets_expr)"

    def test_get_full_layout_expression_spatial_without_heatmap(self):
        """Test get_full_layout_expression for spatial without heatmap."""
        from portal_visualization.view_layout import ViewLayoutConfig

        config = ViewLayoutConfig(minimal=False, is_spatial=True, include_heatmap=False)

        expression = config.get_full_layout_expression()
        assert expression == "(scatterplot | spatial) | ((cell_sets | gene_list) / cell_sets_expr)"

    def test_get_full_layout_expression_non_spatial_with_heatmap(self):
        """Test get_full_layout_expression for non-spatial with heatmap."""
        from portal_visualization.view_layout import ViewLayoutConfig

        config = ViewLayoutConfig(minimal=False, is_spatial=False, include_heatmap=True)

        expression = config.get_full_layout_expression()
        assert expression == "(scatterplot / heatmap) | ((cell_sets | gene_list) / cell_sets_expr)"

    def test_get_full_layout_expression_non_spatial_without_heatmap(self):
        """Test get_full_layout_expression for non-spatial without heatmap."""
        from portal_visualization.view_layout import ViewLayoutConfig

        config = ViewLayoutConfig(minimal=False, is_spatial=False, include_heatmap=False)

        expression = config.get_full_layout_expression()
        assert expression == "scatterplot | ((cell_sets | gene_list) / cell_sets_expr)"

    def test_apply_full_layout_spatial_with_heatmap(self):
        """Test apply_full_layout for spatial dataset with heatmap."""
        from portal_visualization.view_layout import ViewLayoutConfig

        # Create mock Vitessce config
        vc = Mock()

        # Create mock view objects that support | and / operators
        class MockView:
            """Mock Vitessce view object."""

            def __init__(self, name):
                self.name = name

            def __or__(self, other):
                """Mock the | operator."""
                return MockView(f"({self.name} | {other.name})")

            def __truediv__(self, other):
                """Mock the / operator."""
                return MockView(f"({self.name} / {other.name})")

            def set_xywh(self, x, y, w, h):
                """Mock set_xywh method."""
                self.x = x
                self.y = y
                self.w = w
                self.h = h

        scatterplot = MockView("scatterplot")
        spatial = MockView("spatial")
        heatmap = MockView("heatmap")
        cell_sets = MockView("cell_sets")
        gene_list = MockView("gene_list")
        cell_sets_expr = MockView("cell_sets_expr")

        views_dict = {
            "scatterplot": scatterplot,
            "spatial": spatial,
            "heatmap": heatmap,
            "cell_sets": cell_sets,
            "gene_list": gene_list,
            "cell_sets_expr": cell_sets_expr,
        }

        config = ViewLayoutConfig(minimal=False, is_spatial=True, include_heatmap=True)

        # Lines 155-175: Test apply_full_layout
        config.apply_full_layout(vc, views_dict)

        # Verify vc.layout was called
        assert vc.layout.called
        # Get the layout object that was passed
        layout_arg = vc.layout.call_args[0][0]
        # Verify it's a MockView (the result of the layout expression)
        assert isinstance(layout_arg, MockView)

    def test_apply_full_layout_spatial_without_heatmap(self):
        """Test apply_full_layout for spatial dataset without heatmap."""
        from portal_visualization.view_layout import ViewLayoutConfig

        vc = Mock()

        class MockView:
            def __init__(self, name):
                self.name = name

            def __or__(self, other):
                return MockView(f"({self.name} | {other.name})")

            def __truediv__(self, other):
                return MockView(f"({self.name} / {other.name})")

        scatterplot = MockView("scatterplot")
        spatial = MockView("spatial")
        cell_sets = MockView("cell_sets")
        gene_list = MockView("gene_list")
        cell_sets_expr = MockView("cell_sets_expr")

        views_dict = {
            "scatterplot": scatterplot,
            "spatial": spatial,
            "heatmap": None,
            "cell_sets": cell_sets,
            "gene_list": gene_list,
            "cell_sets_expr": cell_sets_expr,
        }

        config = ViewLayoutConfig(minimal=False, is_spatial=True, include_heatmap=False)
        config.apply_full_layout(vc, views_dict)

        assert vc.layout.called
        layout_arg = vc.layout.call_args[0][0]
        assert isinstance(layout_arg, MockView)

    def test_apply_full_layout_non_spatial_with_heatmap(self):
        """Test apply_full_layout for non-spatial dataset with heatmap."""
        from portal_visualization.view_layout import ViewLayoutConfig

        vc = Mock()

        class MockView:
            def __init__(self, name):
                self.name = name

            def __or__(self, other):
                return MockView(f"({self.name} | {other.name})")

            def __truediv__(self, other):
                return MockView(f"({self.name} / {other.name})")

        scatterplot = MockView("scatterplot")
        heatmap = MockView("heatmap")
        cell_sets = MockView("cell_sets")
        gene_list = MockView("gene_list")
        cell_sets_expr = MockView("cell_sets_expr")

        views_dict = {
            "scatterplot": scatterplot,
            "spatial": None,
            "heatmap": heatmap,
            "cell_sets": cell_sets,
            "gene_list": gene_list,
            "cell_sets_expr": cell_sets_expr,
        }

        config = ViewLayoutConfig(minimal=False, is_spatial=False, include_heatmap=True)
        config.apply_full_layout(vc, views_dict)

        assert vc.layout.called
        layout_arg = vc.layout.call_args[0][0]
        assert isinstance(layout_arg, MockView)

    def test_apply_full_layout_non_spatial_without_heatmap(self):
        """Test apply_full_layout for non-spatial dataset without heatmap."""
        from portal_visualization.view_layout import ViewLayoutConfig

        vc = Mock()

        class MockView:
            def __init__(self, name):
                self.name = name

            def __or__(self, other):
                return MockView(f"({self.name} | {other.name})")

            def __truediv__(self, other):
                return MockView(f"({self.name} / {other.name})")

            def set_xywh(self, x, y, w, h):
                self.x = x
                self.y = y
                self.w = w
                self.h = h

        scatterplot = MockView("scatterplot")
        cell_sets = MockView("cell_sets")
        gene_list = MockView("gene_list")
        cell_sets_expr = MockView("cell_sets_expr")

        views_dict = {
            "scatterplot": scatterplot,
            "spatial": None,
            "heatmap": None,
            "cell_sets": cell_sets,
            "gene_list": gene_list,
            "cell_sets_expr": cell_sets_expr,
        }

        config = ViewLayoutConfig(minimal=False, is_spatial=False, include_heatmap=False)
        config.apply_full_layout(vc, views_dict)

        # Lines 155-175: Verify layout was applied and scatterplot was resized
        assert vc.layout.called
        layout_arg = vc.layout.call_args[0][0]
        assert isinstance(layout_arg, MockView)
        # Verify scatterplot.set_xywh was called
        assert hasattr(scatterplot, "x")
        assert scatterplot.x == 0
        assert scatterplot.y == 0
        assert scatterplot.w == 6
        assert scatterplot.h == 12


@pytest.mark.requires_full
class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_layout_config(self):
        """Test create_layout_config factory function."""
        from portal_visualization.view_layout import create_layout_config

        config = create_layout_config(minimal=False, is_spatial=True, include_heatmap=True)

        from portal_visualization.view_layout import ViewLayoutConfig

        assert isinstance(config, ViewLayoutConfig)
        assert config.minimal is False
        assert config.is_spatial is True
        assert config.include_heatmap is True
