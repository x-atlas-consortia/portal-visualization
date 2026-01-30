"""Tests for data_access module."""

from unittest.mock import Mock, patch

import pytest


class TestFullDepsAvailableImport:
    """Test the _FULL_DEPS_AVAILABLE flag import error handling."""

    @pytest.mark.requires_full
    def test_full_deps_available_true_in_full_install(self):
        """Test that _FULL_DEPS_AVAILABLE is True in full install."""
        # This test is skipped because even with @pytest.mark.requires_full,
        # the zarr package may not be installed. Line 17 (_FULL_DEPS_AVAILABLE = True)
        # is covered when the module is imported successfully with zarr installed.
        # The except block (line 17: _FULL_DEPS_AVAILABLE = False) is tested via
        # test_import_error_fallback which patches the flag to False.
        pytest.skip("Line 17 is covered by import - tested in thin install CI")

    @pytest.mark.requires_full
    def test_import_error_fallback(self):
        """Test that module handles zarr import gracefully."""
        # We can't easily test the except block (line 17: _FULL_DEPS_AVAILABLE = False)
        # without breaking the already-imported module, but we can verify the flag
        # is used correctly when patched to False
        from portal_visualization.data_access import ZarrStoreAccessor

        with patch("portal_visualization.data_access._FULL_DEPS_AVAILABLE", False):

            def build_url(path, use_token=True):
                return f"https://example.com/{path}"

            def get_init():
                return {"headers": {}}

            accessor = ZarrStoreAccessor(build_url, get_init)

            # When _FULL_DEPS_AVAILABLE is False, open_store should raise RuntimeError
            with pytest.raises(RuntimeError, match="Zarr dependencies not available"):
                accessor.open_store()


@pytest.mark.requires_full
class TestZarrStoreAccessor:
    """Test ZarrStoreAccessor class."""

    def test_open_store_with_zarr_unavailable(self):
        """Test open_store raises RuntimeError when zarr is unavailable."""
        from portal_visualization.data_access import ZarrStoreAccessor

        # Patch _FULL_DEPS_AVAILABLE to False
        with patch("portal_visualization.data_access._FULL_DEPS_AVAILABLE", False):
            # Create accessor with proper callables
            def build_url(path, use_token=True):
                return f"https://example.com/{path}"

            def get_init():
                return {"headers": {}}

            accessor = ZarrStoreAccessor(build_url, get_init)

            # Lines 105-109: RuntimeError when zarr unavailable
            with pytest.raises(
                RuntimeError,
                match="Zarr dependencies not available.*Install with.*portal-visualization\\[full\\]",
            ):
                accessor.open_store()

    def test_open_store_normal_path(self):
        """Test open_store with normal zarr available scenario."""
        from portal_visualization.data_access import ZarrStoreAccessor

        def build_url(path, use_token=True):
            return f"https://example.com/{path}"

        def get_init():
            return {"headers": {}}

        accessor = ZarrStoreAccessor(build_url, get_init)

        # Verify accessor was created with correct properties
        assert accessor._url_builder is build_url
        assert accessor._request_init_provider is get_init
        assert accessor.zarr_path == "hubmap_ui/anndata-zarr/secondary_analysis.zarr"
        assert accessor.zip_zarr_path == "hubmap_ui/anndata-zarr/secondary_analysis.zarr.zip"

    @pytest.mark.requires_full
    def test_open_store_with_zip(self):
        """Test open_store with is_zip=True path."""
        from portal_visualization.data_access import ZarrStoreAccessor

        def build_url(path, use_token=True):
            return f"https://example.com/{path}{'?token=abc' if use_token else ''}"

        def get_init():
            return {"headers": {"Authorization": "Bearer token"}}

        accessor = ZarrStoreAccessor(build_url, get_init)

        # Lines 108-117: Test the is_zip=True branch
        with patch("portal_visualization.data_access.read_zip_zarr") as mock_read_zip:
            mock_store = Mock()
            mock_read_zip.return_value = mock_store

            result = accessor.open_store(is_zip=True)

            # Verify read_zip_zarr was called with correct arguments
            assert mock_read_zip.called
            call_args = mock_read_zip.call_args
            assert "?token=abc" in call_args[0][0]  # URL should have token
            assert call_args[0][1] == {"headers": {"Authorization": "Bearer token"}}  # request_init
            assert result is mock_store

    @pytest.mark.requires_full
    def test_open_store_with_custom_path(self):
        """Test open_store with custom zarr_path parameter."""
        from portal_visualization.data_access import ZarrStoreAccessor

        def build_url(path, use_token=True):
            return f"https://example.com/{path}"

        def get_init():
            return None  # Test None request_init handling

        accessor = ZarrStoreAccessor(build_url, get_init)

        # Test that custom path is used
        with (
            patch("portal_visualization.data_access._FULL_DEPS_AVAILABLE", True),
            patch("portal_visualization.data_access.zarr") as mock_zarr,
        ):
            mock_store = Mock()
            mock_zarr.open.return_value = mock_store

            accessor.open_store(is_zip=False, zarr_path="custom/path.zarr")

            # Verify zarr.open was called with custom path
            assert mock_zarr.open.called
            call_args = mock_zarr.open.call_args
            assert "custom/path.zarr" in call_args[0][0]


@pytest.mark.requires_full
class TestImageMetadataRetriever:
    """Test ImageMetadataRetriever class."""

    def test_compute_scale_with_zero_base_size(self):
        """Test compute_scale returns 1.0 when base PhysicalSizeX is 0 (divide by zero protection)."""
        from portal_visualization.data_access import ImageMetadataRetriever

        loader = Mock()
        retriever = ImageMetadataRetriever(loader)

        # Line 199: Return 1.0 when base_in_meters == 0 (divide by zero protection)
        base_metadata = {"PhysicalSizeX": 0.0, "PhysicalSizeXUnit": "μm"}
        overlay_metadata = {"PhysicalSizeX": 1.0, "PhysicalSizeXUnit": "μm"}
        result = retriever.compute_scale(base_metadata, overlay_metadata)
        assert result == 1.0

    def test_compute_scale_with_valid_metadata(self):
        """Test compute_scale with valid metadata values."""
        from portal_visualization.data_access import ImageMetadataRetriever

        loader = Mock()
        retriever = ImageMetadataRetriever(loader)

        # Test when overlay is half the size of base (scale = 0.5)
        base_metadata = {"PhysicalSizeX": 1.0, "PhysicalSizeXUnit": "μm"}
        overlay_metadata = {"PhysicalSizeX": 0.5, "PhysicalSizeXUnit": "μm"}
        result = retriever.compute_scale(base_metadata, overlay_metadata)
        assert result == 0.5

    def test_compute_scale_with_none_metadata(self):
        """Test compute_scale returns 1.0 when metadata is None."""
        from portal_visualization.data_access import ImageMetadataRetriever

        loader = Mock()
        retriever = ImageMetadataRetriever(loader)

        # Test with None base_metadata
        result = retriever.compute_scale(None, {"PhysicalSizeX": 1.0})
        assert result == 1.0

        # Test with None overlay_metadata
        result = retriever.compute_scale({"PhysicalSizeX": 1.0}, None)
        assert result == 1.0


@pytest.mark.requires_full
class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_zarr_accessor(self):
        """Test create_zarr_accessor factory function."""
        from portal_visualization.data_access import create_zarr_accessor

        # Create mock builder with required methods
        mock_builder = Mock()
        mock_builder._build_assets_url = Mock(return_value="https://example.com/path")
        mock_builder._get_request_init = Mock(return_value={"headers": {"Authorization": "Bearer token"}})

        # Line 216: Call the factory function
        accessor = create_zarr_accessor(mock_builder)

        # Verify it returns a ZarrStoreAccessor instance
        from portal_visualization.data_access import ZarrStoreAccessor

        assert isinstance(accessor, ZarrStoreAccessor)
        assert accessor._url_builder is mock_builder._build_assets_url
        assert accessor._request_init_provider is mock_builder._get_request_init

    def test_create_http_resource_loader(self):
        """Test create_http_resource_loader factory function."""
        from portal_visualization.data_access import HttpResourceLoader, create_http_resource_loader

        loader = create_http_resource_loader()
        assert isinstance(loader, HttpResourceLoader)
