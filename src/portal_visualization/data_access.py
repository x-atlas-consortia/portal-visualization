"""
Data access abstractions for portal-visualization builders.

This module provides common patterns for accessing external resources (Zarr stores,
HTTP endpoints, image metadata) to reduce code duplication across builders and
improve testability.
"""

from abc import ABC, abstractmethod
from typing import Any

try:
    import zarr

    from .utils import read_zip_zarr

    _FULL_DEPS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FULL_DEPS_AVAILABLE = False


class ResourceLoader(ABC):
    """Abstract interface for loading external resources.

    Allows builders to be tested without making real HTTP requests or file system access.
    """

    @abstractmethod
    def load_json(self, url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
        """Load JSON from a URL.

        Args:
            url: The URL to load from
            headers: Optional HTTP headers

        Returns:
            Parsed JSON as a dictionary
        """
        # pragma: no cover


class HttpResourceLoader(ResourceLoader):
    """Production resource loader using requests library."""

    def load_json(self, url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
        """Load JSON from a URL using requests.

        >>> # This is production code, tested via integration tests
        >>> # Example usage:
        >>> # loader = HttpResourceLoader()
        >>> # data = loader.load_json("https://api.example.com/data.json")
        """
        import requests  # pragma: no cover

        resp = requests.get(url, headers=headers)  # pragma: no cover
        resp.raise_for_status()  # pragma: no cover
        return resp.json()  # pragma: no cover


class ZarrStoreAccessor:
    """Manages access to Zarr stores with support for zip-compressed and regular formats.

    This class encapsulates the common pattern of opening Zarr stores that appears
    in multiple builders (anndata_builders.py, sprm_builders.py, object_by_analyte_builders.py).
    """

    def __init__(self, url_builder, request_init_provider):
        """Initialize the accessor.

        Args:
            url_builder: Callable that takes (rel_path, use_token=bool) and returns URL string
            request_init_provider: Callable that returns dict with request headers/config

        >>> # Example with mock functions:
        >>> def build_url(path, use_token=True):
        ...     return f"https://example.com/{path}"
        >>> def get_init():
        ...     return {"headers": {"Authorization": "Bearer token"}}
        >>> accessor = ZarrStoreAccessor(build_url, get_init)
        >>> accessor.zarr_path
        'hubmap_ui/anndata-zarr/secondary_analysis.zarr'
        """
        self._url_builder = url_builder
        self._request_init_provider = request_init_provider
        self.zarr_path = "hubmap_ui/anndata-zarr/secondary_analysis.zarr"
        self.zip_zarr_path = f"{self.zarr_path}.zip"

    def open_store(self, is_zip: bool = False, zarr_path: str | None = None):
        """Open a Zarr store.

                Args:
                    is_zip: Whether the store is zip-compressed
                    zarr_path: Custom path (defaults to self.zarr_path or self.zip_zarr_path)
        reg
                Returns:
                    Opened Zarr store or None on error

                >>> # Requires zarr package - tested in full install mode
                >>> # Example usage:
                >>> # store = accessor.open_store(is_zip=False)
        """
        if not _FULL_DEPS_AVAILABLE:  # pragma: no cover
            raise RuntimeError("Zarr dependencies not available. Install with: pip install portal-visualization[full]")

        request_init = self._request_init_provider() or {}
        path = zarr_path or (self.zip_zarr_path if is_zip else self.zarr_path)

        if is_zip:
            zarr_url = self._url_builder(path, use_token=True)

            try:
                return read_zip_zarr(zarr_url, request_init)
            except Exception as e:  # pragma: no cover
                print(f"Error opening the zip zarr file. {e}")
                return None
        else:
            zarr_url = self._url_builder(path, use_token=False)
            return zarr.open(zarr_url, mode="r", storage_options={"client_kwargs": request_init})


class ImageMetadataRetriever:
    """Handles retrieval and parsing of image metadata JSON files.

    Centralizes the pattern of fetching metadata files and computing physical scales
    that appears in imaging_builders.py and epic_builders.py.
    """

    def __init__(self, resource_loader: ResourceLoader):
        """Initialize with a resource loader.

        Args:
            resource_loader: ResourceLoader instance for fetching JSON

        >>> from unittest.mock import Mock
        >>> loader = Mock(spec=ResourceLoader)
        >>> retriever = ImageMetadataRetriever(loader)
        >>> retriever.resource_loader is loader
        True
        """
        self.resource_loader = resource_loader

    def get_metadata(self, metadata_url: str) -> dict[str, Any] | None:
        """Fetch image metadata from URL.

        Args:
            metadata_url: URL to metadata JSON file

        Returns:
            Metadata dictionary or None on error

        >>> from unittest.mock import Mock
        >>> loader = Mock()
        >>> loader.load_json.return_value = {"PhysicalSizeX": 0.5, "PhysicalSizeXUnit": "μm"}
        >>> retriever = ImageMetadataRetriever(loader)
        >>> metadata = retriever.get_metadata("https://example.com/metadata.json")
        >>> metadata["PhysicalSizeX"]
        0.5
        """
        try:
            return self.resource_loader.load_json(metadata_url)
        except Exception as e:  # pragma: no cover
            print(f"Error fetching metadata from {metadata_url}: {e}")
            return None

    def compute_scale(self, base_metadata: dict[str, Any] | None, overlay_metadata: dict[str, Any] | None) -> float:
        """Compute scale factor between base and overlay images.

        Args:
            base_metadata: Metadata for base/reference image
            overlay_metadata: Metadata for overlay/segmentation image

        Returns:
            Scale factor (overlay_pixel_size / base_pixel_size)

        >>> from unittest.mock import Mock
        >>> retriever = ImageMetadataRetriever(Mock())
        >>> base = {"PhysicalSizeX": 1.0, "PhysicalSizeXUnit": "μm"}
        >>> overlay = {"PhysicalSizeX": 0.5, "PhysicalSizeXUnit": "μm"}
        >>> retriever.compute_scale(base, overlay)
        0.5
        """
        if not base_metadata or not overlay_metadata:
            return 1.0

        from .constants import image_units

        base_size = base_metadata.get("PhysicalSizeX", 1.0)
        base_unit = base_metadata.get("PhysicalSizeXUnit", "μm")

        overlay_size = overlay_metadata.get("PhysicalSizeX", 1.0)
        overlay_unit = overlay_metadata.get("PhysicalSizeXUnit", "μm")

        # Convert to common unit (meters)
        base_in_meters = base_size / image_units.get(base_unit, 1e6)
        overlay_in_meters = overlay_size / image_units.get(overlay_unit, 1e6)

        return overlay_in_meters / base_in_meters if base_in_meters != 0 else 1.0


# Factory functions for creating standard accessors


def create_zarr_accessor(builder_instance):
    """Create a ZarrStoreAccessor from a builder instance.

    Args:
        builder_instance: ViewConfBuilder instance with _build_assets_url and _get_request_init

    Returns:
        Configured ZarrStoreAccessor

    >>> # Requires a builder with proper methods - tested in builder tests
    >>> # Example:
    >>> # accessor = create_zarr_accessor(my_builder)
    >>> # store = accessor.open_store(is_zip=False)
    """
    return ZarrStoreAccessor(
        url_builder=builder_instance._build_assets_url, request_init_provider=builder_instance._get_request_init
    )


def create_http_resource_loader():
    """Create production HTTP resource loader.

    Returns:
        HttpResourceLoader instance

    >>> loader = create_http_resource_loader()
    >>> isinstance(loader, HttpResourceLoader)
    True
    """
    return HttpResourceLoader()
