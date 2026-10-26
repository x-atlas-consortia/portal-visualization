"""
Zarr store access shared by the portal-visualization builders.
"""

import logging

logger = logging.getLogger(__name__)

try:
    from .utils import read_zarr, read_zip_zarr

    _FULL_DEPS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FULL_DEPS_AVAILABLE = False


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
            # use_token=False (like the non-zip path): auth goes through request_init headers, and
            # keeping the ?token= query out of the URL avoids fsspec treating "?" as a glob char.
            zarr_url = self._url_builder(path, use_token=False)

            try:
                return read_zip_zarr(zarr_url, request_init)
            except Exception as e:  # pragma: no cover
                logger.error("Error opening the zip zarr file. %s", e)
                return None
        else:
            zarr_url = self._url_builder(path, use_token=False)
            return read_zarr(zarr_url, request_init)


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
