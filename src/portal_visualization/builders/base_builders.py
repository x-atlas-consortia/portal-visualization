import urllib
from abc import ABC, abstractmethod
from collections import namedtuple
from functools import cached_property

ConfCells = namedtuple("ConfCells", ["conf", "cells"])


class NullViewConfBuilder:
    def __init__(self, entity, groups_token, assets_endpoint, **kwargs):
        # Just so it has the same signature as the other builders
        pass

    def get_conf_cells(self, **kwargs):
        return ConfCells(None, None)


class ViewConfBuilder(ABC):
    def __init__(self, entity, groups_token, assets_endpoint, **kwargs):
        """Object for building the vitessce configuration.
        :param dict entity: Entity response from search index (from the entity API)
        :param str  groups_token: Groups token for use in authenticating API
        :param str  assets_endpoint: The base URL for the assets API
        :param dict kwargs: Additional keyword arguments
        :param str  kwargs.schema_version: The vitessce schema version to use, default "1.0.15"
        :param bool kwargs.minimal: Whether or not to build a minimal configuration, default False
        """

        self._uuid = entity["uuid"]
        self._groups_token = groups_token
        self._assets_endpoint = assets_endpoint
        self._entity = entity
        self._files = []
        self._schema_version = kwargs.get("schema_version", "1.0.15")
        self._minimal = kwargs.get("minimal", False)

        # Common attributes used by many builders
        self._is_zarr_zip = False

    @cached_property
    def _zarr_accessor(self):
        """Get ZarrStoreAccessor instance for this builder.
        Override this if you need custom zarr access logic.
        """
        from ..data_access import create_zarr_accessor

        return create_zarr_accessor(self)

    @cached_property
    def zarr_store(self):
        """Open the Zarr store using ZarrStoreAccessor.
        Override this if you need custom zarr store logic.
        Default implementation uses ZARR_PATH or ZIP_ZARR_PATH based on _is_zarr_zip flag.
        """
        from ..constants import ZARR_PATH, ZIP_ZARR_PATH

        zarr_path = ZIP_ZARR_PATH if self._is_zarr_zip else ZARR_PATH
        return self._zarr_accessor.open_store(is_zip=self._is_zarr_zip, zarr_path=zarr_path)

    def _detect_zarr_format(self):
        """Detect if zarr files are in .zip format and set _is_zarr_zip flag.
        Returns True if zip format detected, False otherwise.

        >>> builder = _DocTestBuilder(
        ...   entity={"uuid": "uuid", "files": [{"rel_path": "data.zarr.zip"}]},
        ...   groups_token='token',
        ...   assets_endpoint='https://example.com')
        >>> builder._detect_zarr_format()
        True
        >>> builder._is_zarr_zip
        True
        """
        file_paths = self._get_file_paths()
        if any(".zarr.zip" in path for path in file_paths):
            self._is_zarr_zip = True
            return True
        return False

    def _create_vitessce_config(self, name=None, dataset_name=None):
        """Create a VitessceConfig with standardized settings.

        :param str name: Name for the config. Defaults to "HuBMAP Data Portal"
        :param str dataset_name: Name for the dataset. Defaults to self._uuid
        :return: Tuple of (VitessceConfig, VitessceConfigDataset)

        >>> builder = _DocTestBuilder(
        ...   entity={"uuid": "test-uuid"},
        ...   groups_token='token',
        ...   assets_endpoint='https://example.com')
        >>> vc, dataset = builder._create_vitessce_config()
        >>> vc.to_dict()['name']
        'HuBMAP Data Portal'
        >>> type(dataset).__name__
        'VitessceConfigDataset'
        """
        from vitessce import VitessceConfig

        if name is None:
            name = "HuBMAP Data Portal"
        if dataset_name is None:
            dataset_name = self._uuid

        vc = VitessceConfig(name=name, schema_version=self._schema_version)
        dataset = vc.add_dataset(name=dataset_name)
        return vc, dataset

    def _require_file(self, pattern, description=None):
        """Check that a file matching the pattern exists in the entity.

        :param str pattern: File path pattern to search for (supports regex via re.search)
        :param str description: Optional description for error message
        :raises FileNotFoundError: If no file matching pattern is found

        >>> builder = _DocTestBuilder(
        ...   entity={"uuid": "test-uuid", "files": [{"rel_path": "data.zarr/.zgroup"}]},
        ...   groups_token='token',
        ...   assets_endpoint='https://example.com')
        >>> builder._require_file(".zgroup")  # Returns None on success
        >>> builder._require_file("missing.txt")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        FileNotFoundError: ...
        """
        import re

        file_paths = self._get_file_paths()
        # Try exact match first
        if pattern in file_paths:
            return

        # Try regex search
        for path in file_paths:
            if re.search(pattern, path):
                return

        # Not found - raise error
        if description is None:
            description = f"file matching '{pattern}'"
        message = f"Dataset {self._uuid} is missing {description}"
        raise FileNotFoundError(message)

    def _require_files(self, patterns, description=None):
        """Check that all files in the list exist in the entity.

        :param list patterns: List of file path patterns to search for
        :param str description: Optional description for error message
        :raises FileNotFoundError: If any file is missing

        >>> builder = _DocTestBuilder(
        ...   entity={"uuid": "test", "files": [{"rel_path": "a.json"}, {"rel_path": "b.json"}]},
        ...   groups_token='token',
        ...   assets_endpoint='https://example.com')
        >>> builder._require_files(["a.json", "b.json"])
        >>> builder._require_files(["a.json", "missing.json"])  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        FileNotFoundError: ...
        """
        file_paths = self._get_file_paths()
        file_paths_set = set(file_paths)

        # Check if all expected files exist
        missing = [p for p in patterns if p not in file_paths_set]
        if missing:
            if description is None:
                description = f"required files: {missing}"
            message = f"Dataset {self._uuid} is missing {description}"
            raise FileNotFoundError(message)

    @abstractmethod
    def get_conf_cells(self, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def _replace_url_in_file(self, file):
        """Replace url in incoming file object
        :param dict file: File dict which will have its rel_path replaced by url
        :rtype: dict The file with rel_path replaced by url

        >>> from pprint import pprint
        >>> builder = _DocTestBuilder(
        ...   entity={ "uuid": "uuid" },
        ...   groups_token='groups_token',
        ...   assets_endpoint='https://example.com')
        >>> file = {
        ...     'file_type': 'cells.json',
        ...     'rel_path': 'cells.json',
        ...     'coordination_values': { 'obsType': 'cell' } }
        >>> pprint(builder._replace_url_in_file(file))
        {'coordination_values': {'obsType': 'cell'},
         'file_type': 'cells.json',
         'url': 'https://example.com/uuid/cells.json?token=groups_token'}
        """

        return {
            "coordination_values": file["coordination_values"],
            "file_type": file["file_type"],
            "url": self._build_assets_url(file["rel_path"]),
        }

    def _build_assets_url(self, rel_path, use_token=True):
        """Create a url for an asset.
        :param str rel_path: The path off of which the url should be built
        :param bool use_token: Whether or not to append a groups token to the URL, default True
        :rtype: dict The file with rel_path replaced by url

        >>> from pprint import pprint
        >>> builder = _DocTestBuilder(
        ...   entity={ "uuid": "uuid" },
        ...   groups_token='groups_token',
        ...   assets_endpoint='https://example.com')
        >>> builder._build_assets_url("rel_path/to/clusters.ome.tiff")
        'https://example.com/uuid/rel_path/to/clusters.ome.tiff?token=groups_token'

        """
        uuid = self._uuid
        if hasattr(self, "_epic_uuid"):  # pragma: no cover
            uuid = self._epic_uuid
        base_url = urllib.parse.urljoin(self._assets_endpoint, f"{uuid}/{rel_path}")
        token_param = urllib.parse.urlencode({"token": self._groups_token})
        return f"{base_url}?{token_param}" if use_token else base_url

    def _get_request_init(self):
        """Get request headers for requestInit parameter in Vitessce conf.
        This is needed for non-public zarr stores because the client forms URLs for zarr chunks,
        not the above _build_assets_url function.

        >>> builder = _DocTestBuilder(
        ...   entity={"uuid": "uuid", "status": "QA"},
        ...   groups_token='groups_token',
        ...   assets_endpoint='https://example.com')
        >>> builder._get_request_init()
        {'headers': {'Authorization': 'Bearer groups_token'}}

        >>> builder = _DocTestBuilder(
        ...   entity={"uuid": "uuid", "status": "Published"},
        ...   groups_token='groups_token',
        ...   assets_endpoint='https://example.com')
        >>> repr(builder._get_request_init())
        'None'
        """
        if self._entity["status"] == "Published":
            # Extra headers outside of a select few cause extra CORS-preflight requests which
            # can slow down the webpage.  If the dataset is published, we don't need to use
            # header to authenticate access to the assets API.
            # See: https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS#simple_requests
            return None
        return {"headers": {"Authorization": f"Bearer {self._groups_token}"}}

    def _get_file_paths(self):
        """Get all rel_path keys from the entity dict.

        >>> files = [{ "rel_path": "path/to/file" }, { "rel_path": "path/to/other_file" }]
        >>> builder = _DocTestBuilder(
        ...   entity={"uuid": "uuid", "files": files},
        ...   groups_token='groups_token',
        ...   assets_endpoint='https://example.com')
        >>> builder._get_file_paths()
        ['path/to/file', 'path/to/other_file']
        """
        return [file["rel_path"] for file in self._entity["files"]]


class _DocTestBuilder(ViewConfBuilder):  # pragma: no cover
    # The doctests on the methods in this file need a concrete class to instantiate:
    # We need a concrete definition for this method, even if it's never used.
    def get_conf_cells(self, **kwargs):
        pass
