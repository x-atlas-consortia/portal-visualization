import logging
import warnings

from .builder_registry import get_registry, populate_registry
from .builders.base_builders import NullViewConfBuilder

logger = logging.getLogger(__name__)

# Initialize the registry on module import
_registry_initialized = False


def _ensure_registry_initialized():
    """Ensure the builder registry is populated.

    This is called lazily to avoid circular import issues.
    """
    global _registry_initialized
    if not _registry_initialized:
        populate_registry()
        _registry_initialized = True


def _lazy_import_builder(builder_name):
    """Lazy import builder classes to avoid loading heavy dependencies.

    This allows the has_visualization function to work without requiring
    vitessce, zarr, and other heavy dependencies to be installed.

    :param str builder_name: The name of the builder class to import
    :return: The builder class
    :rtype: type

    >>> builder = _lazy_import_builder('NullViewConfBuilder')
    >>> builder.__name__
    'NullViewConfBuilder'
    """
    if builder_name == "NullViewConfBuilder":
        return NullViewConfBuilder

    # Import all builders at once when needed
    from .builders.anndata_builders import (
        MultiomicAnndataZarrViewConfBuilder,
        RNASeqAnnDataZarrViewConfBuilder,
        SpatialMultiomicAnnDataZarrViewConfBuilder,
        SpatialRNASeqAnnDataZarrViewConfBuilder,
        XeniumMultiomicAnnDataZarrViewConfBuilder,
    )
    from .builders.epic_builders import (
        SegmentationMaskBuilder,
    )
    from .builders.imaging_builders import (
        EpicSegImagePyramidViewConfBuilder,
        GeoMxImagePyramidViewConfBuilder,
        ImagePyramidViewConfBuilder,
        IMSViewConfBuilder,
        Kaggle1SegImagePyramidViewConfBuilder,
        KaggleSegImagePyramidViewConfBuilder,
        NanoDESIViewConfBuilder,
        SeqFISHViewConfBuilder,
    )
    from .builders.object_by_analyte_builders import ObjectByAnalyteConfBuilder
    from .builders.scatterplot_builders import (
        ATACSeqViewConfBuilder,
        RNASeqViewConfBuilder,
    )
    from .builders.sprm_builders import (
        MultiImageSPRMAnndataViewConfBuilder,
        SPRMAnnDataViewConfBuilder,
        SPRMJSONViewConfBuilder,
        StitchedCytokitSPRMViewConfBuilder,
        TiledSPRMViewConfBuilder,
    )

    # Map builder names to classes
    builders = {
        "MultiomicAnndataZarrViewConfBuilder": MultiomicAnndataZarrViewConfBuilder,
        "RNASeqAnnDataZarrViewConfBuilder": RNASeqAnnDataZarrViewConfBuilder,
        "SpatialMultiomicAnnDataZarrViewConfBuilder": SpatialMultiomicAnnDataZarrViewConfBuilder,
        "SpatialRNASeqAnnDataZarrViewConfBuilder": SpatialRNASeqAnnDataZarrViewConfBuilder,
        "XeniumMultiomicAnnDataZarrViewConfBuilder": XeniumMultiomicAnnDataZarrViewConfBuilder,
        "EpicSegImagePyramidViewConfBuilder": EpicSegImagePyramidViewConfBuilder,
        "GeoMxImagePyramidViewConfBuilder": GeoMxImagePyramidViewConfBuilder,
        "ImagePyramidViewConfBuilder": ImagePyramidViewConfBuilder,
        "IMSViewConfBuilder": IMSViewConfBuilder,
        "Kaggle1SegImagePyramidViewConfBuilder": Kaggle1SegImagePyramidViewConfBuilder,
        "KaggleSegImagePyramidViewConfBuilder": KaggleSegImagePyramidViewConfBuilder,
        "NanoDESIViewConfBuilder": NanoDESIViewConfBuilder,
        "SeqFISHViewConfBuilder": SeqFISHViewConfBuilder,
        "ObjectByAnalyteConfBuilder": ObjectByAnalyteConfBuilder,
        "ATACSeqViewConfBuilder": ATACSeqViewConfBuilder,
        "RNASeqViewConfBuilder": RNASeqViewConfBuilder,
        "MultiImageSPRMAnndataViewConfBuilder": MultiImageSPRMAnndataViewConfBuilder,
        "SPRMAnnDataViewConfBuilder": SPRMAnnDataViewConfBuilder,
        "SPRMJSONViewConfBuilder": SPRMJSONViewConfBuilder,
        "StitchedCytokitSPRMViewConfBuilder": StitchedCytokitSPRMViewConfBuilder,
        "TiledSPRMViewConfBuilder": TiledSPRMViewConfBuilder,
        "SegmentationMaskBuilder": SegmentationMaskBuilder,
    }

    if builder_name in builders:
        return builders[builder_name]
    else:  # pragma: no cover
        raise ValueError(f"Unknown builder: {builder_name}")


# This function processes the hints and returns a tuple of booleans.
# Part of the public API (exported via __init__.py).
def process_hints(hints):
    """Process vitessce-hints into a tuple of boolean flags.

    :param list hints: List of vitessce-hints from entity
    :return: Tuple of boolean flags for each hint type
    :rtype: tuple

    >>> process_hints(["is_image", "rna"])[0]
    True
    >>> process_hints(["rna"])[0]
    False
    >>> process_hints(None)[0]
    False
    """
    if not hints:
        hints = []
    hints = set(hints)
    is_image = "is_image" in hints
    is_rna = "rna" in hints
    is_atac = "atac" in hints
    is_sprm = "sprm" in hints
    is_codex = "codex" in hints
    is_anndata = "anndata" in hints
    is_json = "json_based" in hints
    is_spatial = "spatial" in hints
    is_support = "is_support" in hints
    is_seg_mask = "segmentation_mask" in hints
    is_geomx = "geomx" in hints
    is_xenium = "xenium" in hints
    is_epic = "epic" in hints

    return (
        is_image,
        is_rna,
        is_atac,
        is_sprm,
        is_codex,
        is_anndata,
        is_json,
        is_spatial,
        is_support,
        is_seg_mask,
        is_geomx,
        is_xenium,
        is_epic,
    )


def get_view_config_builder(entity, get_entity, parent=None):
    """Get the appropriate builder class for an entity.

    Returns a builder class (not an instance) that can be used to generate
    Vitessce configurations for the given entity.

    :param dict entity: Entity response from search index
    :param callable get_entity: Function to retrieve entity by UUID
    :param str parent: Parent entity UUID if this is a support dataset
    :return: Builder class
    :rtype: type
    """
    builder_name = _get_builder_name_from_registry(entity, get_entity, parent)
    return _lazy_import_builder(builder_name)


def _get_builder_name_from_registry(entity, get_entity, parent=None):
    """Get the name of the appropriate builder for an entity using the registry.

    This is the core decision logic that doesn't require importing heavy dependencies.
    Returns the builder class name as a string.

    :param dict entity: Entity response from search index
    :param callable get_entity: Function to retrieve entity by UUID
    :param str parent: Parent entity UUID if this is a support dataset
    :return: Builder class name
    :rtype: str

    >>> _get_builder_name_from_registry({"uuid": "test", "vitessce-hints": ["is_image", "rna"]}, None)
    'SpatialMultiomicAnnDataZarrViewConfBuilder'
    >>> _get_builder_name_from_registry({"uuid": "test", "vitessce-hints": []}, None)
    'NullViewConfBuilder'
    """
    _ensure_registry_initialized()
    registry = get_registry()

    hints = entity.get("vitessce-hints", [])
    assay_type = entity.get("soft_assaytype")
    has_parent = parent is not None
    # Detect EPIC datasets by checking for explicit EPIC hints
    has_epic = "epic" in hints

    # Get parent assay type if parent exists
    parent_assay_type = None
    if parent is not None and get_entity is not None:
        try:
            parent_entity = get_entity(parent)
            parent_assay_type = parent_entity.get("soft_assaytype")
        except (FileNotFoundError, KeyError):
            # Parent entity not found or doesn't have soft_assaytype
            pass

    builder_name = registry.find_builder(
        hints=hints,
        assay_type=assay_type,
        has_parent=has_parent,
        has_epic=has_epic,
        parent_assay_type=parent_assay_type,
    )

    if builder_name is None:  # pragma: no cover
        # No match found - generate detailed diagnostic message
        error_msg = registry.format_no_match_message(
            hints=hints,
            assay_type=assay_type,
            has_parent=has_parent,
            has_epic=has_epic,
            parent_assay_type=parent_assay_type,
        )
        warnings.warn(f"Builder selection failed:\n{error_msg}", UserWarning, stacklevel=2)
        # Fallback to NullViewConfBuilder
        return "NullViewConfBuilder"

    return builder_name


def has_visualization(entity, get_entity, parent=None):
    """Check if an entity has a visualization without loading heavy dependencies.

    This function works with the thin install (no [full] extras required).

    :param dict entity: Entity response from search index
    :param callable get_entity: Function to retrieve entity by UUID
    :param str parent: Parent entity UUID if this is a support dataset
    :return: True if the entity has a visualization, False otherwise
    :rtype: bool

    >>> has_visualization({"uuid": "test", "vitessce-hints": ["rna"]}, lambda x: {})
    True
    >>> has_visualization({"uuid": "test", "vitessce-hints": []}, lambda x: {})
    False
    """
    if entity.get("uuid") is None:
        raise ValueError("Provided entity does not have a uuid")
    builder_name = _get_builder_name_from_registry(entity, get_entity, parent)
    return builder_name != "NullViewConfBuilder"
