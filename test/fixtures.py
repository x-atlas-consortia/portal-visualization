"""
Test fixture factories for portal-visualization builders.

This module provides factory functions for creating test entities and mock Zarr stores,
reducing the need for 41+ static JSON fixtures and simplifying test setup.
"""

from typing import Any


def make_entity(
    uuid: str = "test-uuid-1234",
    status: str = "Published",
    soft_assaytype: str | None = None,
    hints: list[str] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Create a test entity with configurable properties.

    Args:
        uuid: Entity UUID
        status: Publication status ("Published", "QA", etc.)
        soft_assaytype: Assay type string
        hints: List of vitessce-hints
        **kwargs: Additional entity fields

    Returns:
        Entity dictionary

    >>> entity = make_entity(uuid="abc123", hints=["is_image", "rna"])
    >>> entity["uuid"]
    'abc123'
    >>> entity["vitessce-hints"]
    ['is_image', 'rna']
    """
    entity = {
        "uuid": uuid,
        "status": status,
        "vitessce-hints": hints or [],
    }

    if soft_assaytype:
        entity["soft_assaytype"] = soft_assaytype

    entity.update(kwargs)
    return entity


def make_rna_seq_entity(
    uuid: str = "rna-seq-test",
    is_annotated: bool = False,
    is_published: bool = True,
    is_zip: bool = False,
    has_marker_gene: bool = False,
    **kwargs,
) -> dict[str, Any]:
    """Create an RNA-seq entity with common variations.

    Args:
        uuid: Entity UUID
        is_annotated: Whether entity has annotations
        is_published: Whether entity is published
        is_zip: Whether zarr is zip-compressed
        has_marker_gene: Whether to include marker gene metadata
        **kwargs: Additional fields

    Returns:
        Configured entity

    >>> entity = make_rna_seq_entity(is_annotated=True, is_published=True)
    >>> "rna" in entity["vitessce-hints"]
    True
    >>> entity["status"]
    'Published'
    """
    hints = ["rna"]
    if is_annotated:
        hints.append("is_annotated")

    status = "Published" if is_published else "QA"

    entity = make_entity(uuid=uuid, status=status, hints=hints, **kwargs)

    if has_marker_gene:
        entity["marker_gene"] = {"name": "CD4", "ensembl_id": "ENSG00000010610"}

    return entity


def make_spatial_entity(
    uuid: str = "spatial-test", is_multiome: bool = False, assay_type: str | None = None, **kwargs
) -> dict[str, Any]:
    """Create a spatial omics entity.

    Args:
        uuid: Entity UUID
        is_multiome: Whether this is multiomics data
        assay_type: Specific assay type (VISIUM, XENIUM, etc.)
        **kwargs: Additional fields

    Returns:
        Configured spatial entity

    >>> entity = make_spatial_entity(is_multiome=True)
    >>> "spatial" in entity["vitessce-hints"]
    True
    >>> "is_image" in entity["vitessce-hints"]
    True
    """
    hints = ["is_image", "rna", "spatial"]

    if is_multiome:
        hints.append("is_multiome")

    return make_entity(uuid=uuid, soft_assaytype=assay_type, hints=hints, **kwargs)


def make_imaging_entity(
    uuid: str = "imaging-test",
    assay_type: str | None = None,
    is_support: bool = False,
    is_seg_mask: bool = False,
    **kwargs,
) -> dict[str, Any]:
    """Create an imaging entity.

    Args:
        uuid: Entity UUID
        assay_type: Imaging assay type (CODEX, IMS, etc.)
        is_support: Whether this is a support dataset
        is_seg_mask: Whether this is a segmentation mask
        **kwargs: Additional fields

    Returns:
        Configured imaging entity

    >>> entity = make_imaging_entity(assay_type="CODEX [Cytokit + SPRM]")
    >>> "is_image" in entity["vitessce-hints"]
    True
    >>> entity["soft_assaytype"]
    'CODEX [Cytokit + SPRM]'
    """
    hints = ["is_image"]

    if is_support:
        hints.append("is_support")
    if is_seg_mask:
        hints.append("segmentation_mask")

    return make_entity(uuid=uuid, soft_assaytype=assay_type, hints=hints, **kwargs)


def make_sprm_entity(
    uuid: str = "sprm-test", is_anndata: bool = False, is_json: bool = False, assay_type: str | None = None, **kwargs
) -> dict[str, Any]:
    """Create a SPRM entity.

    Args:
        uuid: Entity UUID
        is_anndata: Whether data is in AnnData format
        is_json: Whether data is JSON-based
        assay_type: Assay type
        **kwargs: Additional fields

    Returns:
        Configured SPRM entity

    >>> entity = make_sprm_entity(is_anndata=True)
    >>> "is_sprm" in entity["vitessce-hints"]
    True
    >>> "is_anndata" in entity["vitessce-hints"]
    True
    """
    hints = ["is_image", "is_sprm"]

    if is_anndata:
        hints.append("is_anndata")
    if is_json:
        hints.append("is_json")

    return make_entity(uuid=uuid, soft_assaytype=assay_type, hints=hints, **kwargs)


# Mock Zarr store factory functions


def create_mock_zarr_group():
    """Create a mock Zarr group for testing.

    This is a lightweight mock that avoids importing zarr in thin install mode.

    Returns:
        Mock Zarr group object

    >>> group = create_mock_zarr_group()
    >>> group is not None
    True
    """
    try:
        import zarr

        return zarr.open_group()
    except ImportError:  # pragma: no cover
        # Return a simple dict-based mock for thin install tests
        return {}


def populate_anndata_zarr(
    zarr_group,
    obs_count: int = 100,
    var_count: int = 50,
    is_annotated: bool = False,
    cluster_columns: list[str] | None = None,
    embedding_keys: list[str] | None = None,
) -> None:
    """Populate a Zarr group with AnnData structure.

    Args:
        zarr_group: Zarr group to populate
        obs_count: Number of observations
        var_count: Number of variables
        is_annotated: Whether to add annotation columns
        cluster_columns: List of clustering column names
        embedding_keys: List of embedding names (e.g., ["X_umap", "X_pca"])

    >>> # Requires zarr package
    >>> # group = create_mock_zarr_group()
    >>> # populate_anndata_zarr(group, obs_count=50, is_annotated=True)
    """
    try:
        import numpy as np
        import zarr
    except ImportError:  # pragma: no cover
        return  # Skip for thin install

    # Create observation indices
    obs_index = [f"cell_{i}" for i in range(obs_count)]
    var_index = [f"gene_{i}" for i in range(var_count)]

    # Create obs group
    obs = zarr_group.create_group("obs")
    obs["_index"] = zarr.array(obs_index)

    # Add cluster columns
    if cluster_columns:
        for col in cluster_columns:
            obs[col] = zarr.array(np.random.randint(0, 5, size=obs_count))

    # Add annotations
    if is_annotated:
        obs["predicted_label"] = zarr.array([f"celltype_{i % 3}" for i in range(obs_count)])
        obs["predicted.ASCT.celltype"] = zarr.array([f"asct_type_{i % 3}" for i in range(obs_count)])
        obs["predicted_CLID"] = zarr.array([f"CL:{1000000 + i % 5}" for i in range(obs_count)])
        obs["CL_Label"] = zarr.array([f"cl_label_{i % 3}" for i in range(obs_count)])
        obs["final_level_labels"] = zarr.array([f"final_label_{i % 3}" for i in range(obs_count)])
        obs["full_hierarchical_labels"] = zarr.array([f"hierarchy_{i % 3}" for i in range(obs_count)])
        obs["annotation_method"] = zarr.array(["azimuth"] * obs_count)
        obs["predicted.celltype.l1"] = zarr.array([f"az_l1_{i % 3}" for i in range(obs_count)])
        obs["predicted.celltype.l2"] = zarr.array([f"az_l2_{i % 3}" for i in range(obs_count)])

        # Add annotation metadata
        uns = zarr_group.create_group("uns", overwrite=True)
        ann_meta = uns.create_group("annotation_metadata", overwrite=True)
        ann_meta["is_annotated"] = zarr.array(True)

    # Create var group
    var = zarr_group.create_group("var")
    var["_index"] = zarr.array(var_index)

    # Add embeddings
    if embedding_keys:
        obsm = zarr_group.create_group("obsm")
        for key in embedding_keys:
            obsm[key] = zarr.array(np.random.rand(obs_count, 2))

    # Add X data
    zarr_group["X"] = zarr.array(np.random.rand(obs_count, var_count))


def populate_spatial_zarr(zarr_group, obs_count: int = 100, has_spatial_coords: bool = True, **kwargs) -> None:
    """Populate Zarr with spatial transcriptomics data.

    Args:
        zarr_group: Zarr group to populate
        obs_count: Number of observations
        has_spatial_coords: Whether to include spatial coordinates
        **kwargs: Additional args for populate_anndata_zarr

    >>> # Requires zarr package
    >>> # group = create_mock_zarr_group()
    >>> # populate_spatial_zarr(group, obs_count=50)
    """
    try:
        import numpy as np
        import zarr
    except ImportError:  # pragma: no cover
        return

    # Add standard AnnData structure
    populate_anndata_zarr(zarr_group, obs_count=obs_count, **kwargs)

    # Add spatial coordinates
    if has_spatial_coords:
        obsm = zarr_group.require_group("obsm")
        obsm["spatial"] = zarr.array(np.random.rand(obs_count, 2) * 1000)  # Pixel coordinates


def populate_multiome_zarr(zarr_group, obs_count: int = 100, modalities: list[str] | None = None) -> None:
    """Populate Zarr with multiomics (h5mu) structure.

    Args:
        zarr_group: Zarr group to populate
        obs_count: Number of observations
        modalities: List of modality names (e.g., ["rna", "atac"])

    >>> # Requires zarr package
    >>> # group = create_mock_zarr_group()
    >>> # populate_multiome_zarr(group, modalities=["rna", "atac"])
    """
    try:
        import numpy as np
        import zarr
    except ImportError:  # pragma: no cover
        return

    modalities = modalities or ["rna", "atac"]

    # Create mod group
    mod = zarr_group.create_group("mod")

    for modality in modalities:
        mod_group = mod.create_group(modality)

        # Add basic structure for each modality
        obs = mod_group.create_group("obs")
        obs["_index"] = zarr.array([f"cell_{i}" for i in range(obs_count)])

        var = mod_group.create_group("var")
        var["_index"] = zarr.array([f"feature_{i}" for i in range(50)])

        # Add cluster columns specific to modality
        obs[f"leiden_{modality}"] = zarr.array(np.random.randint(0, 5, size=obs_count))


# Legacy compatibility functions


def make_legacy_entity_dict(entity_type: str, **kwargs) -> dict[str, Any]:
    """Create entity dict matching legacy fixture format.

    Provides backward compatibility with existing test patterns.

    Args:
        entity_type: Type of entity ("rna_seq", "spatial", "imaging", "sprm")
        **kwargs: Entity configuration

    Returns:
        Entity dictionary

    >>> entity = make_legacy_entity_dict("rna_seq", is_annotated=True)
    >>> "rna" in entity["vitessce-hints"]
    True
    """
    factories = {
        "rna_seq": make_rna_seq_entity,
        "spatial": make_spatial_entity,
        "imaging": make_imaging_entity,
        "sprm": make_sprm_entity,
    }

    if entity_type not in factories:
        raise ValueError(f"Unknown entity type: {entity_type}")

    return factories[entity_type](**kwargs)
