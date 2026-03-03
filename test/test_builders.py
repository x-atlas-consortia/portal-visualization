#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from os import environ
from pathlib import Path

import pytest

from src.portal_visualization.builder_factory import (
    get_view_config_builder,
    has_visualization,
)

from .fixtures import (
    create_mock_zarr_group,
    make_rna_seq_entity,
    populate_anndata_zarr,
    populate_multiome_zarr,
)

# Tests that instantiate builders and generate configs require [full] dependencies
pytest_requires_full = pytest.mark.requires_full

try:
    import yaml
    import zarr

    from src.portal_visualization.builders.imaging_builders import (
        Kaggle1SegImagePyramidViewConfBuilder,
        KaggleSegImagePyramidViewConfBuilder,
    )
    from src.portal_visualization.paths import IMAGE_PYRAMID_DIR
    from src.portal_visualization.utils import get_found_images, read_zip_zarr

    FULL_DEPS_AVAILABLE = True
except ImportError:
    FULL_DEPS_AVAILABLE = False

groups_token = environ.get("GROUPS_TOKEN", "groups_token")
assets_url = environ.get("ASSETS_URL", "https://example.com")


def str_presenter(dumper, data):
    # From https://stackoverflow.com/a/33300001
    if len(data.splitlines()) > 1:  # check for multiline string
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(str, str_presenter)


@dataclass
class MockResponse:
    content: str


good_entity_paths = list((Path(__file__).parent / "good-fixtures").glob("*/*-entity.json"))

# Exclude fixtures that have been replaced by programmatic tests
excluded_fixtures = {
    # RNASeqAnnDataZarrViewConfBuilder
    "fake-is-not-annotated-published-entity.json",
    "fake-is-not-annotated-qa-entity.json",
    "fake-is-not-annotated-minimal-published-entity.json",
    "fake-asct-is-annotated-published-entity.json",
    "fake-asct-is-annotated-qa-entity.json",
    "fake-asct-is-annotated-zip-published-entity.json",
    "fake-predicted-label-is-annotated-published-entity.json",
    "fake-predicted-label-is-annotated-qa-entity.json",
    "fake-is-annotated-pan-az-published-entity.json",
    # MultiomicAnndataZarrViewConfBuilder
    "fake-multiome-entity.json",
    "fake-multiome-is-annotated-entity.json",
    "fake-multiome-is-annotated-pan-az-entity.json",
    # SpatialMultiomicAnnDataZarrViewConfBuilder
    "fake-visium-entity.json",
    # XeniumMultiomicAnnDataZarrViewConfBuilder
    "fake-xenium-entity.json",
    # SpatialRNASeqAnnDataZarrViewConfBuilder
    "ea4cfecb8495b36694d9a951510dc3c6-minimal-entity.json",
    "ea4cfecb8495b36694d9a951510dc3c6-marker=gene123-entity.json",
    # ObjectByAnalyteConfBuilder
    "fake-object-by-analyte-entity.json",
    "many-embeddings-object-by-analyte-entity.json",
    "no-feature-labels-object-by-analyte-entity.json",
    "single-embedding-object-by-analyte-entity.json",
    "spatial-object-by-analyte-entity.json",
    "three-embeddings-object-by-analyte-entity.json",
    "uniprot-object-by-analyte-entity.json",
    "zero-dimensions-object-by-analyte-entity.json",
    # TiledSPRMViewConfBuilder
    "no-cells-entity.json",
    "with-cells-entity.json",
    # StitchedCytokitSPRMViewConfBuilder
    "04e7385339167e541ad42a2636e18398-entity.json",
    # ImagePyramidViewConfBuilder, KaggleSegImagePyramidViewConfBuilder,
    # ImsImagePyramidViewConfBuilder, NanoDESIImagePyramidViewConfBuilder,
    # MultiImageSPRMViewConfBuilder, ATACSeqViewConfBuilder, RNASeqViewConfBuilder,
    # SeqFISHViewConfBuilder - all use "fake-entity.json"
    "fake-entity.json",
    # ImagePyramidViewConfBuilder (unique UUID)
    "3bc3ad12-entity.json",
    # GeoMxImagePyramidViewConfBuilder
    "bc7239d27b79e087c788600261f073e5-entity.json",
    "bc7239d27b79e087c788600261f073e5-zarr-zip-entity.json",
    # SegmentationMaskBuilder
    "fake-zarr-zip-entity.json",
    # NullViewConfBuilder (now programmatic)
    "empty-entity.json",
    "fake-no-support-entity.json",
}
good_entity_paths = [p for p in good_entity_paths if p.name not in excluded_fixtures]

# Note: All good entity fixtures have been migrated to programmatic tests
# This list is now empty but kept for backward compatibility

bad_entity_paths = list((Path(__file__).parent / "bad-fixtures").glob("*-entity.json"))

# Note: Bad entity fixtures serve as important documentation for error cases
# We keep them as regression tests alongside programmatic error tests

assaytypes_path = Path(__file__).parent / "assaytype-fixtures"
assert assaytypes_path.is_dir()

default_assaytype = {
    "soft_assaytype": "Null",
    "vitessce-hints": [],
}


def get_entity(input):
    uuid = input.get("uuid") if not isinstance(input, str) else input
    if uuid is None:  # pragma: no cover
        return default_assaytype
    assay = json.loads(assaytypes_path.joinpath(f"{uuid}.json").read_text())
    return assay


# Mock support entity for Kaggle-1 tests: provides base images from parent's support entity
_kaggle1_support_entity = {
    "uuid": "kaggle1-support-uuid",
    "files": [
        {"rel_path": "ometiff-pyramids/lab_processed/images/B001_SB-reg005.ome.tif"},
        {"rel_path": "output_offsets/lab_processed/images/B001_SB-reg005.offsets.json"},
        {"rel_path": "image_metadata/lab_processed/images/B001_SB-reg005.metadata.json"},
    ],
}


def _mock_find_support_entity(uuid):
    return _kaggle1_support_entity


# Construct test cases for has_visualization.
# Initial values are edge cases (null view conf builder)
has_visualization_test_cases = [
    (False, {"uuid": "2c2179ea741d3bbb47772172a316a2bf"}),
    (False, {"uuid": "f9ae931b8b49252f150d7f8bf1d2d13f-bad"}),
]
excluded_uuids = {entity["uuid"] for _, entity in has_visualization_test_cases}

# All other values are good entities which should have a visualization
for path in good_entity_paths:
    entity = json.loads(path.read_text())
    uuid = entity.get("uuid")
    if uuid in excluded_uuids or path.parent.name == "NullViewConfBuilder":
        continue
    has_visualization_test_cases.append((True, entity))

# NOTE: programmatic_test_cases (defined later) are added to
# has_visualization_test_cases after they're generated; see code below programmatic_test_cases.


@pytest.mark.parametrize(
    "has_vis_entity",
    has_visualization_test_cases,
    ids=lambda e: (f"has_visualization={e[0]}_uuid={e[1].get('uuid', 'no-uuid')}" if isinstance(e, tuple) else str(e)),
)
def test_has_visualization(has_vis_entity):
    has_vis, entity = has_vis_entity
    parent = entity.get("parent") or None  # Only used for image pyramids
    assert has_vis == has_visualization(entity, get_entity, parent)


def test_has_visualization_no_uuid():
    with pytest.raises(ValueError, match="does not have a uuid"):
        has_visualization({}, get_entity)


def is_annotated_entity(entity_path):
    # Check for various annotated patterns
    # - "is-annotated" from old fixture files
    # - "asct-annotated", "predicted-label-annotated", "pan-az-annotated" from programmatic tests
    name = entity_path.name
    return "is-annotated" in name or ("annotated" in name and "not-annotated" not in name)


def is_multiome_entity(entity_path):
    return "multiome" in entity_path.name


def is_pan_azimuth_entity(entity_path):
    return "pan-az" in entity_path.name


def is_visium_entity(entity_path):
    return "visium" in entity_path.name


def is_xenium_entity(entity_path):
    return "xenium" in entity_path.name


def is_zip_entity(entity_path):
    return "zip" in entity_path.name


def is_marker_entity(entity_path):
    return "marker" in entity_path.name


def is_asct_entity(entity_path):
    return "asct" in entity_path.name


def is_azimuth_labeled_entity(entity_path):
    return "predicted-label" in entity_path.name


def is_object_by_analyte_entity(entity_path):
    return "object-by-analyte" in entity_path.name


def mock_zarr_store(entity_path, mocker, obs_count):
    """Create a mock Zarr store for testing.

    Uses fixture factories for structure, but maintains backward compatibility
    with existing test fixtures by matching the old zarr structure exactly.
    """
    # Determine entity configuration from filename
    is_annotated = is_annotated_entity(entity_path)
    is_multiome = is_multiome_entity(entity_path)
    is_pan_azimuth = is_pan_azimuth_entity(entity_path)
    is_visium = is_visium_entity(entity_path)
    is_marker = is_marker_entity(entity_path)
    is_asct = is_asct_entity(entity_path)
    is_azimuth_labeled = is_azimuth_labeled_entity(entity_path)

    # Create base Zarr group
    z = create_mock_zarr_group()
    obs_index = [str(i) for i in range(obs_count)]

    if is_multiome:
        # Determine cluster names based on entity type first
        if is_pan_azimuth:
            cluster_names = [
                "leiden_wnn",
                "leiden_rna",
                "final_level_labels",
                "full_hierarchical_labels",
                "CL_Label",
                "azimuth_broad",
                "azimuth_medium",
                "azimuth_fine",
            ]
        else:
            cluster_names = ["leiden_wnn", "leiden_rna", "cluster_cbg", "cluster_cbb"]
            if is_annotated:
                cluster_names.append("predicted_label")

        # Use fixture factory for multiome structure
        modalities = ["rna"]
        if "atac" in entity_path.name.lower():
            modalities.append("atac")

        populate_multiome_zarr(z, obs_count=obs_count, modalities=modalities)

        # Add cluster groups (populate_multiome_zarr creates leiden_rna/leiden_wnn, so we need to handle carefully)
        obs = z["mod/rna/obs"]
        # Only create groups that don't already exist as arrays
        existing_keys = set(obs.keys())
        groups_to_create = [name for name in cluster_names if name not in existing_keys]

        if groups_to_create:
            groups = obs.create_groups(*groups_to_create)
            for group in groups:
                group["categories"] = zarr.array(["0", "1", "2"])

        # Convert leiden arrays to groups with categories if they exist
        for name in ["leiden_wnn", "leiden_rna"]:
            if name in existing_keys and isinstance(obs[name], zarr.core.Array):
                # Delete the array and create a group instead
                del obs[name]
                group = obs.create_group(name)
                group["categories"] = zarr.array(["0", "1", "2"])
    else:
        # Create regular AnnData structure
        if is_annotated and not is_marker:
            # Use populate_anndata_zarr for annotated entities (includes all obs paths)
            populate_anndata_zarr(z, obs_count=obs_count, var_count=50, is_annotated=True)
        else:
            # Create manual structure for non-annotated or marker entities
            obs = z.create_group("obs")
            obs["_index"] = zarr.array(obs_index)

            # Add marker genes if needed
            if is_marker:
                gene_array = zarr.array(["ENSG00000139618", "ENSG00000139619", "ENSG00000139620"])
                obs["marker_gene_0"] = zarr.array(obs_index)
                obs.attrs["encoding-version"] = "0.1.0"

                var = z.create_group("var")
                var.attrs["_index"] = "index"
                var["index"] = gene_array
                var["hugo_symbol"] = zarr.array([0, 1, 2])
                var["hugo_symbol"].attrs["categories"] = "hugo_categories"
                var["hugo_categories"] = zarr.array(["gene123", "gene456", "gene789"])

    # Add annotation-specific metadata
    if is_annotated:
        obs_prefix_path = "mod/rna/obs" if is_multiome else "obs"
        obs_group = z[obs_prefix_path]
        path = f"{'mod/rna/' if is_multiome else ''}uns/annotation_metadata/is_annotated"
        z[path] = True

        if is_asct:
            # Create categorical array for ASCT
            obs_group["predicted.ASCT.celltype"] = zarr.array([f"asct_{i % 3}" for i in range(obs_count)])
        elif is_azimuth_labeled:
            obs_group["predicted_label"] = zarr.array([f"celltype_{i % 3}" for i in range(obs_count)])
            obs_group["predicted_CLID"] = zarr.array([f"CL:{1000000 + i % 3}" for i in range(obs_count)])
        elif is_pan_azimuth and not is_multiome:
            # For non-multiome pan-azimuth, we need to create azimuth columns
            azimuth_cols = ["azimuth_broad", "azimuth_medium", "azimuth_fine"]
            for col in azimuth_cols:
                # Create as group with categories (multiome already has these)
                group = obs_group.create_group(col)
                group["categories"] = zarr.array(["type_a", "type_b", "type_c"])
            # For multiome pan-azimuth, azimuth columns were already created in the multiome logic above

    # Add Visium-specific metadata
    if is_visium:
        z["uns/spatial/visium/scalefactors/spot_diameter_micrometers"] = 200.0

    # Mock HTTP requests for object-by-analyte entities
    if is_object_by_analyte_entity(entity_path):
        entity = json.loads(entity_path.read_text())
        mock_response = mocker.Mock()
        mock_response.json.return_value = entity.get("secondary_analysis_metadata")
        mock_response.raise_for_status.return_value = None
        mocker.patch("requests.get", return_value=mock_response)

    # Mock image metadata retrieval (used by imaging builders, harmless for others)
    mocker.patch("src.portal_visualization.builders.imaging_builders.get_image_metadata", return_value=None)
    mocker.patch("src.portal_visualization.builders.epic_builders.get_image_metadata", return_value=None)
    # Mock read_metadata_from_url (SegmentationMaskBuilder fetches zarr metadata over HTTP)
    mocker.patch(
        "src.portal_visualization.builders.epic_builders.SegmentationMaskBuilder.read_metadata_from_url",
        return_value=[],
    )

    # Apply mocks
    mocker.patch("zarr.open", return_value=z)
    if is_zip_entity(entity_path):
        mocker.patch("src.portal_visualization.data_access.read_zip_zarr", return_value=z)


# Programmatic test configurations to replace JSON fixtures
# These can be used instead of loading from good-fixtures/


def generate_rna_seq_test_cases():
    """Generate RNASeqAnnDataZarrViewConfBuilder test cases programmatically."""
    test_cases = []

    # Base UUID for RNA-seq tests
    base_uuid = "e65175561b4b17da5352e3837aa0e497"

    # Test case 1: Not annotated, published
    test_cases.append(
        (
            "RNASeqAnnDataZarrViewConfBuilder/generated-not-annotated-published",
            make_rna_seq_entity(
                uuid=base_uuid,
                is_annotated=False,
                is_published=True,
                soft_assaytype="salmon_sn_rnaseq_10x",
                data_types=["salmon_sn_rnaseq_10x"],
                files=[{"rel_path": "hubmap_ui/anndata-zarr/secondary_analysis.zarr/.zgroup"}],
            ),
        )
    )

    # Test case 2: Not annotated, QA
    test_cases.append(
        (
            "RNASeqAnnDataZarrViewConfBuilder/generated-not-annotated-qa",
            make_rna_seq_entity(
                uuid=f"{base_uuid}-qa",
                is_annotated=False,
                is_published=False,
                soft_assaytype="salmon_sn_rnaseq_10x",
                data_types=["salmon_sn_rnaseq_10x"],
                files=[{"rel_path": "hubmap_ui/anndata-zarr/secondary_analysis.zarr/.zgroup"}],
            ),
        )
    )

    # Test case 3: Not annotated, minimal, published
    test_cases.append(
        (
            "RNASeqAnnDataZarrViewConfBuilder/generated-not-annotated-minimal-published",
            make_rna_seq_entity(
                uuid=f"{base_uuid}-minimal",
                is_annotated=False,
                is_published=True,
                soft_assaytype="salmon_sn_rnaseq_10x",
                data_types=["salmon_sn_rnaseq_10x"],
                files=[{"rel_path": "hubmap_ui/anndata-zarr/secondary_analysis.zarr/.zgroup"}],
            ),
        )
    )

    # Test case 4: ASCT annotated, published
    test_cases.append(
        (
            "RNASeqAnnDataZarrViewConfBuilder/generated-asct-annotated-published",
            make_rna_seq_entity(
                uuid=f"{base_uuid}-asct-pub",
                is_annotated=True,
                is_published=True,
                soft_assaytype="salmon_sn_rnaseq_10x",
                data_types=["salmon_sn_rnaseq_10x"],
                files=[{"rel_path": "hubmap_ui/anndata-zarr/secondary_analysis.zarr/.zgroup"}],
            ),
        )
    )

    # Test case 5: ASCT annotated, QA
    test_cases.append(
        (
            "RNASeqAnnDataZarrViewConfBuilder/generated-asct-annotated-qa",
            make_rna_seq_entity(
                uuid=f"{base_uuid}-asct-qa",
                is_annotated=True,
                is_published=False,
                soft_assaytype="salmon_sn_rnaseq_10x",
                data_types=["salmon_sn_rnaseq_10x"],
                files=[{"rel_path": "hubmap_ui/anndata-zarr/secondary_analysis.zarr/.zgroup"}],
            ),
        )
    )

    # Test case 6: Zip compressed
    test_cases.append(
        (
            "RNASeqAnnDataZarrViewConfBuilder/generated-asct-annotated-zip",
            make_rna_seq_entity(
                uuid=f"{base_uuid}-zip",
                is_annotated=True,
                is_published=True,
                soft_assaytype="salmon_sn_rnaseq_10x",
                data_types=["salmon_sn_rnaseq_10x"],
                files=[{"rel_path": "hubmap_ui/anndata-zarr/secondary_analysis.zarr.zip"}],
            ),
        )
    )

    # Test case 7: Predicted label annotated, published
    test_cases.append(
        (
            "RNASeqAnnDataZarrViewConfBuilder/generated-predicted-label-annotated-published",
            make_rna_seq_entity(
                uuid=f"{base_uuid}-predicted-pub",
                is_annotated=True,
                is_published=True,
                soft_assaytype="salmon_sn_rnaseq_10x",
                data_types=["salmon_sn_rnaseq_10x"],
                files=[{"rel_path": "hubmap_ui/anndata-zarr/secondary_analysis.zarr/.zgroup"}],
            ),
        )
    )

    # Test case 8: Predicted label annotated, QA
    test_cases.append(
        (
            "RNASeqAnnDataZarrViewConfBuilder/generated-predicted-label-annotated-qa",
            make_rna_seq_entity(
                uuid=f"{base_uuid}-predicted-qa",
                is_annotated=True,
                is_published=False,
                soft_assaytype="salmon_sn_rnaseq_10x",
                data_types=["salmon_sn_rnaseq_10x"],
                files=[{"rel_path": "hubmap_ui/anndata-zarr/secondary_analysis.zarr/.zgroup"}],
            ),
        )
    )

    # Test case 9: Pan-Azimuth annotated, published
    test_cases.append(
        (
            "RNASeqAnnDataZarrViewConfBuilder/generated-pan-az-annotated-published",
            make_rna_seq_entity(
                uuid=f"{base_uuid}-pan-az",
                is_annotated=True,
                is_published=True,
                soft_assaytype="salmon_sn_rnaseq_10x",
                data_types=["salmon_sn_rnaseq_10x"],
                files=[{"rel_path": "hubmap_ui/anndata-zarr/secondary_analysis.zarr/.zgroup"}],
            ),
        )
    )

    # Test case 9: Marker gene test to cover line 389
    test_cases.append(
        (
            "RNASeqAnnDataZarrViewConfBuilder/generated-marker=gene123",
            make_rna_seq_entity(
                uuid=f"{base_uuid}-marker",
                is_annotated=False,
                is_published=True,
                soft_assaytype="salmon_sn_rnaseq_10x",
                data_types=["salmon_sn_rnaseq_10x"],
                files=[{"rel_path": "hubmap_ui/anndata-zarr/secondary_analysis.zarr/.zgroup"}],
            ),
        )
    )

    return test_cases


def generate_multiome_test_cases():
    """Generate MultiomicAnndataZarrViewConfBuilder test cases programmatically."""
    from .fixtures import make_entity

    test_cases = []
    base_uuid = "a1234567890abcdef1234567890abcde"

    # Test case 1: Basic multiome, not annotated
    test_cases.append(
        (
            "MultiomicAnndataZarrViewConfBuilder/generated-multiome",
            make_entity(
                uuid=base_uuid,
                status="Published",
                hints=["rna", "atac", "is_sc"],
                soft_assaytype="multiome",
                data_types=["multiome"],
                files=[{"rel_path": "hubmap_ui/mudata-zarr/secondary_analysis.zarr/.zgroup"}],
            ),
        )
    )

    # Test case 2: Multiome with ASCT annotation
    test_cases.append(
        (
            "MultiomicAnndataZarrViewConfBuilder/generated-multiome-is-annotated",
            make_entity(
                uuid=f"{base_uuid}-annotated",
                status="Published",
                hints=["rna", "atac", "is_annotated", "is_sc"],
                soft_assaytype="multiome",
                data_types=["multiome"],
                files=[{"rel_path": "hubmap_ui/mudata-zarr/secondary_analysis.zarr/.zgroup"}],
            ),
        )
    )

    # Test case 3: Multiome with pan-azimuth annotation
    test_cases.append(
        (
            "MultiomicAnndataZarrViewConfBuilder/generated-multiome-is-annotated-pan-az",
            make_entity(
                uuid=f"{base_uuid}-pan-az",
                status="Published",
                hints=["rna", "atac", "is_annotated", "is_sc"],
                soft_assaytype="multiome",
                data_types=["multiome"],
                files=[{"rel_path": "hubmap_ui/mudata-zarr/secondary_analysis.zarr/.zgroup"}],
            ),
        )
    )

    return test_cases


def generate_spatial_multiome_test_cases():
    """Generate SpatialMultiomicAnnDataZarrViewConfBuilder test cases programmatically."""
    from .fixtures import make_entity

    test_cases = []
    base_uuid = "b1234567890abcdef1234567890abcde"

    # Test case 1: Visium spatial multiome
    test_cases.append(
        (
            "SpatialMultiomicAnnDataZarrViewConfBuilder/generated-visium",
            make_entity(
                uuid=base_uuid,
                status="Published",
                hints=["rna", "is_image", "anndata", "spatial"],
                soft_assaytype="visium-no-probes",
                data_types=["visium-no-probes"],
                files=[{"rel_path": "hubmap_ui/anndata-zarr/secondary_analysis.zarr/.zgroup"}],
            ),
        )
    )

    return test_cases


def generate_xenium_test_cases():
    """Generate XeniumMultiomicAnnDataZarrViewConfBuilder test cases programmatically."""
    from .fixtures import make_entity

    test_cases = []
    base_uuid = "c1234567890abcdef1234567890abcde"

    # Test case 1: Xenium spatial - requires "xenium" and "is_image" hints
    test_cases.append(
        (
            "XeniumMultiomicAnnDataZarrViewConfBuilder/generated-xenium",
            make_entity(
                uuid=base_uuid,
                status="Published",
                hints=["xenium", "is_image"],
                data_types=["xenium"],
                files=[
                    {"rel_path": "hubmap_ui/anndata-zarr/secondary_analysis.zarr.zip"},
                    {"rel_path": "Xenium.zarr/.zgroup"},
                    {"rel_path": "Xenium.zarr/images/morphology_focus/.zgroup"},
                ],
            ),
        )
    )

    return test_cases


def generate_spatial_rna_seq_test_cases():
    """Generate SpatialRNASeqAnnDataZarrViewConfBuilder test cases programmatically."""
    from .fixtures import make_entity

    test_cases = []
    base_uuid = "d1234567890abcdef1234567890abcde"

    # Test case 1: Spatial RNA-seq minimal - uses salmon_rnaseq_slideseq assay
    test_cases.append(
        (
            "SpatialRNASeqAnnDataZarrViewConfBuilder/generated-minimal",
            make_entity(
                uuid=f"{base_uuid}-minimal",
                status="Published",
                hints=["is_sc", "rna"],
                soft_assaytype="salmon_rnaseq_slideseq",
                data_types=["salmon_rnaseq_slideseq"],
                files=[{"rel_path": "hubmap_ui/anndata-zarr/secondary_analysis.zarr/.zgroup"}],
            ),
        )
    )

    # Test case 2: Spatial RNA-seq with marker
    test_cases.append(
        (
            "SpatialRNASeqAnnDataZarrViewConfBuilder/generated-marker-gene123",
            make_entity(
                uuid=f"{base_uuid}-marker",
                status="Published",
                hints=["is_sc", "rna"],
                soft_assaytype="salmon_rnaseq_slideseq",
                data_types=["salmon_rnaseq_slideseq"],
                files=[{"rel_path": "hubmap_ui/anndata-zarr/secondary_analysis.zarr/.zgroup"}],
            ),
        )
    )

    return test_cases


def generate_object_by_analyte_test_cases():
    """Generate ObjectByAnalyteConfBuilder test cases programmatically."""
    from .fixtures import make_entity

    test_cases = []
    base_uuid = "a1583467d20b420f20e7d97528305021"

    # Standard files for all object-by-analyte datasets
    standard_files = [
        {"rel_path": "extras/transformations/hubmap_ui/mudata-zarr/calculated_metadata.json"},
        {"rel_path": "extras/transformations/hubmap_ui/mudata-zarr/secondary_analysis.zarr.zip"},
        {"rel_path": "extras/transformations/hubmap_ui/mudata-zarr/secondary_analysis_metadata.json"},
    ]

    # Test variants with different metadata configurations
    # Format: (variant_name, modality_name, var_keys, n_vars, num_embeddings, is_spatial)
    variants = [
        ("fake-object-by-analyte", "HT_processed", ["hugo_symbol", "mean", "n_cells", "std"], 29078, 1, False),
        (
            "many-embeddings-object-by-analyte",
            "many_embeddings",
            ["hugo_symbol", "mean", "n_cells", "std"],
            22000,
            1,
            False,
        ),
        (
            "no-feature-labels-object-by-analyte",
            "no_labels_data",
            ["feature_id", "mean", "n_cells", "std"],
            15000,
            1,
            False,
        ),
        (
            "single-embedding-object-by-analyte",
            "single_embedding",
            ["hugo_symbol", "mean", "n_cells", "std"],
            18000,
            1,
            False,
        ),
        ("spatial-object-by-analyte", "spatial_data", ["hugo_symbol", "mean", "n_cells", "std"], 25000, 1, True),
        (
            "three-embeddings-object-by-analyte",
            "three_embeddings",
            ["hugo_symbol", "mean", "n_cells", "std"],
            20000,
            1,
            False,
        ),
        ("uniprot-object-by-analyte", "protein_data", ["uniprot_id", "mean", "n_cells", "std"], 20000, 1, False),
        ("zero-dimensions-object-by-analyte", "empty_data", ["hugo_symbol", "mean", "n_cells", "std"], 0, 1, False),
        # Additional test cases for multiple embeddings to cover scatterplot layout edge cases
        (
            "two-embeddings-object-by-analyte",
            "protein_2emb",
            ["hugo_symbol", "mean", "n_cells", "std"],
            15000,
            2,
            False,
        ),
        (
            "three-embeddings-layout-object-by-analyte",
            "protein_3emb",
            ["hugo_symbol", "mean", "n_cells", "std"],
            15000,
            3,
            False,
        ),
        (
            "four-embeddings-object-by-analyte",
            "protein_4emb",
            ["hugo_symbol", "mean", "n_cells", "std"],
            15000,
            4,
            False,
        ),
    ]

    for i, (variant_name, modality_name, var_keys, n_vars, num_embeddings, is_spatial) in enumerate(variants):
        # Create embedding keys based on num_embeddings
        obsm_keys = [f"X_embedding_{j}" if j > 0 else "X_umap" for j in range(num_embeddings)]

        # Add X_spatial to obsm_keys if this is a spatial variant
        if is_spatial:
            obsm_keys.append("X_spatial")

        metadata = {
            "epic_type": "mudata",
            "modalities": [
                {
                    "annotations": ["leiden"],
                    "n_obs": 1000,
                    "n_vars": n_vars,
                    "name": modality_name,
                    "obs_keys": ["sample_id"],
                    "obsm_keys": obsm_keys,
                    "var_keys": var_keys,
                }
            ],
            "n_obs": 1000,
            "n_vars": {modality_name: n_vars},
            "obs_keys": ["sample_id"],
            "obsm_keys": [modality_name],
            "shape": [1000, n_vars],
            "var_keys": var_keys,
        }
        test_cases.append(
            (
                f"ObjectByAnalyteConfBuilder/generated-{variant_name}",
                make_entity(
                    uuid=f"{base_uuid}-{i}",
                    status="Published",
                    hints=["epic"],
                    soft_assaytype="object-x-analyte",
                    data_types=None,
                    files=standard_files,
                    secondary_analysis_metadata=metadata,
                ),
            )
        )

    return test_cases


def generate_tiled_sprm_test_cases():
    """Generate TiledSPRMViewConf Builder test cases programmatically."""
    from .fixtures import make_entity

    test_cases = []

    # Test case 1: No cells variant
    test_cases.append(
        (
            "TiledSPRMViewConfBuilder/generated-no-cells",
            make_entity(
                uuid="b69d1e2ad1bf1455eee991fce301b191-no-cells",
                status="Published",
                soft_assaytype="codex_cytokit_v1",
                data_types=["codex_cytokit_v1"],
                hints=["codex", "is_image", "is_tiled", "json_based"],
                files=[
                    {"rel_path": "output/extract/expressions/ome-tiff/reg1.ome.tiff"},
                ],
                immediate_ancestors=[{"data_types": ["codex_cytokit_v1"]}],
                mapped_data_types=["CODEX [Cytokit + SPRM]"],
            ),
        )
    )

    # Test case 2: With cells variant
    test_cases.append(
        (
            "TiledSPRMViewConfBuilder/generated-with-cells",
            make_entity(
                uuid="b69d1e2ad1bf1455eee991fce301b191-with-cells",
                status="Published",
                soft_assaytype="codex_cytokit_v1",
                data_types=["codex_cytokit_v1"],
                hints=["codex", "is_image", "is_tiled", "json_based"],
                files=[
                    {"rel_path": "output/extract/expressions/ome-tiff/reg1.ome.tiff"},
                    {"rel_path": "output_json/reg1.cells.json"},
                    {"rel_path": "output_json/reg1.cell-sets.json"},
                    {"rel_path": "output_json/reg1.clusters.json"},
                ],
                mapped_data_types=["CODEX [Cytokit + SPRM]"],
            ),
        )
    )

    return test_cases


def generate_stitched_cytokit_sprm_test_cases():
    """Generate StitchedCytokitSPRMViewConfBuilder test cases programmatically."""
    from .fixtures import make_entity

    test_cases = []

    # Test case: Stitched SPRM with AnnData Zarr
    test_cases.append(
        (
            "StitchedCytokitSPRMViewConfBuilder/generated-04e7385339167e541ad42a2636e18398",
            make_entity(
                uuid="04e7385339167e541ad42a2636e18398",
                status="Published",
                soft_assaytype="codex_cytokit_v1",
                data_types=["codex_cytokit_v1"],
                hints=["codex", "is_image", "is_tiled"],
                files=[
                    {
                        "rel_path": "anndata-zarr/reg1_stitched_expressions-anndata.zarr/.zgroup",
                        "description": "AnnData Zarr store for storing and visualizing SPRM outputs.",
                    },
                    {
                        "rel_path": "ometiff-pyramids/stitched/expressions/reg1_stitched_expressions.ome.tif",
                        "description": "OME-TIFF pyramid file",
                    },
                    {
                        "rel_path": "ometiff-pyramids/stitched/mask/reg1_stitched_mask.ome.tif",
                        "description": "OME-TIFF pyramid file",
                    },
                ],
                mapped_data_types=["CODEX [Cytokit + SPRM]"],
                metadata={
                    "dag_provenance_list": [
                        {
                            "name": "sprm-to-anndata.cwl",
                            "origin": "https://github.com/hubmapconsortium/portal-containers",
                        }
                    ]
                },
            ),
        )
    )

    return test_cases


def generate_imaging_builder_test_cases():
    """Generate imaging builder test cases programmatically."""
    from .fixtures import make_entity

    test_cases = []

    # ImagePyramidViewConfBuilder
    test_cases.append(
        (
            "ImagePyramidViewConfBuilder/generated-fake",
            make_entity(
                uuid="f9ae931b8b49252f150d7f8bf1d2d13f",
                status="QA",
                soft_assaytype="image_pyramid",
                data_types=["image_pyramid", "PAS"],
                hints=["is_support", "pyramid", "is_image"],
                files=[
                    {"rel_path": "ometiff-pyramids/processedMicroscopy/VAN0003-LK-33-2-PAS_FFPE.ome.tif"},
                    {"rel_path": "ometiff-pyramids/separate/should-be-ignored.ome.tif"},
                    {"rel_path": "output_offsets/processedMicroscopy/VAN0003-LK-33-2-PAS_FFPE.offsets.json"},
                ],
                immediate_ancestors=[{"data_types": ["PAS"]}],
                parent={"uuid": "8adc3c31ca84ec4b958ed20a7c4f4919"},
            ),
        )
    )

    # KaggleSegImagePyramidViewConfBuilder (Kaggle-2: base images co-located, no parent)
    test_cases.append(
        (
            "KaggleSegImagePyramidViewConfBuilder/generated-fake",
            make_entity(
                uuid="23a25976beb8c02ab589b13a05b28c55",
                status="QA",
                soft_assaytype="h-and-e",
                data_types=["Histology"],
                hints=["segmentation_mask", "pyramid", "is_image"],
                files=[
                    {"rel_path": "ometiff-pyramids/lab_processed/images/B001_SB-reg005.ome.tif"},
                    {"rel_path": "output_offsets/lab_processed/images/B001_SB-reg005.offsets.json"},
                    {"rel_path": "image_metadata/lab_processed/images/B001_SB-reg005.metadata.json"},
                    {"rel_path": "ometiff-pyramids/B001_SB-reg005.segmentations.ome.tif"},
                    {"rel_path": "output_offsets/B001_SB-reg005.segmentations.offsets.json"},
                    {"rel_path": "image_metadata/B001_SB-reg005.segmentations.metadata.json"},
                ],
                immediate_ancestors=[{"data_types": ["Histology"]}],
            ),
        )
    )

    # IMSViewConfBuilder
    test_cases.append(
        (
            "IMSViewConfBuilder/generated-fake",
            make_entity(
                uuid="a6116772446f6d1c1f6b3d2e9735cfe0",
                status="QA",
                soft_assaytype="image_pyramid",
                hints=["is_support", "pyramid", "is_image"],
                files=[
                    {"rel_path": "ometiff-pyramids/ometiffs/separate/VAN0003-LK-32-21-IMS_NegMode_mz909.606.ome.tif"},
                    {"rel_path": "ometiff-pyramids/ometiffs/separate/VAN0003-LK-32-21-IMS_NegMode_mz922.609.ome.tif"},
                    {"rel_path": "ometiff-pyramids/ometiffs/VAN0003-LK-32-21-IMS_NegMode_multilayer.ome.tif"},
                    {
                        "rel_path": "output_offsets/ometiffs/separate/VAN0003-LK-32-21-IMS_NegMode_mz909.606.offsets.json"
                    },
                    {
                        "rel_path": "output_offsets/ometiffs/separate/VAN0003-LK-32-21-IMS_NegMode_mz922.609.offsets.json"
                    },
                    {"rel_path": "output_offsets/ometiffs/VAN0003-LK-32-21-IMS_NegMode_multilayer.offsets.json"},
                ],
                parent={"uuid": "3bc3ad124014a632d558255626bf38c9"},
            ),
        )
    )

    # NanoDESIViewConfBuilder
    test_cases.append(
        (
            "NanoDESIViewConfBuilder/generated-fake",
            make_entity(
                uuid="e1c4370da5523ab5c9be581d1d76ca20",
                status="QA",
                soft_assaytype="image_pyramid",
                data_types=["image_pyramid"],
                hints=["is_image", "pyramid", "is_support"],
                files=[
                    {"rel_path": "ometiff-pyramids/ometiffs/VAN0003-LK-32-21-IMS_NegMode_multilayer.ome.tif"},
                    {"rel_path": "output_offsets/ometiffs/VAN0003-LK-32-21-IMS_NegMode_multilayer.offsets.json"},
                ],
                parent={"uuid": "6b93107731199733f266bbd0f3bc9747"},
            ),
        )
    )

    return test_cases


def generate_geomx_test_cases():
    """Generate GeoMxImagePyramidViewConfBuilder test cases programmatically."""
    from .fixtures import make_entity

    test_cases = []

    # Test case 1: Standard zarr (not zip)
    test_cases.append(
        (
            "GeoMxImagePyramidViewConfBuilder/generated-fake",
            make_entity(
                uuid="bc7239d27b79e087c788600261f073e5-zarr",
                status="QA",
                soft_assaytype="",
                data_types=["Histology"],
                hints=["geomx", "is_image"],
                files=[
                    {"rel_path": "ometiff-pyramids/GeoMx4_Niedernhofer_Project_060.segmentations.ome.tif"},
                    {"rel_path": "ometiff-pyramids/lab_processed/images/GeoMx4_Niedernhofer_Project_060.ome.tif"},
                    {"rel_path": "output_offsets/GeoMx4_Niedernhofer_Project_060.segmentations.offsets.json"},
                    {"rel_path": "output_offsets/lab_processed/images/GeoMx4_Niedernhofer_Project_060.offsets.json"},
                    {"rel_path": "image_metadata/GeoMx4_Niedernhofer_Project_060.segmentations.metadata.json"},
                    {"rel_path": "image_metadata/lab_processed/images/GeoMx4_Niedernhofer_Project_060.metadata.json"},
                    {"rel_path": "output_ome_segments/GeoMx4_Niedernhofer_Project_060.obsSegmentations.json"},
                    {"rel_path": "output_ome_segments/GeoMx4_Niedernhofer_Project_060.roi.zarr/.zgroup"},
                    {"rel_path": "output_ome_segments/GeoMx4_Niedernhofer_Project_060.aoi.zarr/.zgroup"},
                ],
                immediate_ancestors=[{"data_types": ["Histology"]}],
            ),
        )
    )

    # Test case 2: Zip zarr variant
    test_cases.append(
        (
            "GeoMxImagePyramidViewConfBuilder/generated-fake-zarr-zip",
            make_entity(
                uuid="bc7239d27b79e087c788600261f073e5-zip",
                status="QA",
                soft_assaytype="",
                data_types=["Histology"],
                hints=["geomx", "is_image"],
                files=[
                    {"rel_path": "ometiff-pyramids/GeoMx4_Niedernhofer_Project_060.segmentations.ome.tif"},
                    {"rel_path": "ometiff-pyramids/lab_processed/images/GeoMx4_Niedernhofer_Project_060.ome.tif"},
                    {"rel_path": "output_offsets/GeoMx4_Niedernhofer_Project_060.segmentations.offsets.json"},
                    {"rel_path": "output_offsets/lab_processed/images/GeoMx4_Niedernhofer_Project_060.offsets.json"},
                    {"rel_path": "image_metadata/GeoMx4_Niedernhofer_Project_060.segmentations.metadata.json"},
                    {"rel_path": "image_metadata/lab_processed/images/GeoMx4_Niedernhofer_Project_060.metadata.json"},
                    {"rel_path": "output_ome_segments/GeoMx4_Niedernhofer_Project_060.obsSegmentations.json"},
                    {"rel_path": "output_ome_segments/GeoMx4_Niedernhofer_Project_060.roi.zarr.zip"},
                    {"rel_path": "output_ome_segments/GeoMx4_Niedernhofer_Project_060.aoi.zarr.zip"},
                ],
                immediate_ancestors=[{"data_types": ["Histology"]}],
            ),
        )
    )

    return test_cases


def generate_multi_image_sprm_test_cases():
    """Generate MultiImageSPRMAnndataViewConfBuilder test cases programmatically."""
    from .fixtures import make_entity

    test_cases = []

    # Multi-image SPRM with marker gene parameter
    files = [
        {"rel_path": "anndata-zarr/reg001_S20030086_region_001_expr-anndata.zarr/.zgroup"},
        {"rel_path": "anndata-zarr/reg001_S20030086_region_001_expr-anndata.zarr/obs/.zgroup"},
        {"rel_path": "anndata-zarr/reg001_S20030086_region_001_expr-anndata.zarr/obsm/.zgroup"},
        {"rel_path": "anndata-zarr/reg001_S20030086_region_001_expr-anndata.zarr/var/.zgroup"},
        {"rel_path": "data.json"},
        {"rel_path": "output_offsets/pipeline_output/expr/reg001_S20030086_region_001_expr.offsets.json"},
        {"rel_path": "ometiff-pyramids/pipeline_output/expr/reg001_S20030086_region_001_expr.ome.tif"},
        {"rel_path": "output_offsets/pipeline_output/mask/reg001_S20030086_region_001_mask.offsets.json"},
        {"rel_path": "ometiff-pyramids/pipeline_output/mask/reg001_S20030086_region_001_mask.ome.tif"},
        {"rel_path": "anndata-zarr/reg001_S20030085_region_001_expr-anndata.zarr/.zgroup"},
        {"rel_path": "anndata-zarr/reg001_S20030085_region_001_expr-anndata.zarr/obs/.zgroup"},
        {"rel_path": "anndata-zarr/reg001_S20030085_region_001_expr-anndata.zarr/obsm/.zgroup"},
        {"rel_path": "anndata-zarr/reg001_S20030085_region_001_expr-anndata.zarr/var/.zgroup"},
        {"rel_path": "output_offsets/pipeline_output/expr/reg001_S20030085_region_001_expr.offsets.json"},
        {"rel_path": "ometiff-pyramids/pipeline_output/expr/reg001_S20030085_region_001_expr.ome.tif"},
        {"rel_path": "output_offsets/pipeline_output/mask/reg001_S20030085_region_001_mask.offsets.json"},
        {"rel_path": "ometiff-pyramids/pipeline_output/mask/reg001_S20030085_region_001_mask.ome.tif"},
    ]

    test_cases.append(
        (
            "MultiImageSPRMAnndataViewConfBuilder/generated-fake-marker=gene123",
            make_entity(
                uuid="2b3e99536a7da4f78bf02a8b6ce92b30",
                status="Published",
                soft_assaytype="celldive_deepcell",
                data_types=["celldive_deepcell"],
                hints=["is_tiled", "is_image", "anndata", "sprm"],
                files=files,
            ),
        )
    )

    # Zip variant - tests that zarr_store() correctly appends .zip suffix
    zip_files = [
        {"rel_path": "anndata-zarr/reg001_S20030086_region_001_expr-anndata.zarr.zip"},
        {"rel_path": "data.json"},
        {"rel_path": "output_offsets/pipeline_output/expr/reg001_S20030086_region_001_expr.offsets.json"},
        {"rel_path": "ometiff-pyramids/pipeline_output/expr/reg001_S20030086_region_001_expr.ome.tif"},
        {"rel_path": "output_offsets/pipeline_output/mask/reg001_S20030086_region_001_mask.offsets.json"},
        {"rel_path": "ometiff-pyramids/pipeline_output/mask/reg001_S20030086_region_001_mask.ome.tif"},
    ]

    test_cases.append(
        (
            "MultiImageSPRMAnndataViewConfBuilder/generated-fake-zip-marker=gene123",
            make_entity(
                uuid="2b3e99536a7da4f78bf02a8b6ce92b30-zip",
                status="Published",
                soft_assaytype="celldive_deepcell",
                data_types=["celldive_deepcell"],
                hints=["is_tiled", "is_image", "anndata", "sprm"],
                files=zip_files,
            ),
        )
    )

    return test_cases


def generate_legacy_json_test_cases():
    """Generate legacy JSON-based builder test cases programmatically."""
    from .fixtures import make_entity

    test_cases = []

    # ATACSeqViewConfBuilder
    test_cases.append(
        (
            "ATACSeqViewConfBuilder/generated-fake",
            make_entity(
                uuid="fake-atac-seq-uuid",
                status="QA",
                soft_assaytype="sn_atac_seq",
                data_types=["sn_atac_seq"],
                hints=["is_sc", "atac", "json_based"],
                files=[
                    {"rel_path": "output/umap_coords_clusters.cells.json"},
                    {"rel_path": "output/umap_coords_clusters.cell-sets.json"},
                ],
            ),
        )
    )

    # RNASeqViewConfBuilder
    test_cases.append(
        (
            "RNASeqViewConfBuilder/generated-fake",
            make_entity(
                uuid="c019a1cd35aab4d2b4a6ff221e92aaab",
                status="Published",
                soft_assaytype="salmon_sn_rnaseq_10x",
                data_types=["salmon_rnaseq_10x"],
                hints=["is_sc", "rna", "json_based"],
                files=[
                    {"rel_path": "cluster-marker-genes/output/cluster_marker_genes.cells.json"},
                    {"rel_path": "cluster-marker-genes/output/cluster_marker_genes.cell-sets.json"},
                ],
                mapped_data_types=[],
            ),
        )
    )

    return test_cases


def generate_seqfish_test_cases():
    """Generate SeqFISHViewConfBuilder test cases programmatically."""
    from .fixtures import make_entity

    test_cases = []

    # SeqFISH image pyramid
    test_cases.append(
        (
            "SeqFISHViewConfBuilder/generated-fake",
            make_entity(
                uuid="9db61adfc017670a196ea9b3ca1852a0",
                status="QA",
                soft_assaytype="seqFish",
                data_types=["image_pyramid", "seqFish"],
                hints=["is_image", "pyramid", "is_support"],
                files=[
                    {"rel_path": "ometiff-pyramids/final_mRNA_background/MMStack_Pos12.ome.tif"},
                    {"rel_path": "ometiff-pyramids/final_mRNA_background/MMStack_Pos13.ome.tif"},
                    {"rel_path": "ometiff-pyramids/HybCycle_12/MMStack_Pos12.ome.tif"},
                    {"rel_path": "ometiff-pyramids/HybCycle_12/MMStack_Pos13.ome.tif"},
                    {"rel_path": "output_offsets/final_mRNA_background/MMStack_Pos12.offsets.json"},
                    {"rel_path": "output_offsets/final_mRNA_background/MMStack_Pos13.offsets.json"},
                    {"rel_path": "output_offsets/HybCycle_12/MMStack_Pos12.offsets.json"},
                    {"rel_path": "output_offsets/HybCycle_12/MMStack_Pos13.offsets.json"},
                ],
                parent={"uuid": "c6a254b2dc2ed46b002500ade163a7cc"},
            ),
        )
    )

    return test_cases


def generate_epic_seg_test_cases():
    """Generate SegmentationMaskBuilder test cases programmatically."""
    from .fixtures import make_entity

    test_cases = []

    # Standard zarr variant
    test_cases.append(
        (
            "SegmentationMaskBuilder/generated-fake",
            make_entity(
                uuid="df7cac7cb67a822f7007b57c4d8f5e7d",
                status="QA",
                soft_assaytype="PAS",
                data_types=["image_pyramid", "PAS"],
                hints=["segmentation_mask", "is_image", "pyramid", "epic"],
                files=[
                    {"rel_path": "extras/transformations/ometiff-pyramids/lab_processed/images/91706.ome.tif"},
                    {"rel_path": "extras/transformations/output_offsets/lab_processed/images/91706.offsets.json"},
                    {"rel_path": "extras/transformations/image_metadata/lab_processed/images/91706.metadata.json"},
                    {"rel_path": "extras/transformations/ometiff-pyramids/91706.segmentations.ome.tif"},
                    {"rel_path": "extras/transformations/output_offsets/91706.segmentations.offsets.json"},
                    {"rel_path": "extras/transformations/image_metadata/91706.segmentations.metadata.json"},
                    {
                        "rel_path": "extras/transformations/hubmap_ui/seg-to-mudata-zarr/objects.zarr/arteries-arterioles.zarr/.zgroup"
                    },
                    {
                        "rel_path": "extras/transformations/hubmap_ui/seg-to-mudata-zarr/objects.zarr/tubules.zarr/.zgroup"
                    },
                    {
                        "rel_path": "extras/transformations/hubmap_ui/seg-to-mudata-zarr/objects.zarr/glomeruli.zarr/.zgroup"
                    },
                    {"rel_path": "extras/transformations/hubmap_ui/seg-to-mudata-zarr/objects.zarr/metadata.json"},
                ],
                immediate_ancestors=[{"data_types": ["PAS"]}],
            ),
        )
    )

    # Zarr.zip variant
    test_cases.append(
        (
            "SegmentationMaskBuilder/generated-fake-zarr-zip",
            make_entity(
                uuid="df7cac7cb67a822f7007b57c4d8f5e7d-zip",
                status="QA",
                soft_assaytype="PAS",
                data_types=["image_pyramid", "PAS"],
                hints=["segmentation_mask", "is_image", "pyramid", "epic"],
                files=[
                    {"rel_path": "extras/transformations/ometiff-pyramids/lab_processed/images/91706.ome.tif"},
                    {"rel_path": "extras/transformations/output_offsets/lab_processed/images/91706.offsets.json"},
                    {"rel_path": "extras/transformations/image_metadata/lab_processed/images/91706.metadata.json"},
                    {"rel_path": "extras/transformations/ometiff-pyramids/91706.segmentations.ome.tif"},
                    {"rel_path": "extras/transformations/output_offsets/91706.segmentations.offsets.json"},
                    {"rel_path": "extras/transformations/image_metadata/91706.segmentations.metadata.json"},
                    {"rel_path": "extras/transformations/hubmap_ui/seg-to-mudata-zarr/objects.zarr.zip"},
                ],
                immediate_ancestors=[{"data_types": ["PAS"]}],
            ),
        )
    )

    return test_cases


def generate_kaggle1_seg_test_cases():
    """Generate Kaggle1SegImagePyramidViewConfBuilder test cases programmatically."""
    from .fixtures import make_entity

    test_cases = []

    # Kaggle-1 (external): seg masks only, base images in parent's support entity
    test_cases.append(
        (
            "Kaggle1SegImagePyramidViewConfBuilder/generated-fake",
            make_entity(
                uuid="kaggle1-seg-fake-uuid",
                status="QA",
                soft_assaytype="h-and-e",
                data_types=["Histology"],
                hints=["segmentation_mask", "pyramid", "is_image"],
                files=[
                    {"rel_path": "ometiff-pyramids/B001_SB-reg005.segmentations.ome.tif"},
                    {"rel_path": "output_offsets/B001_SB-reg005.segmentations.offsets.json"},
                    {"rel_path": "image_metadata/B001_SB-reg005.segmentations.metadata.json"},
                ],
                immediate_ancestors=[{"data_types": ["Histology"]}],
                parent={"uuid": "kaggle1-parent-uuid"},
            ),
        )
    )

    # Kaggle-1 (co-located): base images AND seg masks both in entity files
    test_cases.append(
        (
            "Kaggle1SegImagePyramidViewConfBuilder/generated-colocated",
            make_entity(
                uuid="kaggle1-seg-colocated-uuid",
                status="QA",
                soft_assaytype="h-and-e",
                data_types=["Histology"],
                hints=["segmentation_mask", "pyramid", "is_image"],
                files=[
                    {"rel_path": "ometiff-pyramids/lab_processed/images/VAN0052-RK-3-81-PAS.ome.tif"},
                    {"rel_path": "output_offsets/lab_processed/images/VAN0052-RK-3-81-PAS.offsets.json"},
                    {"rel_path": "image_metadata/lab_processed/images/VAN0052-RK-3-81-PAS.metadata.json"},
                    {"rel_path": "ometiff-pyramids/VAN0052-RK-3-81-PAS.ome_mask.ome.tif"},
                    {"rel_path": "output_offsets/VAN0052-RK-3-81-PAS.ome_mask.offsets.json"},
                    {"rel_path": "image_metadata/VAN0052-RK-3-81-PAS.ome_mask.metadata.json"},
                ],
                immediate_ancestors=[{"data_types": ["Histology"]}],
                parent={"uuid": "kaggle1-parent-uuid"},
            ),
        )
    )

    return test_cases


def generate_null_builder_test_cases():
    """Generate NullViewConfBuilder test cases programmatically."""
    from .fixtures import make_entity

    test_cases = []

    # Empty entity (no data types, no hints)
    test_cases.append(
        (
            "NullViewConfBuilder/generated-empty",
            make_entity(
                uuid="2c2179ea741d3bbb47772172a316a2bf",
                soft_assaytype="bulk-RNA",
                data_types=[],
                hints=[],
            ),
        )
    )

    # No visualization support (is_support without parent)
    test_cases.append(
        (
            "NullViewConfBuilder/generated-fake-no-support",
            make_entity(
                uuid="f9ae931b8b49252f150d7f8bf1d2d13f-bad",
                status="QA",
                soft_assaytype="image_pyramid",
                data_types=["image_pyramid", "PAS"],
                hints=["is_support", "pyramid", "is_image"],
                files=[
                    {"rel_path": "ometiff-pyramids/processedMicroscopy/VAN0003-LK-33-2-PAS_FFPE.ome.tif"},
                    {"rel_path": "ometiff-pyramids/separate/should-be-ignored.ome.tif"},
                    {"rel_path": "output_offsets/processedMicroscopy/VAN0003-LK-33-2-PAS_FFPE.offsets.json"},
                ],
                immediate_ancestors=[{"data_types": ["PAS"]}],
                # Note: no parent, which causes NullViewConfBuilder selection
            ),
        )
    )

    # Image pyramid without is_support and without parent
    test_cases.append(
        (
            "NullViewConfBuilder/generated-image-pyramid-no-parent",
            make_entity(
                uuid="a3b4c5d6e7f890123456789012345678",
                status="Published",
                soft_assaytype="image_pyramid",
                data_types=["image_pyramid", "PAS"],
                hints=["pyramid", "is_image"],  # Note: NO is_support hint
                files=[
                    {"rel_path": "ometiff-pyramids/lab_processed/images/sample.ome.tif"},
                    {"rel_path": "output_offsets/lab_processed/images/sample.offsets.json"},
                ],
                # Note: no parent, which causes NullViewConfBuilder selection
            ),
        )
    )

    # Image pyramid with parent but no special hints (covers builder_factory line 237)
    test_cases.append(
        (
            "NullViewConfBuilder/generated-image-pyramid-with-parent-no-hints",
            make_entity(
                uuid="b4c5d6e7f890123456789012345678ab",
                status="Published",
                soft_assaytype="image_pyramid",
                data_types=["image_pyramid"],
                hints=["pyramid", "is_image"],  # has parent but NOT seg_mask, NOT is_support
                files=[
                    {"rel_path": "ometiff-pyramids/lab_processed/images/sample2.ome.tif"},
                    {"rel_path": "output_offsets/lab_processed/images/sample2.offsets.json"},
                ],
                parent={"uuid": "parent123456789"},  # Has parent but not the right hints
            ),
        )
    )

    return test_cases


# Generate programmatic test cases
programmatic_test_cases = (
    generate_rna_seq_test_cases()
    + generate_multiome_test_cases()
    + generate_spatial_multiome_test_cases()
    + generate_xenium_test_cases()
    + generate_spatial_rna_seq_test_cases()
    + generate_object_by_analyte_test_cases()
    + generate_tiled_sprm_test_cases()
    + generate_stitched_cytokit_sprm_test_cases()
    + generate_imaging_builder_test_cases()
    + generate_geomx_test_cases()
    + generate_multi_image_sprm_test_cases()
    + generate_legacy_json_test_cases()
    + generate_seqfish_test_cases()
    + generate_epic_seg_test_cases()
    + generate_kaggle1_seg_test_cases()
    + generate_null_builder_test_cases()
)

# Add programmatic test entities to has_visualization_test_cases
for test_id, entity in programmatic_test_cases:
    builder_name = test_id.split("/")[0]
    if builder_name == "NullViewConfBuilder":
        has_visualization_test_cases.append((False, entity))
    elif entity.get("uuid") not in excluded_uuids:
        has_visualization_test_cases.append((True, entity))


@pytest.mark.requires_full
def test_read_zip_zarr_opens_store(mocker):
    # Mock the fsspec filesystem and zarr open
    mock_fs = mocker.Mock()
    mock_mapper = mocker.Mock()
    mock_zarr_obj = mocker.Mock()

    mock_fs.get_mapper.return_value = mock_mapper

    mocker.patch("src.portal_visualization.utils.fsspec.filesystem", return_value=mock_fs)
    mocker.patch("src.portal_visualization.utils.zarr.open", return_value=mock_zarr_obj)

    dummy_url = "https://example.com/fake.zarr.zip"
    request_init = {"headers": {"Authorization": "Bearer token"}}

    result = read_zip_zarr(dummy_url, request_init)

    assert result == mock_zarr_obj
    mock_fs.get_mapper.assert_called_once_with("")


@pytest.mark.parametrize(
    ("test_id", "entity"),
    programmatic_test_cases,
    ids=lambda tc: tc[0] if isinstance(tc, tuple) else str(tc),
)
@pytest.mark.requires_full
def test_programmatic_entity_to_vitessce_conf(test_id, entity, mocker):
    """Test builder with programmatically generated entities (no JSON fixtures)."""

    # Create a mock entity_path-like object for compatibility with mock_zarr_store
    class MockEntityPath:
        def __init__(self, name, entity_data):
            self.name = name
            self.parent = type("Parent", (), {"name": test_id.split("/")[0]})()
            self._entity_data = entity_data

        def read_text(self):
            return json.dumps(self._entity_data)

    entity_path = MockEntityPath(test_id, entity)

    # Mock the zarr store
    mock_zarr_store(entity_path, mocker, 5)

    # Get parent from entity for builder selection
    parent = entity.get("parent") or None  # Only used for image pyramids

    # Get builder
    Builder = get_view_config_builder(entity, get_entity, parent)
    expected_builder = test_id.split("/")[0]
    assert Builder.__name__ == expected_builder

    # Extract marker from test_id if present (e.g., "...-marker=gene123")
    marker = None
    if "marker=" in test_id:
        # Extract marker value from test_id format like "...-marker=gene123"
        marker_part = [part for part in test_id.split("-") if part.startswith("marker=")]
        if marker_part:
            marker = marker_part[0].split("=")[1]

    # Extract minimal from test_id if present
    minimal = "minimal" in test_id

    # Build configuration - pass parent and get_entity for builders that need them
    parent_uuid = entity.get("parent", {}).get("uuid") if entity.get("parent") else None

    # Provide mock find_support_entity for Kaggle-1 builder tests
    find_support_entity = None
    if "Kaggle1" in test_id:
        find_support_entity = _mock_find_support_entity

    builder = Builder(
        entity,
        groups_token,
        assets_url,
        get_entity=get_entity,
        parent=parent_uuid,
        minimal=minimal,
        find_support_entity=find_support_entity,
    )
    conf, cells = builder.get_conf_cells(marker=marker)

    # Special case: NullViewConfBuilder returns None
    if expected_builder == "NullViewConfBuilder":
        assert conf is None
        return

    # EPIC builders now handle everything internally, no wrapper needed

    # Basic validation - should produce valid config
    assert conf is not None

    # Handle both single config (dict) and multiple configs (list)
    if isinstance(conf, list):
        # Multi-tab view (e.g., multiome builders)
        assert len(conf) > 0
        for config in conf:
            assert "datasets" in config
            assert "layout" in config
            assert len(config["datasets"]) > 0
    else:
        # Single config
        assert "datasets" in conf
        assert "layout" in conf
        assert len(conf["datasets"]) > 0


@pytest.mark.parametrize("entity_path", good_entity_paths, ids=lambda path: f"{path.parent.name}/{path.name}")
@pytest.mark.requires_full
def test_entity_to_vitessce_conf(entity_path, mocker):
    mock_zarr_store(entity_path, mocker, 5)

    possible_marker = entity_path.name.split("-")[-2]
    marker = possible_marker.split("=")[1] if possible_marker.startswith("marker=") else None
    entity = json.loads(entity_path.read_text())
    parent = entity.get("parent") or None  # Only used for image pyramids

    Builder = get_view_config_builder(entity, get_entity, parent)
    # Check if this is a minimal test case
    minimal = "minimal" in entity_path.name
    builder = Builder(entity, groups_token, assets_url, get_entity=get_entity, parent=parent, minimal=minimal)
    conf, cells = builder.get_conf_cells(marker=marker)

    # Uncomment to generate a fixture
    # print(json.dumps(conf, indent=2))

    assert Builder.__name__ == entity_path.parent.name
    compare_confs(entity_path, conf, cells)


@pytest.mark.parametrize("entity_path", bad_entity_paths, ids=lambda path: path.name)
@pytest.mark.requires_full
def test_entity_to_error(entity_path, mocker):
    mock_zarr_store(entity_path, mocker, 5)

    entity = json.loads(entity_path.read_text())

    # get_view_config_builder always uses the registry, which returns
    # NullViewConfBuilder for empty entities instead of raising an exception
    if entity == {}:
        parent = entity.get("parent") or None
        Builder = get_view_config_builder(entity, get_entity, parent=parent)
        assert Builder.__name__ == "NullViewConfBuilder"
        return

    with pytest.raises(Exception) as error_info:  # noqa: PT011, PT012
        parent = entity.get("parent") or None  # Only used for image pyramids
        Builder = get_view_config_builder(entity, get_entity, parent=parent)
        builder = Builder(entity, "groups_token", "https://example.com/")
        builder.get_conf_cells()
    actual_error = f"{error_info.type.__name__}: {error_info.value.args[0]}"

    error_expected_path = entity_path.parent / entity_path.name.replace("-entity.json", "-error.txt")
    expected_error = error_expected_path.read_text().strip()
    assert actual_error == expected_error


def clean_cells(cells):
    return [
        {k: v for k, v in dict(c).items() if k not in {"metadata", "id", "execution_count", "outputs"}} for c in cells
    ]


def compare_confs(entity_path, conf, cells):
    expected_conf_path = entity_path.parent / entity_path.name.replace("-entity", "-conf")
    expected_conf = json.loads(expected_conf_path.read_text())

    # Compare normalized JSON strings so the diff is easier to read,
    # and there are fewer false positives.
    assert json.dumps(conf, indent=2, sort_keys=True) == json.dumps(expected_conf, indent=2, sort_keys=True)

    expected_cells_path = entity_path.parent / entity_path.name.replace("-entity.json", "-cells.yaml")
    if expected_cells_path.is_file():
        expected_cells = yaml.safe_load(expected_cells_path.read_text())

        # Uncomment to generate a fixture
        # print(yaml.dump(clean_cells(cells)))

        # Compare as YAML to match fixture.
        assert yaml.dump(clean_cells(cells)) == yaml.dump(expected_cells)


@pytest.fixture
def mock_seg_image_pyramid_builder():
    from .fixtures import make_entity

    class MockBuilder(KaggleSegImagePyramidViewConfBuilder):
        def _get_file_paths(self):
            return []

    # Kaggle-2 entity: base images co-located, no parent
    entity = make_entity(
        uuid="23a25976beb8c02ab589b13a05b28c55",
        status="QA",
        soft_assaytype="h-and-e",
        data_types=["Histology"],
        hints=["segmentation_mask", "pyramid", "is_image"],
        files=[
            {"rel_path": "ometiff-pyramids/lab_processed/images/B001_SB-reg005.ome.tif"},
            {"rel_path": "output_offsets/lab_processed/images/B001_SB-reg005.offsets.json"},
            {"rel_path": "image_metadata/lab_processed/images/B001_SB-reg005.metadata.json"},
            {"rel_path": "ometiff-pyramids/B001_SB-reg005.segmentations.ome.tif"},
            {"rel_path": "output_offsets/B001_SB-reg005.segmentations.offsets.json"},
            {"rel_path": "image_metadata/B001_SB-reg005.segmentations.metadata.json"},
        ],
        immediate_ancestors=[{"data_types": ["Histology"]}],
    )
    return MockBuilder(entity, groups_token, assets_url)


@pytest.mark.requires_full
def test_filtered_images_not_found(mock_seg_image_pyramid_builder):
    mock_seg_image_pyramid_builder.seg_image_pyramid_regex = IMAGE_PYRAMID_DIR
    try:
        mock_seg_image_pyramid_builder._add_segmentation_image(None)
    except FileNotFoundError as e:
        assert str(e) == f"Dataset {mock_seg_image_pyramid_builder._uuid} is missing segmentation image pyramid files"  # noqa: PT017


@pytest.mark.requires_full
def test_filtered_images_no_regex(mock_seg_image_pyramid_builder):
    mock_seg_image_pyramid_builder.seg_image_pyramid_regex = None
    try:
        mock_seg_image_pyramid_builder._add_segmentation_image(None)
    except ValueError as e:
        assert str(e) == "seg_image_pyramid_regex is not set. Cannot find segmentation images."  # noqa: PT017


@pytest.mark.requires_full
def test_find_segmentation_images_runtime_error():
    with pytest.raises(RuntimeError) as e:  # noqa: PT012
        try:
            raise FileNotFoundError("No files found in the directory")
        except Exception as err:
            raise RuntimeError(f"Error while searching for segmentation images: {err}")  # noqa: B904

    assert "Error while searching for segmentation images:" in str(e.value)
    assert "No files found in the directory" in str(e.value)


@pytest.mark.requires_full
def test_get_found_images():
    file_paths = [
        "image_pyramid/sample.ome.tiff",
        "image_pyramid/sample_separate/sample.ome.tiff",
    ]
    regex = "image_pyramid"
    result = get_found_images(regex, file_paths)
    assert len(result) == 1
    assert result[0] == "image_pyramid/sample.ome.tiff"


@pytest.mark.requires_full
def test_get_found_images_error_handling():
    file_paths = [
        "image_pyramid/sample.ome.tiff",
        "image_pyramid/sample_separate/sample.ome.tiff",
    ]
    regex = "["  # invalid regex, forces re.error

    with pytest.raises(RuntimeError) as excinfo:  # noqa: PT012
        try:
            get_found_images(regex, file_paths)
        except Exception as e:
            raise RuntimeError(f"Error while searching for pyramid images: {e}")  # noqa: B904

    assert "Error while searching for pyramid images" in str(excinfo.value)


# Heatmap test cases use programmatic entities
heatmap_test_entities = [
    ("SpatialMultiomicAnnDataZarrViewConfBuilder", "visium", generate_spatial_multiome_test_cases()[0][1]),
    ("SpatialRNASeqAnnDataZarrViewConfBuilder", "spatial-rnaseq", generate_spatial_rna_seq_test_cases()[0][1]),
]


@pytest.mark.parametrize(
    "builder_entity",
    heatmap_test_entities,
    ids=lambda x: x[0] if isinstance(x, tuple) else str(x),
)
@pytest.mark.requires_full
def test_large_dataset_hides_heatmap(builder_entity, mocker):
    """Test that datasets with >100k observations hide heatmap views."""
    builder_name, entity_type, entity = builder_entity

    # Create mock entity path for mock_zarr_store compatibility
    class MockEntityPath:
        def __init__(self, name, entity_type, entity_data):
            self.name = f"generated-{entity_type}"  # Use entity_type for detection by helper functions
            self.parent = type("Parent", (), {"name": builder_name})()
            self._entity_data = entity_data

        def read_text(self):
            return json.dumps(self._entity_data)

    entity_path = MockEntityPath(builder_name, entity_type, entity)
    mock_zarr_store(entity_path, mocker, 150000)

    Builder = get_view_config_builder(entity, get_entity)
    builder = Builder(entity, groups_token, assets_url)
    conf, _ = builder.get_conf_cells()

    # Verify that heatmap is not in the layout
    layout_str = json.dumps(conf["layout"])
    assert "heatmap" not in layout_str.lower(), "Heatmap should not be present for large datasets"

    # Verify that other views are still present
    assert "scatterplot" in layout_str.lower() or "spatial" in layout_str.lower(), (
        "Scatterplot/spatial should still be present"
    )
    assert "cellSets" in layout_str or "obsSets" in layout_str, "Cell sets should still be present"


@pytest.mark.parametrize(
    "builder_entity",
    heatmap_test_entities,
    ids=lambda x: x[0] if isinstance(x, tuple) else str(x),
)
@pytest.mark.requires_full
def test_small_dataset_includes_heatmap(builder_entity, mocker):
    """Test that datasets with <100k observations include heatmap views."""
    builder_name, entity_type, entity = builder_entity

    # Create mock entity path for mock_zarr_store compatibility
    class MockEntityPath:
        def __init__(self, name, entity_type, entity_data):
            self.name = f"generated-{entity_type}"  # Use entity_type for detection by helper functions
            self.parent = type("Parent", (), {"name": builder_name})()
            self._entity_data = entity_data

        def read_text(self):
            return json.dumps(self._entity_data)

    entity_path = MockEntityPath(builder_name, entity_type, entity)
    mock_zarr_store(entity_path, mocker, 5000)

    Builder = get_view_config_builder(entity, get_entity)
    builder = Builder(entity, groups_token, assets_url)
    conf, _ = builder.get_conf_cells()

    # Verify that heatmap IS in the layout
    layout_str = json.dumps(conf["layout"])
    assert "heatmap" in layout_str.lower(), "Heatmap should be present for small datasets"
    assert "heatmap" in layout_str.lower(), "Heatmap should be present for small datasets"


@pytest.mark.requires_full
def test_xenium_large_dataset_hides_heatmap(mocker):
    """Test that Xenium datasets with >100k observations hide heatmap views.

    Xenium uses dual zarr stores (regular adata + spatial data), so it needs special handling.
    """
    # Use programmatic entity from generator
    builder_name, entity = generate_xenium_test_cases()[0]

    # Create mock zarr store for the regular adata zarr
    z = zarr.open_group()
    obs_count = 150000
    obs_index = [str(i) for i in range(obs_count)]

    # Xenium is multiome, so create mod/rna/obs structure
    obs = z.create_group("mod/rna/obs")
    z.create_group("mod/rna/var")  # Required for multiome structure
    obs["_index"] = zarr.array(obs_index)

    # Add required multiome groups
    group_names = ["leiden_wnn", "leiden_rna"]
    groups = obs.create_groups(*group_names)
    for group in groups:
        group["categories"] = zarr.array(["0", "1", "2"])

    # Also create regular obs group
    obs_regular = z.create_group("obs")
    obs_regular["_index"] = zarr.array(obs_index)

    # Mock zarr.open to return our mocked zarr store
    mocker.patch("zarr.open", return_value=z)

    # Mock read_zip_zarr in data_access module where ZarrStoreAccessor uses it
    mocker.patch("src.portal_visualization.data_access.read_zip_zarr", return_value=z)

    Builder = get_view_config_builder(entity, get_entity)
    builder = Builder(entity, groups_token, assets_url)
    conf, _ = builder.get_conf_cells()

    # Verify that heatmap is not in the layout
    layout_str = json.dumps(conf["layout"])
    assert "heatmap" not in layout_str.lower(), "Heatmap should not be present for large Xenium datasets"

    # Verify spatial view is still present
    assert "spatial" in layout_str.lower(), "Spatial view should still be present"


@pytest_requires_full
def test_kaggle1_builder_parent_as_dict(mocker):
    """Test that Kaggle1 builder handles parent passed as full entity dict (as in client.py)."""
    mocker.patch("src.portal_visualization.builders.imaging_builders.get_image_metadata", return_value=None)

    entity = {
        "uuid": "test-uuid",
        "vitessce-hints": ["segmentation_mask", "pyramid", "is_image"],
        "files": [
            {"rel_path": "ometiff-pyramids/seg.segmentations.ome.tif"},
            {"rel_path": "output_offsets/seg.segmentations.offsets.json"},
            {"rel_path": "image_metadata/seg.segmentations.metadata.json"},
        ],
    }

    support_entity = {
        "uuid": "support-uuid",
        "files": [
            {"rel_path": "ometiff-pyramids/lab_processed/images/base.ome.tif"},
            {"rel_path": "output_offsets/lab_processed/images/base.offsets.json"},
            {"rel_path": "image_metadata/lab_processed/images/base.metadata.json"},
        ],
    }

    # Parent passed as dict (how client.py passes it)
    parent_dict = {"uuid": "parent-uuid-123", "soft_assaytype": "PAS"}
    called_with = []

    def mock_find_support(uuid):
        called_with.append(uuid)
        return support_entity

    builder = Kaggle1SegImagePyramidViewConfBuilder(
        entity,
        groups_token="token",
        assets_endpoint="https://example.com",
        parent=parent_dict,
        find_support_entity=mock_find_support,
    )
    conf, cells = builder.get_conf_cells()
    assert conf is not None
    # Verify find_support_entity was called with the UUID string, not the dict
    assert called_with == ["parent-uuid-123"]


@pytest_requires_full
def test_kaggle1_builder_no_parent():
    """Test that Kaggle1 builder raises ValueError when parent is None."""
    entity = {
        "uuid": "test-uuid",
        "vitessce-hints": ["segmentation_mask", "pyramid", "is_image"],
        "files": [{"rel_path": "ometiff-pyramids/seg.ome.tif"}],
    }
    builder = Kaggle1SegImagePyramidViewConfBuilder(entity, groups_token="token", assets_endpoint="https://example.com")
    with pytest.raises(ValueError, match="requires a parent dataset"):
        builder.get_conf_cells()


@pytest_requires_full
def test_kaggle1_builder_metadata_files_fallback(mocker):
    """Test that Kaggle1 builder falls back to metadata.files when files is missing."""
    mocker.patch("src.portal_visualization.builders.imaging_builders.get_image_metadata", return_value=None)

    entity = {
        "uuid": "test-uuid",
        "vitessce-hints": ["segmentation_mask", "pyramid", "is_image"],
        "files": [
            {"rel_path": "ometiff-pyramids/seg.segmentations.ome.tif"},
            {"rel_path": "output_offsets/seg.segmentations.offsets.json"},
            {"rel_path": "image_metadata/seg.segmentations.metadata.json"},
        ],
    }

    # Support entity with files in metadata (not top-level)
    support_entity = {
        "uuid": "support-meta-uuid",
        "metadata": {
            "files": [
                {"rel_path": "ometiff-pyramids/lab_processed/images/base.ome.tif"},
                {"rel_path": "output_offsets/lab_processed/images/base.offsets.json"},
                {"rel_path": "image_metadata/lab_processed/images/base.metadata.json"},
            ]
        },
    }

    builder = Kaggle1SegImagePyramidViewConfBuilder(
        entity,
        groups_token="token",
        assets_endpoint="https://example.com",
        parent="parent-uuid",
        find_support_entity=lambda uuid: support_entity,
    )
    conf, cells = builder.get_conf_cells()
    assert conf is not None
    assert "datasets" in conf


@pytest_requires_full
def test_kaggle1_builder_no_images_in_support():
    """Test that Kaggle1 builder raises FileNotFoundError when support entity has no images."""
    entity = {
        "uuid": "test-uuid",
        "vitessce-hints": ["segmentation_mask", "pyramid", "is_image"],
        "files": [{"rel_path": "ometiff-pyramids/seg.ome.tif"}],
    }

    # Support entity with no ome.tif files
    support_entity = {
        "uuid": "support-empty-uuid",
        "files": [{"rel_path": "some/other/file.json"}],
    }

    builder = Kaggle1SegImagePyramidViewConfBuilder(
        entity,
        groups_token="token",
        assets_endpoint="https://example.com",
        parent="parent-uuid",
        find_support_entity=lambda uuid: support_entity,
    )
    with pytest.raises(FileNotFoundError, match="missing base image pyramid files"):
        builder.get_conf_cells()


@pytest_requires_full
def test_kaggle1_builder_no_support_entity():
    """Test that Kaggle1 builder raises ValueError when find_support_entity returns None."""
    entity = {
        "uuid": "test-uuid",
        "vitessce-hints": ["segmentation_mask", "pyramid", "is_image"],
        "files": [{"rel_path": "ometiff-pyramids/seg.ome.tif"}],
    }

    builder = Kaggle1SegImagePyramidViewConfBuilder(
        entity,
        groups_token="token",
        assets_endpoint="https://example.com",
        parent="parent-uuid",
        find_support_entity=lambda uuid: None,
    )
    with pytest.raises(ValueError, match="could not find support entity"):
        builder.get_conf_cells()


@pytest_requires_full
def test_kaggle1_builder_no_token(mocker):
    """Test Kaggle1 builder URL generation without auth token."""
    mocker.patch("src.portal_visualization.builders.imaging_builders.get_image_metadata", return_value=None)

    entity = {
        "uuid": "test-uuid",
        "vitessce-hints": ["segmentation_mask", "pyramid", "is_image"],
        "files": [
            {"rel_path": "ometiff-pyramids/seg.segmentations.ome.tif"},
            {"rel_path": "output_offsets/seg.segmentations.offsets.json"},
            {"rel_path": "image_metadata/seg.segmentations.metadata.json"},
        ],
    }

    support_entity = {
        "uuid": "support-uuid",
        "files": [
            {"rel_path": "ometiff-pyramids/lab_processed/images/base.ome.tif"},
            {"rel_path": "output_offsets/lab_processed/images/base.offsets.json"},
            {"rel_path": "image_metadata/lab_processed/images/base.metadata.json"},
        ],
    }

    builder = Kaggle1SegImagePyramidViewConfBuilder(
        entity,
        groups_token=None,
        assets_endpoint="https://example.com",
        parent="parent-uuid",
        find_support_entity=lambda uuid: support_entity,
    )
    conf, cells = builder.get_conf_cells()
    assert conf is not None
    # Verify URLs don't have token parameter
    datasets = conf.get("datasets", [])
    assert len(datasets) > 0


@pytest_requires_full
def test_kaggle1_builder_base_image_source_support_entity(mocker):
    """Test that base_image_source is 'support_entity' when base images come from parent's support."""
    mocker.patch("src.portal_visualization.builders.imaging_builders.get_image_metadata", return_value=None)

    entity = {
        "uuid": "test-uuid",
        "vitessce-hints": ["segmentation_mask", "pyramid", "is_image"],
        "files": [
            {"rel_path": "ometiff-pyramids/seg.segmentations.ome.tif"},
            {"rel_path": "output_offsets/seg.segmentations.offsets.json"},
            {"rel_path": "image_metadata/seg.segmentations.metadata.json"},
        ],
    }

    support_entity = {
        "uuid": "support-uuid",
        "files": [
            {"rel_path": "ometiff-pyramids/lab_processed/images/base.ome.tif"},
            {"rel_path": "output_offsets/lab_processed/images/base.offsets.json"},
            {"rel_path": "image_metadata/lab_processed/images/base.metadata.json"},
        ],
    }

    builder = Kaggle1SegImagePyramidViewConfBuilder(
        entity,
        groups_token="token",
        assets_endpoint="https://example.com",
        parent="parent-uuid",
        find_support_entity=lambda uuid: support_entity,
    )
    assert builder.base_image_source is None
    builder.get_conf_cells()
    assert builder.base_image_source == "support_entity"


@pytest_requires_full
def test_kaggle1_builder_base_image_source_colocated(mocker):
    """Test that base_image_source is 'colocated' when base images are in entity's own files."""
    mocker.patch("src.portal_visualization.builders.imaging_builders.get_image_metadata", return_value=None)

    entity = {
        "uuid": "test-uuid",
        "vitessce-hints": ["segmentation_mask", "pyramid", "is_image"],
        "files": [
            {"rel_path": "ometiff-pyramids/lab_processed/images/base.ome.tif"},
            {"rel_path": "output_offsets/lab_processed/images/base.offsets.json"},
            {"rel_path": "image_metadata/lab_processed/images/base.metadata.json"},
            {"rel_path": "ometiff-pyramids/seg.segmentations.ome.tif"},
            {"rel_path": "output_offsets/seg.segmentations.offsets.json"},
            {"rel_path": "image_metadata/seg.segmentations.metadata.json"},
        ],
    }

    builder = Kaggle1SegImagePyramidViewConfBuilder(
        entity,
        groups_token="token",
        assets_endpoint="https://example.com",
        parent="parent-uuid",
        find_support_entity=lambda uuid: None,
    )
    assert builder.base_image_source is None
    builder.get_conf_cells()
    assert builder.base_image_source == "colocated"


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="Generate fixtures")
    parser.add_argument("--input", required=True, type=Path, help="Input JSON path")

    args = parser.parse_args()
    entity = json.loads(args.input.read_text())
    Builder = get_view_config_builder(entity, get_entity)
    builder = Builder(entity, "groups_token", "https://example.com/")
    conf, cells = builder.get_conf_cells()

    print(yaml.dump(clean_cells(cells), default_style="|"))
