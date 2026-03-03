import pytest

try:
    from src.portal_visualization.builders.epic_builders import SegmentationMaskBuilder

    FULL_DEPS_AVAILABLE = True
except ImportError:
    FULL_DEPS_AVAILABLE = False
    # Skip entire module during collection if full dependencies not available
    pytest.skip("requires [full] optional dependencies", allow_module_level=True)

# Mark all tests in this file as requiring [full] dependencies
pytestmark = pytest.mark.requires_full


def test_segmentation_mask_builder_with_colocated_files(mocker):
    """Test that SegmentationMaskBuilder works with base images and seg masks in the same entity."""
    entity = {
        "uuid": "seg-mask-uuid",
        "status": "QA",
        "vitessce-hints": ["segmentation_mask", "is_image", "pyramid", "epic"],
        "files": [
            {"rel_path": "extras/transformations/ometiff-pyramids/lab_processed/images/91706.ome.tif"},
            {"rel_path": "extras/transformations/output_offsets/lab_processed/images/91706.offsets.json"},
            {"rel_path": "extras/transformations/image_metadata/lab_processed/images/91706.metadata.json"},
            {"rel_path": "extras/transformations/ometiff-pyramids/91706.segmentations.ome.tif"},
            {"rel_path": "extras/transformations/output_offsets/91706.segmentations.offsets.json"},
            {"rel_path": "extras/transformations/image_metadata/91706.segmentations.metadata.json"},
        ],
    }

    mocker.patch("src.portal_visualization.builders.epic_builders.get_image_metadata", return_value={})

    builder = SegmentationMaskBuilder(entity, groups_token="token", assets_endpoint="https://example.com")

    assert builder is not None
    assert builder._entity["uuid"] == "seg-mask-uuid"


def test_segmentation_mask_builder_metadata_fallback(mocker):
    """Test that SegmentationMaskBuilder handles missing metadata files gracefully."""
    entity = {
        "uuid": "seg-mask-uuid",
        "status": "QA",
        "vitessce-hints": ["segmentation_mask", "is_image", "pyramid", "epic"],
        "files": [
            {"rel_path": "extras/transformations/ometiff-pyramids/lab_processed/images/91706.ome.tif"},
            {"rel_path": "extras/transformations/output_offsets/lab_processed/images/91706.offsets.json"},
            # No metadata file present
            {"rel_path": "extras/transformations/ometiff-pyramids/91706.segmentations.ome.tif"},
            {"rel_path": "extras/transformations/output_offsets/91706.segmentations.offsets.json"},
        ],
    }

    mocker.patch("src.portal_visualization.builders.epic_builders.get_image_metadata", return_value=None)

    builder = SegmentationMaskBuilder(entity, groups_token="token", assets_endpoint="https://example.com")

    # Call internal method to test metadata fallback (returns None when metadata unavailable)
    metadata = builder._get_base_image_metadata()
    assert metadata is None
