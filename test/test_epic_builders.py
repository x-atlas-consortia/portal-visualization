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


def test_segmentation_mask_builder_requires_parent():
    """Test that SegmentationMaskBuilder requires a parent entity."""
    entity = {
        "uuid": "test-uuid",
        "vitessce-hints": ["segmentation_mask", "is_support"],
        "files": [],
    }

    # Should fail without parent
    builder = SegmentationMaskBuilder(
        entity, groups_token="token", assets_endpoint="https://example.com", get_entity=None, parent=None
    )
    with pytest.raises(ValueError, match="SegmentationMaskBuilder requires a parent"):
        builder.get_conf_cells()


def test_segmentation_mask_builder_requires_get_entity():
    """Test that SegmentationMaskBuilder requires get_entity callback."""
    entity = {
        "uuid": "test-uuid",
        "vitessce-hints": ["segmentation_mask", "is_support"],
        "files": [],
    }

    # Should fail without get_entity callback
    builder = SegmentationMaskBuilder(
        entity, groups_token="token", assets_endpoint="https://example.com", get_entity=None, parent="parent-uuid"
    )
    with pytest.raises(ValueError, match="SegmentationMaskBuilder requires get_entity callback"):
        builder.get_conf_cells()


def test_segmentation_mask_builder_with_parent(mocker):
    """Test that SegmentationMaskBuilder works with a parent entity."""
    # Mock parent entity
    parent_entity = {
        "uuid": "parent-uuid",
        "vitessce-hints": ["is_image"],
        "soft_assaytype": "CODEX",
        "files": [
            {"rel_path": "stitched/expressions/expr_0.ome.tiff"},
            {"rel_path": "stitched/ome-tiff-offsets/expr_0_offsets.json"},
        ],
    }

    # Mock segmentation mask entity
    entity = {
        "uuid": "seg-mask-uuid",
        "vitessce-hints": ["segmentation_mask", "is_support"],
        "files": [
            {"rel_path": "segmentation_masks_Probabilities_0.ome.tiff"},
        ],
    }

    def get_entity(uuid):
        if uuid == "parent-uuid":
            return parent_entity
        return None

    # Mock file reading for metadata
    mocker.patch("src.portal_visualization.builders.epic_builders.get_image_metadata", return_value={})

    builder = SegmentationMaskBuilder(
        entity, groups_token="token", assets_endpoint="https://example.com", get_entity=get_entity, parent="parent-uuid"
    )

    # Just verify it doesn't crash - full config validation is done in test_builders.py
    assert builder is not None
    assert builder._entity["uuid"] == "seg-mask-uuid"


def test_segmentation_mask_builder_metadata_fallback():
    """Test that SegmentationMaskBuilder handles missing metadata files gracefully."""
    # Parent entity without metadata files
    parent_entity = {
        "uuid": "parent-uuid",
        "vitessce-hints": ["is_image"],
        "soft_assaytype": "CODEX",
        "files": [
            {"rel_path": "stitched/expressions/expr_0.ome.tiff"},
            {"rel_path": "stitched/ome-tiff-offsets/expr_0_offsets.json"},
        ],
    }

    entity = {
        "uuid": "seg-mask-uuid",
        "vitessce-hints": ["segmentation_mask", "is_support"],
        "files": [
            {"rel_path": "extras/transformations/ometiff-pyramids/91706.segmentations.ome.tif"},
            {"rel_path": "extras/transformations/output_offsets/91706.segmentations.offsets.json"},
        ],
    }

    def get_entity(uuid):
        if uuid == "parent-uuid":
            return parent_entity
        return None

    builder = SegmentationMaskBuilder(
        entity, groups_token="token", assets_endpoint="https://example.com", get_entity=get_entity, parent="parent-uuid"
    )

    # Call internal method to test metadata fallback (returns empty dict when file not found)
    metadata = builder._get_base_image_metadata("91706.segmentations", parent_entity)
    assert metadata == {}
