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
    with pytest.raises(ValueError, match="SegmentationMaskBuilder requires a parent"):
        builder = SegmentationMaskBuilder(
            entity, groups_token="token", assets_endpoint="https://example.com", get_entity=None, parent=None
        )
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
