"""Tests for experimental builder registry feature.

These tests cover the registry code paths when USE_BUILDER_REGISTRY=1.
The registry is experimental and not used in production.
"""

from unittest.mock import patch

import pytest


@pytest.mark.requires_full
class TestRegistryExperimental:
    """Test experimental registry-based builder selection."""

    def test_registry_enabled_with_matching_builder(self):
        """Test registry path when USE_BUILDER_REGISTRY=1 with a match."""
        from portal_visualization.builder_factory import _get_builder_name

        entity = {
            "uuid": "test-uuid",
            "soft_assaytype": "CODEX",
            "vitessce-hints": ["is_image", "codex"],
        }

        # Enable registry
        with patch("portal_visualization.builder_factory.USE_BUILDER_REGISTRY", 1):
            # Line 183: Registry call when enabled
            builder_name = _get_builder_name(entity, lambda x: entity, None, None)

            # Should get a builder from registry
            assert builder_name is not None
            assert builder_name != "NullViewConfBuilder"

    def test_registry_enabled_with_no_match(self):
        """Test registry fallback to NullViewConfBuilder when no match."""
        from portal_visualization.builder_factory import _get_builder_name_from_registry

        # Create entity with hints that won't match any registered builder
        entity = {
            "uuid": "test-uuid",
            "soft_assaytype": "completely_unknown_assay_type_that_will_never_exist",
            "vitessce-hints": ["totally_unknown_hint_12345", "another_nonexistent_hint"],
        }

        # Line 312: Should fallback to NullViewConfBuilder when no match found
        builder_name = _get_builder_name_from_registry(entity, lambda x: entity, None, None)
        assert builder_name == "NullViewConfBuilder"

    def test_registration_matches_edge_cases(self):
        """Test BuilderRegistration.matches with various conditions."""
        from portal_visualization.builder_registry import BuilderRegistration

        # Test with forbidden hints
        reg = BuilderRegistration(
            builder_name="TestBuilder",
            required_hints={"is_image"},
            forbidden_hints={"exclude_me"},
            priority=5,
        )

        # Line 74: Return False when forbidden hint present (line 64 check)
        assert reg.matches(["is_image", "exclude_me"], None, False, False) is False

        # Test with assay type requirement
        reg = BuilderRegistration(
            builder_name="TestBuilder",
            required_hints={"is_image"},
            assay_types={"CODEX"},
            priority=5,
        )

        # Line 70: Return False when assay_type doesn't match
        assert reg.matches(["is_image"], "wrong_assay", False, False) is False
        # Should match when correct
        assert reg.matches(["is_image"], "CODEX", False, False) is True

        # Test requires_epic check (line 76)
        reg = BuilderRegistration(
            builder_name="TestBuilder",
            required_hints={"is_image"},
            requires_epic=True,
            priority=5,
        )
        # Should not match when no epic provided
        assert reg.matches(["is_image"], None, False, False) is False
        # Should match when epic provided
        assert reg.matches(["is_image"], None, False, True) is True

        # Test requires_parent check (line 74)
        reg = BuilderRegistration(
            builder_name="TestBuilder",
            required_hints={"is_image"},
            requires_parent=True,
            priority=5,
        )
        # Line 74: Return False when parent required but not provided
        assert reg.matches(["is_image"], None, False, False, None) is False
        # Should match when parent provided
        assert reg.matches(["is_image"], None, True, False, None) is True

        # Test parent_assay_types check (line 84)
        reg = BuilderRegistration(
            builder_name="TestBuilder",
            required_hints={"is_image"},
            parent_assay_types={"SEQFISH"},
            priority=5,
        )
        # Line 84: Return False when parent_assay_type doesn't match
        assert reg.matches(["is_image"], None, True, False, "wrong_type") is False
        # Should match when correct parent assay type
        assert reg.matches(["is_image"], None, True, False, "SEQFISH") is True

    def test_find_builder_returns_none_when_no_match(self):
        """Test BuilderRegistry.find_builder returns None when no matches."""
        from portal_visualization.builder_registry import BuilderRegistry

        registry = BuilderRegistry()
        registry.register("OnlyForCodex", required_hints=["is_image"], assay_types={"CODEX"})

        # Line 159: Return None when no matches
        result = registry.find_builder(["is_image"], "DIFFERENT_ASSAY", False, False, None)
        assert result is None

    def test_registry_parent_assay_type_fetching(self):
        """Test that _get_builder_name_from_registry fetches parent assay type."""
        from portal_visualization.assays import SEQFISH
        from portal_visualization.builder_factory import _get_builder_name_from_registry

        # Create parent entity with SEQFISH assay type
        parent_entity = {
            "uuid": "parent-uuid",
            "soft_assaytype": SEQFISH,
        }

        # Create support entity (image pyramid)
        support_entity = {
            "uuid": "support-uuid",
            "vitessce-hints": ["is_support", "is_image"],
        }

        # Mock get_entity to return parent
        def mock_get_entity(uuid):
            if uuid == "parent-uuid":
                return parent_entity
            return support_entity

        # Lines 311-312: Should fetch parent entity and extract soft_assaytype
        builder_name = _get_builder_name_from_registry(
            support_entity,
            mock_get_entity,
            "parent-uuid",  # parent UUID
            None,
        )

        # Should match SeqFISHViewConfBuilder based on parent assay type
        assert builder_name == "SeqFISHViewConfBuilder"
