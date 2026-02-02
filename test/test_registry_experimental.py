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
            builder_name = _get_builder_name(entity, lambda x: entity, None)

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
        builder_name = _get_builder_name_from_registry(entity, lambda x: entity, None)
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
        )

        # Should match SeqFISHViewConfBuilder based on parent assay type
        assert builder_name == "SeqFISHViewConfBuilder"

    def test_registry_parent_fetch_error_handling(self):
        """Test that _get_builder_name_from_registry handles missing parent gracefully."""
        from portal_visualization.builder_factory import _get_builder_name_from_registry

        # Create support entity (image pyramid)
        support_entity = {
            "uuid": "support-uuid",
            "vitessce-hints": ["is_support", "is_image"],
        }

        # Mock get_entity that raises FileNotFoundError
        def mock_get_entity_fail(uuid):
            raise FileNotFoundError(f"No such file: {uuid}")

        # Lines 314-316: Should handle FileNotFoundError gracefully
        builder_name = _get_builder_name_from_registry(
            support_entity,
            mock_get_entity_fail,
            "nonexistent-parent",  # parent UUID that doesn't exist
        )

        # Should fallback to generic ImagePyramidViewConfBuilder
        assert builder_name == "ImagePyramidViewConfBuilder"

    def test_get_match_diagnostics_forbidden_hints(self):
        """Test get_match_diagnostics with forbidden hints."""
        from portal_visualization.builder_registry import BuilderRegistry

        registry = BuilderRegistry()
        registry.register("NoJSON", required_hints=["is_image"], forbidden_hints=["json_based"], priority=10)

        # Line 242: Test forbidden hints check
        diagnostics = registry.get_match_diagnostics(["is_image", "json_based"], None)

        assert diagnostics["selected"] is None
        assert len(diagnostics["non_matching_reasons"]) > 0
        reason = next(r for r in diagnostics["non_matching_reasons"] if r["builder"] == "NoJSON")
        assert any("forbidden hints" in r for r in reason["reasons"])

    def test_get_match_diagnostics_assay_type_mismatch(self):
        """Test get_match_diagnostics with assay type mismatch."""
        from portal_visualization.builder_registry import BuilderRegistry

        registry = BuilderRegistry()
        registry.register("CodexOnly", required_hints=["is_image"], assay_types=["CODEX"], priority=10)

        # Line 246: Test assay_type not in assay_types
        diagnostics = registry.get_match_diagnostics(["is_image"], "DIFFERENT")

        assert diagnostics["selected"] is None
        reason = next(r for r in diagnostics["non_matching_reasons"] if r["builder"] == "CodexOnly")
        assert any("assay_type" in r for r in reason["reasons"])

    def test_get_match_diagnostics_parent_assay_type_mismatch(self):
        """Test get_match_diagnostics with parent assay type mismatch."""
        from portal_visualization.builder_registry import BuilderRegistry

        registry = BuilderRegistry()
        registry.register("SeqFISHOnly", required_hints=["is_image"], parent_assay_types=["SEQFISH"], priority=10)

        # Line 250: Test parent_assay_type not in parent_assay_types
        diagnostics = registry.get_match_diagnostics(["is_image"], None, has_parent=True, parent_assay_type="WRONG")

        assert diagnostics["selected"] is None
        reason = next(r for r in diagnostics["non_matching_reasons"] if r["builder"] == "SeqFISHOnly")
        assert any("parent_assay_type" in r for r in reason["reasons"])

    def test_get_match_diagnostics_requires_parent(self):
        """Test get_match_diagnostics when parent is required but missing."""
        from portal_visualization.builder_registry import BuilderRegistry

        registry = BuilderRegistry()
        registry.register("NeedsParent", required_hints=["is_image"], requires_parent=True, priority=10)

        # Line 254: Test requires_parent but has_parent=False
        diagnostics = registry.get_match_diagnostics(["is_image"], None, has_parent=False)

        assert diagnostics["selected"] is None
        reason = next(r for r in diagnostics["non_matching_reasons"] if r["builder"] == "NeedsParent")
        assert any("requires parent" in r for r in reason["reasons"])

    def test_get_match_diagnostics_requires_epic(self):
        """Test get_match_diagnostics when EPIC is required but missing."""
        from portal_visualization.builder_registry import BuilderRegistry

        registry = BuilderRegistry()
        registry.register("NeedsEpic", required_hints=["is_support"], requires_epic=True, priority=10)

        # Line 258: Test requires_epic but has_epic=False
        diagnostics = registry.get_match_diagnostics(["is_support"], None, has_epic=False)

        assert diagnostics["selected"] is None
        reason = next(r for r in diagnostics["non_matching_reasons"] if r["builder"] == "NeedsEpic")
        assert any("EPIC UUID" in r for r in reason["reasons"])

    def test_format_no_match_message_with_match(self):
        """Test format_no_match_message when a builder matches."""
        from portal_visualization.builder_registry import BuilderRegistry

        registry = BuilderRegistry()
        registry.register("TestBuilder", required_hints=["is_image"], priority=10)

        # Line 313: Test when diagnostics["selected"] is truthy
        message = registry.format_no_match_message(["is_image"], None)

        assert "Selected builder: TestBuilder" in message
        assert "priority=10" in message

    def test_format_no_match_message_with_parent_assay_type(self):
        """Test format_no_match_message includes parent_assay_type when provided."""
        from portal_visualization.builder_registry import BuilderRegistry

        registry = BuilderRegistry()
        # Don't register any builders to ensure no match

        # Line 324: Test has_parent and parent_assay_type branch
        message = registry.format_no_match_message(["is_image"], None, has_parent=True, parent_assay_type="SEQFISH")

        assert "Parent assay type: SEQFISH" in message

    def test_format_no_match_message_with_epic(self):
        """Test format_no_match_message includes EPIC info when provided."""
        from portal_visualization.builder_registry import BuilderRegistry

        registry = BuilderRegistry()
        # Don't register any builders to ensure no match

        # Line 326: Test has_epic branch
        message = registry.format_no_match_message(["is_support"], None, has_epic=True)

        assert "Has EPIC UUID: True" in message
