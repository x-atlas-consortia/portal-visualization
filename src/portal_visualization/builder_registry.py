"""
Builder registration system for portal-visualization.

This module provides a decorator-based registry for builders, allowing them to
declare which hint combinations they support. This replaces the monolithic
conditional logic in builder_factory.py with a more maintainable, discoverable pattern.
"""

from dataclasses import dataclass, field


@dataclass
class BuilderRegistration:
    """Metadata about a registered builder.

    Attributes:
        builder_name: Fully qualified name of the builder class (e.g., 'RNASeqAnnDataZarrViewConfBuilder')
        required_hints: Hints that must be present (e.g., ['is_image', 'rna'])
        forbidden_hints: Hints that must NOT be present (e.g., ['is_json'])
        assay_types: Specific assay types this builder handles (None = any)
        parent_assay_types: Parent assay types this builder handles (None = any)
        priority: Selection priority when multiple builders match (higher = preferred)
        requires_parent: Whether this builder requires a parent entity
        requires_epic: Whether this builder requires an EPIC UUID
    """

    builder_name: str
    required_hints: set[str] = field(default_factory=set)
    forbidden_hints: set[str] = field(default_factory=set)
    assay_types: set[str] | None = None
    parent_assay_types: set[str] | None = None
    priority: int = 0
    requires_parent: bool = False
    requires_epic: bool = False

    def matches(
        self,
        hints: list[str],
        assay_type: str | None,
        has_parent: bool,
        has_epic: bool,
        parent_assay_type: str | None = None,
    ) -> bool:
        """Check if this builder matches the given criteria.

        Args:
            hints: List of vitessce-hints from entity
            assay_type: Assay type from entity
            has_parent: Whether a parent UUID was provided
            has_epic: Whether an EPIC UUID was provided
            parent_assay_type: Assay type of the parent entity (if has_parent)

        Returns:
            True if this builder should be used

        >>> reg = BuilderRegistration(
        ...     builder_name="TestBuilder",
        ...     required_hints={"is_image", "rna"},
        ...     forbidden_hints={"is_json"}
        ... )
        >>> reg.matches(["is_image", "rna", "spatial"], None, False, False)
        True
        >>> reg.matches(["is_image", "rna", "is_json"], None, False, False)
        False
        >>> reg.matches(["is_image"], None, False, False)
        False
        """
        hint_set = set(hints)

        # Check required hints are present
        if not self.required_hints.issubset(hint_set):
            return False

        # Check forbidden hints are absent
        if self.forbidden_hints & hint_set:
            return False

        # Check assay type if specified
        if self.assay_types is not None and assay_type not in self.assay_types:
            return False

        # Check parent assay type if specified
        if self.parent_assay_types is not None and parent_assay_type not in self.parent_assay_types:
            return False

        # Check parent/epic requirements
        if self.requires_parent and not has_parent:
            return False

        return not (self.requires_epic and not has_epic)


class BuilderRegistry:
    """Registry for builder classes.

    Maintains a mapping of hint combinations to builder classes, allowing
    dynamic lookup without monolithic conditional logic.

    >>> registry = BuilderRegistry()
    >>> registry.register(
    ...     builder_name="TestBuilder",
    ...     required_hints=["is_image"],
    ...     priority=10
    ... )
    >>> match = registry.find_builder(["is_image", "rna"], None, False, False)
    >>> match
    'TestBuilder'
    """

    def __init__(self):
        self._registrations: list[BuilderRegistration] = []

    def register(
        self,
        builder_name: str,
        required_hints: list[str] | None = None,
        forbidden_hints: list[str] | None = None,
        assay_types: list[str] | None = None,
        parent_assay_types: list[str] | None = None,
        priority: int = 0,
        requires_parent: bool = False,
        requires_epic: bool = False,
    ) -> None:
        """Register a builder.

        Args:
            builder_name: Name of the builder class
            required_hints: Hints that must be present
            forbidden_hints: Hints that must NOT be present
            assay_types: Specific assay types this builder handles
            parent_assay_types: Parent assay types this builder handles
            priority: Selection priority (higher wins)
            requires_parent: Whether builder needs parent entity
            requires_epic: Whether builder needs EPIC UUID

        >>> registry = BuilderRegistry()
        >>> registry.register("MyBuilder", required_hints=["is_image"], priority=5)
        >>> len(registry._registrations)
        1
        """
        registration = BuilderRegistration(
            builder_name=builder_name,
            required_hints=set(required_hints or []),
            forbidden_hints=set(forbidden_hints or []),
            assay_types=set(assay_types) if assay_types else None,
            parent_assay_types=set(parent_assay_types) if parent_assay_types else None,
            priority=priority,
            requires_parent=requires_parent,
            requires_epic=requires_epic,
        )
        self._registrations.append(registration)

    def find_builder(
        self,
        hints: list[str],
        assay_type: str | None,
        has_parent: bool = False,
        has_epic: bool = False,
        parent_assay_type: str | None = None,
    ) -> str | None:
        """Find the best matching builder for the given criteria.

        Args:
            hints: List of vitessce-hints
            assay_type: Assay type string
            has_parent: Whether parent UUID is available
            has_epic: Whether EPIC UUID is available
            parent_assay_type: Assay type of parent entity (if has_parent)

        Returns:
            Builder name or None if no match

        >>> registry = BuilderRegistry()
        >>> registry.register("LowPriority", required_hints=["is_image"], priority=1)
        >>> registry.register("HighPriority", required_hints=["is_image"], priority=10)
        >>> registry.find_builder(["is_image"], None)
        'HighPriority'
        """
        matching = [
            reg
            for reg in self._registrations
            if reg.matches(hints, assay_type, has_parent, has_epic, parent_assay_type)
        ]

        if not matching:
            return None

        # Return highest priority match
        best = max(matching, key=lambda r: r.priority)
        return best.builder_name

    def get_match_diagnostics(
        self,
        hints: list[str],
        assay_type: str | None,
        has_parent: bool = False,
        has_epic: bool = False,
        parent_assay_type: str | None = None,
    ) -> dict:
        """Get detailed diagnostic information about builder matching.

        This is useful for debugging why a particular builder was (or wasn't) selected.

        Args:
            hints: List of vitessce-hints
            assay_type: Assay type string
            has_parent: Whether parent UUID is available
            has_epic: Whether EPIC UUID is available
            parent_assay_type: Assay type of parent entity (if has_parent)

        Returns:
            Dictionary with diagnostic information including:
            - selected: Name of selected builder (or None)
            - matching_count: Number of builders that matched
            - matching_builders: List of matching builder names with priorities
            - non_matching_reasons: Why builders didn't match

        >>> registry = BuilderRegistry()
        >>> registry.register("TestBuilder", required_hints=["is_image"], priority=10)
        >>> diag = registry.get_match_diagnostics(["is_image"], None)
        >>> diag['selected']
        'TestBuilder'
        >>> diag['matching_count']
        1
        """
        hint_set = set(hints)
        matching = []
        non_matching_reasons = []

        for reg in self._registrations:
            # Check each constraint and track why it didn't match
            reasons = []

            # Check required hints
            missing_hints = reg.required_hints - hint_set
            if missing_hints:
                reasons.append(f"missing required hints: {sorted(missing_hints)}")

            # Check forbidden hints
            forbidden_present = reg.forbidden_hints & hint_set
            if forbidden_present:
                reasons.append(f"has forbidden hints: {sorted(forbidden_present)}")

            # Check assay type
            if reg.assay_types is not None and assay_type not in reg.assay_types:
                reasons.append(f"assay_type '{assay_type}' not in {sorted(reg.assay_types)}")

            # Check parent assay type
            if reg.parent_assay_types is not None and parent_assay_type not in reg.parent_assay_types:
                reasons.append(f"parent_assay_type '{parent_assay_type}' not in {sorted(reg.parent_assay_types)}")

            # Check parent requirement
            if reg.requires_parent and not has_parent:
                reasons.append("requires parent but none provided")

            # Check epic requirement
            if reg.requires_epic and not has_epic:
                reasons.append("requires EPIC UUID but none provided")

            if reasons:
                non_matching_reasons.append({"builder": reg.builder_name, "reasons": reasons})
            else:
                matching.append({"builder": reg.builder_name, "priority": reg.priority})

        # Sort matching by priority (highest first)
        matching.sort(key=lambda x: x["priority"], reverse=True)

        selected = matching[0]["builder"] if matching else None

        return {
            "selected": selected,
            "matching_count": len(matching),
            "matching_builders": matching,
            "non_matching_reasons": non_matching_reasons,
            "search_criteria": {
                "hints": sorted(hints),
                "assay_type": assay_type,
                "has_parent": has_parent,
                "has_epic": has_epic,
                "parent_assay_type": parent_assay_type,
            },
        }

    def format_no_match_message(
        self,
        hints: list[str],
        assay_type: str | None,
        has_parent: bool = False,
        has_epic: bool = False,
        parent_assay_type: str | None = None,
    ) -> str:
        """Generate a detailed error message when no builder matches.

        Args:
            hints: List of vitessce-hints
            assay_type: Assay type string
            has_parent: Whether parent UUID is available
            has_epic: Whether EPIC UUID is available
            parent_assay_type: Assay type of parent entity (if has_parent)

        Returns:
            Formatted error message with diagnostic information

        >>> registry = BuilderRegistry()
        >>> registry.register("TestBuilder", required_hints=["is_image"], priority=10)
        >>> msg = registry.format_no_match_message(["rna"], None)
        >>> "No builder found" in msg
        True
        """
        diagnostics = self.get_match_diagnostics(hints, assay_type, has_parent, has_epic, parent_assay_type)

        if diagnostics["selected"]:
            return f"Selected builder: {diagnostics['selected']} (priority={diagnostics['matching_builders'][0]['priority']})"

        # No match - build detailed error message
        lines = [
            "No builder found matching the following criteria:",
            f"  Hints: {sorted(hints) if hints else '(none)'}",
            f"  Assay type: {assay_type or '(none)'}",
            f"  Has parent: {has_parent}",
        ]

        if has_parent and parent_assay_type:
            lines.append(f"  Parent assay type: {parent_assay_type}")
        if has_epic:
            lines.append(f"  Has EPIC UUID: {has_epic}")

        lines.append("")
        lines.append(f"Evaluated {len(self._registrations)} builders:")

        # Group reasons by why they didn't match
        if diagnostics["non_matching_reasons"]:
            # Show top 5 closest matches (fewest reasons)
            sorted_reasons = sorted(diagnostics["non_matching_reasons"], key=lambda x: len(x["reasons"]))
            lines.append("")
            lines.append("Closest matches (failed due to):")
            for item in sorted_reasons[:5]:
                lines.append(f"  - {item['builder']}:")
                for reason in item["reasons"]:
                    lines.append(f"      {reason}")

        return "\n".join(lines)

    def list_builders(self) -> list[str]:
        """List all registered builder names.

        Returns:
            List of builder names

        >>> registry = BuilderRegistry()
        >>> registry.register("Builder1")
        >>> registry.register("Builder2")
        >>> sorted(registry.list_builders())
        ['Builder1', 'Builder2']
        """
        return [reg.builder_name for reg in self._registrations]


# Global registry instance
_REGISTRY = BuilderRegistry()


def get_registry() -> BuilderRegistry:
    """Get the global builder registry.

    Returns:
        Global BuilderRegistry instance

    >>> registry = get_registry()
    >>> isinstance(registry, BuilderRegistry)
    True
    """
    return _REGISTRY


def populate_registry():
    """Populate the registry with all builder mappings.

    This function replicates the logic from builder_factory._get_builder_name()
    as registry entries, maintaining backward compatibility.

    The declarative configuration list below provides:
    - Self-documenting builder criteria
    - Easy modification and additions
    - Clear priority ordering
    - Maintains string-based registration (no imports)

    Example usage:
        populate_registry()
        # Registry now contains all builder mappings
    """
    # Import assay constants that actually exist
    from .assays import MALDI_IMS, NANODESI, SALMON_RNASSEQ_SLIDE, SEQFISH

    # Priority levels (higher = more specific match gets selected)
    PRIORITY_SPECIFIC = 100  # Very specific combinations
    PRIORITY_MODERATE = 50  # Moderate specificity
    PRIORITY_FALLBACK = 10  # Broad fallback matches

    # Declarative builder configurations
    # Each entry defines a builder's selection criteria and priority
    builder_configs = [
        # ============================================================
        # EPIC and Segmentation Mask Builders (Highest Priority)
        # ============================================================
        {
            "builder": "ObjectByAnalyteConfBuilder",
            "description": "EPIC object-by-analyte datasets (no image/pyramid hints)",
            "required_hints": ["epic"],
            "forbidden_hints": ["is_support", "segmentation_mask", "is_image", "pyramid"],
            "priority": PRIORITY_SPECIFIC + 30,
        },
        {
            "builder": "EpicSegImagePyramidViewConfBuilder",
            "description": "EPIC segmentation masks with parent dataset",
            "required_hints": ["segmentation_mask"],
            "requires_parent": True,
            "requires_epic": True,
            "priority": PRIORITY_SPECIFIC + 20,
        },
        {
            "builder": "KaggleSegImagePyramidViewConfBuilder",
            "description": "Kaggle segmentation masks (non-EPIC) with parent",
            "required_hints": ["pyramid", "is_image", "segmentation_mask"],
            "requires_parent": True,
            "forbidden_hints": ["epic"],
            "priority": PRIORITY_SPECIFIC + 15,
        },
        {
            "builder": "SegmentationMaskBuilder",
            "description": "EPIC segmentation mask support datasets",
            "required_hints": ["pyramid", "epic", "is_image", "segmentation_mask"],
            "requires_epic": True,
            "requires_parent": True,
            "priority": PRIORITY_SPECIFIC + 10,
        },
        # ============================================================
        # Spatial Multiomics (Very Specific)
        # ============================================================
        {
            "builder": "XeniumMultiomicAnnDataZarrViewConfBuilder",
            "description": "Xenium spatial multiomics (is_image + xenium)",
            "required_hints": ["is_image", "xenium"],
            "priority": PRIORITY_SPECIFIC + 5,
        },
        {
            "builder": "SpatialMultiomicAnnDataZarrViewConfBuilder",
            "description": "Visium spatial RNA (is_image + rna, e.g., Visium)",
            "required_hints": ["is_image", "rna"],
            "priority": PRIORITY_SPECIFIC,
        },
        # ============================================================
        # SPRM Imaging Builders (Moderate Priority)
        # ============================================================
        {
            "builder": "MultiImageSPRMAnndataViewConfBuilder",
            "description": "CellDIVE and other SPRM with AnnData (is_image + sprm + anndata)",
            "required_hints": ["is_image", "sprm", "anndata"],
            "priority": PRIORITY_MODERATE + 15,
        },
        {
            "builder": "TiledSPRMViewConfBuilder",
            "description": "Legacy JSON-based CODEX (is_image + codex + json)",
            "required_hints": ["is_image", "json_based", "codex"],
            "priority": PRIORITY_MODERATE + 10,
        },
        {
            "builder": "StitchedCytokitSPRMViewConfBuilder",
            "description": "CODEX without JSON (is_image + codex, no json)",
            "required_hints": ["is_image", "codex"],
            "forbidden_hints": ["json_based"],
            "priority": PRIORITY_MODERATE + 5,
        },
        {
            "builder": "GeoMxImagePyramidViewConfBuilder",
            "description": "GeoMx imaging datasets",
            "required_hints": ["geomx", "is_image"],
            "priority": PRIORITY_MODERATE + 3,
        },
        # ============================================================
        # Multiomics and RNA-seq (Moderate Priority)
        # ============================================================
        {
            "builder": "MultiomicAnndataZarrViewConfBuilder",
            "description": "Multiomics without imaging (rna + atac, no is_image)",
            "required_hints": ["rna", "atac"],
            "forbidden_hints": ["is_image"],
            "priority": PRIORITY_MODERATE,
        },
        {
            "builder": "RNASeqViewConfBuilder",
            "description": "JSON-based RNA-seq (rna + json, no imaging)",
            "required_hints": ["rna", "json_based"],
            "forbidden_hints": ["is_image"],
            "priority": PRIORITY_MODERATE - 3,
        },
        {
            "builder": "SpatialRNASeqAnnDataZarrViewConfBuilder",
            "description": "Spatial RNA-seq by assay type (Salmon RNA-seq Slide)",
            "required_hints": ["rna"],
            "assay_types": [SALMON_RNASSEQ_SLIDE],
            "forbidden_hints": ["is_image", "json_based"],
            "priority": PRIORITY_MODERATE - 5,
        },
        # ============================================================
        # SPRM Non-Imaging (Fallback Priority)
        # ============================================================
        {
            "builder": "SPRMJSONViewConfBuilder",
            "description": "SPRM with JSON (no imaging)",
            "required_hints": ["sprm", "json_based"],
            "priority": PRIORITY_FALLBACK + 15,
        },
        {
            "builder": "SPRMAnnDataViewConfBuilder",
            "description": "SPRM with AnnData (no imaging)",
            "required_hints": ["sprm", "anndata"],
            "priority": PRIORITY_FALLBACK + 10,
        },
        # ============================================================
        # Generic Sequencing Data (Fallback Priority)
        # ============================================================
        {
            "builder": "RNASeqAnnDataZarrViewConfBuilder",
            "description": "Generic RNA-seq with AnnData/Zarr",
            "required_hints": ["rna"],
            "priority": PRIORITY_FALLBACK + 5,
        },
        {
            "builder": "ATACSeqViewConfBuilder",
            "description": "ATAC-seq datasets",
            "required_hints": ["atac"],
            "priority": PRIORITY_FALLBACK + 5,
        },
        # ============================================================
        # Support Image Pyramids with Parent-Specific Builders
        # ============================================================
        {
            "builder": "SeqFISHViewConfBuilder",
            "description": "SeqFISH support images (parent assay type = seqFISH)",
            "required_hints": ["is_support", "is_image"],
            "parent_assay_types": [SEQFISH],
            "requires_parent": True,
            "forbidden_hints": ["segmentation_mask"],
            "priority": PRIORITY_FALLBACK + 10,
        },
        {
            "builder": "IMSViewConfBuilder",
            "description": "MALDI-IMS support images (parent assay type = MALDI IMS)",
            "required_hints": ["is_support", "is_image"],
            "parent_assay_types": [MALDI_IMS],
            "requires_parent": True,
            "forbidden_hints": ["segmentation_mask"],
            "priority": PRIORITY_FALLBACK + 10,
        },
        {
            "builder": "NanoDESIViewConfBuilder",
            "description": "NanoDESI support images (parent assay type = NanoDESI)",
            "required_hints": ["is_support", "is_image"],
            "parent_assay_types": [NANODESI],
            "requires_parent": True,
            "forbidden_hints": ["segmentation_mask"],
            "priority": PRIORITY_FALLBACK + 10,
        },
        {
            "builder": "ImagePyramidViewConfBuilder",
            "description": "Generic support image pyramid (fallback for other parent assay types)",
            "required_hints": ["is_support", "is_image"],
            "requires_parent": True,
            "forbidden_hints": ["segmentation_mask"],
            "priority": PRIORITY_FALLBACK + 8,
        },
        # ============================================================
        # Direct Imaging by Assay Type (Fallback Priority)
        # ============================================================
        {
            "builder": "IMSViewConfBuilder",
            "description": "Direct MALDI-IMS imaging (assay type, not support)",
            "assay_types": [MALDI_IMS],
            "priority": PRIORITY_FALLBACK + 2,
        },
        {
            "builder": "NanoDESIViewConfBuilder",
            "description": "Direct NanoDESI imaging (assay type, not support)",
            "assay_types": [NANODESI],
            "priority": PRIORITY_FALLBACK + 2,
        },
        # ============================================================
        # Null Builder (Absolute Fallback)
        # ============================================================
        {
            "builder": "NullViewConfBuilder",
            "description": "Fallback for datasets without visualization support",
            "priority": 0,
        },
    ]

    # Register all builders from the configuration list
    for config in builder_configs:
        builder_name = config.pop("builder")
        _ = config.pop("description", None)  # Remove description (documentation only, not used in registration)
        _REGISTRY.register(builder_name, **config)
