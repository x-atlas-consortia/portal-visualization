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
        priority: Selection priority when multiple builders match (higher = preferred)
        requires_parent: Whether this builder requires a parent entity
        requires_epic: Whether this builder requires an EPIC UUID
    """

    builder_name: str
    required_hints: set[str] = field(default_factory=set)
    forbidden_hints: set[str] = field(default_factory=set)
    assay_types: set[str] | None = None
    priority: int = 0
    requires_parent: bool = False
    requires_epic: bool = False

    def matches(self, hints: list[str], assay_type: str | None, has_parent: bool, has_epic: bool) -> bool:
        """Check if this builder matches the given criteria.

        Args:
            hints: List of vitessce-hints from entity
            assay_type: Assay type from entity
            has_parent: Whether a parent UUID was provided
            has_epic: Whether an EPIC UUID was provided

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
            priority=priority,
            requires_parent=requires_parent,
            requires_epic=requires_epic,
        )
        self._registrations.append(registration)

    def find_builder(
        self, hints: list[str], assay_type: str | None, has_parent: bool = False, has_epic: bool = False
    ) -> str | None:
        """Find the best matching builder for the given criteria.

        Args:
            hints: List of vitessce-hints
            assay_type: Assay type string
            has_parent: Whether parent UUID is available
            has_epic: Whether EPIC UUID is available

        Returns:
            Builder name or None if no match

        >>> registry = BuilderRegistry()
        >>> registry.register("LowPriority", required_hints=["is_image"], priority=1)
        >>> registry.register("HighPriority", required_hints=["is_image"], priority=10)
        >>> registry.find_builder(["is_image"], None)
        'HighPriority'
        """
        matching = [reg for reg in self._registrations if reg.matches(hints, assay_type, has_parent, has_epic)]

        if not matching:
            return None

        # Return highest priority match
        best = max(matching, key=lambda r: r.priority)
        return best.builder_name

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


def register_builder(
    required_hints: list[str] | None = None,
    forbidden_hints: list[str] | None = None,
    assay_types: list[str] | None = None,
    priority: int = 0,
    requires_parent: bool = False,
    requires_epic: bool = False,
):
    """Decorator to register a builder class.

    This decorator allows builders to self-declare their matching criteria,
    making the registration process more discoverable and maintainable.

    Example:
        >>> @register_builder(required_hints=["is_image", "rna"], priority=10)
        ... class MyBuilder:
        ...     pass
        >>> "MyBuilder" in _REGISTRY.list_builders()
        True

    Args:
        required_hints: Hints that must be present
        forbidden_hints: Hints that must NOT be present
        assay_types: Specific assay types this builder handles
        priority: Selection priority (higher wins)
        requires_parent: Whether builder needs parent entity
        requires_epic: Whether builder needs EPIC UUID

    Returns:
        Decorator function
    """

    def decorator(cls):
        _REGISTRY.register(
            builder_name=cls.__name__,
            required_hints=required_hints,
            forbidden_hints=forbidden_hints,
            assay_types=assay_types,
            priority=priority,
            requires_parent=requires_parent,
            requires_epic=requires_epic,
        )
        return cls

    return decorator


def get_registry() -> BuilderRegistry:
    """Get the global builder registry.

    Returns:
        Global BuilderRegistry instance

    >>> registry = get_registry()
    >>> isinstance(registry, BuilderRegistry)
    True
    """
    return _REGISTRY


def populate_legacy_registry():
    """Populate the registry with all existing builder mappings.

    This function replicates the logic from builder_factory._get_builder_name()
    as registry entries, maintaining backward compatibility while allowing
    gradual migration to decorator-based registration.

    Example usage:
        populate_legacy_registry()
        # Registry now contains all builder mappings
    """
    # Import assay constants that actually exist
    from .assays import CODEX_CYTOKIT, MALDI_IMS, NANODESI, SALMON_RNASSEQ_SLIDE, SEQFISH

    # Priority levels (higher = more specific match gets selected)
    PRIORITY_SPECIFIC = 100  # Very specific combinations
    PRIORITY_MODERATE = 50  # Moderate specificity
    PRIORITY_FALLBACK = 10  # Broad fallback matches

    # EPIC and segmentation mask builders (highest priority - most specific)
    _REGISTRY.register(
        "EpicSegImagePyramidViewConfBuilder",
        required_hints=["is_support", "is_image"],
        requires_parent=True,
        requires_epic=True,
        priority=PRIORITY_SPECIFIC + 20,
    )

    _REGISTRY.register(
        "SegmentationMaskBuilder", required_hints=["is_support"], requires_epic=True, priority=PRIORITY_SPECIFIC + 10
    )

    # Spatial multiomics (very specific)
    _REGISTRY.register(
        "XeniumMultiomicAnnDataZarrViewConfBuilder",
        required_hints=["is_image", "rna", "spatial", "xenium"],
        priority=PRIORITY_SPECIFIC + 5,
    )

    _REGISTRY.register(
        "SpatialMultiomicAnnDataZarrViewConfBuilder",
        required_hints=["is_image", "rna", "spatial"],
        priority=PRIORITY_SPECIFIC,
    )

    # SPRM builders (image + SPRM combinations)
    _REGISTRY.register(
        "MultiImageSPRMAnndataViewConfBuilder",
        required_hints=["is_image", "is_sprm", "is_anndata"],
        priority=PRIORITY_MODERATE + 15,
    )

    _REGISTRY.register(
        "TiledSPRMViewConfBuilder", required_hints=["is_image", "is_json", "codex"], priority=PRIORITY_MODERATE + 10
    )

    _REGISTRY.register(
        "StitchedCytokitSPRMViewConfBuilder",
        required_hints=["is_image", "codex"],
        assay_types=[CODEX_CYTOKIT],
        priority=PRIORITY_MODERATE + 5,
    )

    # SeqFISH (support dataset with parent)
    _REGISTRY.register(
        "SeqFISHViewConfBuilder",
        required_hints=["is_support", "is_image"],
        requires_parent=True,
        assay_types=[SEQFISH],
        priority=PRIORITY_MODERATE + 8,
    )

    # GeoMx
    _REGISTRY.register("GeoMxImagePyramidViewConfBuilder", required_hints=["geomx"], priority=PRIORITY_MODERATE + 3)

    # Multiomics (no image)
    _REGISTRY.register(
        "MultiomicAnndataZarrViewConfBuilder", required_hints=["is_multiome"], priority=PRIORITY_MODERATE
    )

    # Spatial RNA-seq
    _REGISTRY.register(
        "SpatialRNASeqAnnDataZarrViewConfBuilder",
        required_hints=["rna", "spatial"],
        assay_types=[SALMON_RNASSEQ_SLIDE],
        priority=PRIORITY_MODERATE - 5,
    )

    # SPRM with JSON
    _REGISTRY.register(
        "SPRMJSONViewConfBuilder", required_hints=["is_sprm", "is_json"], priority=PRIORITY_FALLBACK + 15
    )

    # SPRM with AnnData
    _REGISTRY.register(
        "SPRMAnnDataViewConfBuilder", required_hints=["is_sprm", "is_anndata"], priority=PRIORITY_FALLBACK + 10
    )

    # Object-by-analyte
    _REGISTRY.register(
        "ObjectByAnalyteConfBuilder", required_hints=["is_object_by_analyte"], priority=PRIORITY_FALLBACK + 8
    )

    # Sequencing data
    _REGISTRY.register("RNASeqAnnDataZarrViewConfBuilder", required_hints=["rna"], priority=PRIORITY_FALLBACK + 5)

    _REGISTRY.register("ATACSeqViewConfBuilder", required_hints=["atac"], priority=PRIORITY_FALLBACK + 5)

    _REGISTRY.register("RNASeqViewConfBuilder", required_hints=["is_sc"], priority=PRIORITY_FALLBACK + 3)

    # IMS imaging
    _REGISTRY.register("IMSViewConfBuilder", assay_types=[MALDI_IMS], priority=PRIORITY_FALLBACK + 2)

    # NanoDESI imaging
    _REGISTRY.register("NanoDESIViewConfBuilder", assay_types=[NANODESI], priority=PRIORITY_FALLBACK + 2)

    # Fallback image pyramid builder
    _REGISTRY.register("ImagePyramidViewConfBuilder", required_hints=["is_image"], priority=PRIORITY_FALLBACK)

    # Null builder (absolute fallback - no hints required)
    _REGISTRY.register("NullViewConfBuilder", priority=0)
