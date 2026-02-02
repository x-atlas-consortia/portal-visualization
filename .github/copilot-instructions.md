# AI Coding Agent Instructions for portal-visualization

## Project Overview

This is a **HuBMAP visualization configuration generator** that converts dataset metadata into Vitessce viewer configurations. The library is used by both `portal-ui` and `search-api` to dynamically generate interactive visualizations for biological datasets (imaging, RNA-seq, ATAC-seq, etc.).

**Core Workflow**: Dataset entity JSON → `builder_factory.py` selects appropriate Builder → Builder generates Vitessce config → Rendered in portal or returned via API

## Installation Modes

The package supports two installation modes:

### Thin Install (Default)

```bash
pip install portal-visualization
```

- **Purpose**: Check if datasets have visualization support (`has_visualization()`)
- **Size**: <1 MB (pure Python, no heavy dependencies)
- **Use case**: Services that only need to filter/check datasets
- **Available functions**: `has_visualization()`, `process_hints()`

### Full Install

```bash
pip install portal-visualization[full]
```

- **Purpose**: Complete visualization generation capabilities
- **Size**: ~150 MB (includes vitessce, zarr, aiohttp, nbformat, etc.)
- **Use case**: portal-ui, search-api, actual visualization rendering
- **Available functions**: All builder classes and utilities

**Important**: Downstream projects (portal-ui, search-api) must install with `[full]` extras.

## Architecture & Key Concepts

### Builder Selection Patterns

The system supports two approaches for selecting the appropriate builder:

#### 1. Registry-Based Selection (New, Recommended)

- **Entry point**: `builder_factory.get_view_config_builder(entity, get_entity, parent, epic_uuid)`
- **Feature flag**: Set `USE_BUILDER_REGISTRY=1` environment variable to enable (experimental)
- **Architecture**: Declarative configuration list in `builder_registry.py:populate_legacy_registry()`
  - Each builder has a config dict with `required_hints`, `forbidden_hints`, `assay_types`, `parent_assay_types`, `priority`
  - Builders are registered via `_REGISTRY.register(builder_name, **config)`
  - Selection uses priority-based matching (higher priority wins when multiple match)
- **Parent assay type support**: Builders can specify `parent_assay_types` to match based on parent dataset's assay type
  - Example: `SeqFISHViewConfBuilder` requires `parent_assay_types=[SEQFISH]`
- **Benefits**: Self-documenting, easy to modify, clear priority ordering, DRY principle
- **Maintains compatibility**: Still uses string-based registration with lazy imports for thin install

#### 2. Legacy Factory Pattern (Fallback)

- **Selection logic**: Uses `vitessce-hints` metadata array from entity to determine which builder class to instantiate
  - Hints like `["is_image", "rna", "spatial"]` → `SpatialMultiomicAnnDataZarrViewConfBuilder`
  - Hints like `["is_image", "codex"]` → `StitchedCytokitSPRMViewConfBuilder`
  - See `builder_factory.py:_get_builder_name()` for the full decision tree
- **Active when**: `USE_BUILDER_REGISTRY=0` (default) or registry returns no match

#### Common Elements

- **Lazy imports**: Builder classes are imported only when needed to support thin install
- **Visualization lifting**: Image pyramids are "vis-lifted" from support datasets to their parent dataset pages via `parent` parameter

### Builder Hierarchy

All builders inherit from `ViewConfBuilder` (abstract base in `builders/base_builders.py`):

- **Core method**: `get_conf_cells(**kwargs)` returns `ConfCells(conf_dict, notebook_cells)` namedtuple
- **Common utilities**:
  - `_build_assets_url(rel_path)` constructs authenticated asset URLs with token params
  - `_get_request_init()` provides auth headers for Zarr stores (non-public data requires Bearer token)
- **Imaging builders**: Extend `AbstractImagingViewConfBuilder` for OME-TIFF pyramid handling
- **AnnData builders**: Handle Zarr-backed AnnData stores for sequencing data

### File Path Conventions

Builders discover data files using regex patterns defined in `paths.py`:

- Image pyramids: `stitched/expressions/` or `stitched_expressions/`
- Segmentation masks: `segmentation_masks_Probabilities_*` or `kaggle_mask/`
- Offsets: `output_offsets/*.offsets.json` (for optimized image loading via Viv)
- Image metadata: `image_metadata/*.metadata.json` (physical size units for scaling)

## Development Workflows

### Testing

Run tests via `./test.sh` which:

1. Validates README matches `vis-preview.py --help` output (docs must stay in sync!)
2. Runs `ruff` linting
3. Executes pytest with **100% coverage requirement** (`--doctest-modules` enabled)

**Test modes**:

- Full test suite: `./test.sh` (requires `[full]` extras)
- Thin install tests only: `pytest -m "not requires_full"` (no heavy dependencies needed)
- Tests requiring full install are marked with `@pytest.mark.requires_full`

Fixture structure:

- `test/good-fixtures/BuilderName/uuid-entity.json` → fixtures for valid datasets
- `test/bad-fixtures/uuid-entity.json` → error case testing
- `test/assaytype-fixtures/uuid.json` → mock assay type metadata

### Adding New Assay Support

#### Registry-Based Approach (Recommended)

1. Define assay constant in `assays.py` if needed (e.g., `SEQFISH = "seqFish"`)
2. Create builder class in appropriate `builders/*_builders.py` file
3. Add configuration entry to `builder_registry.py:populate_legacy_registry()`:
   ```python
   {
       "builder": "YourNewBuilder",
       "description": "Your assay description",
       "required_hints": ["hint1", "hint2"],
       "forbidden_hints": ["hint3"],  # optional
       "assay_types": [YOUR_ASSAY],  # optional
       "parent_assay_types": [PARENT_ASSAY],  # optional
       "priority": PRIORITY_SPECIFIC,  # adjust as needed
   },
   ```
4. Add builder name to `builder_factory.py:_lazy_import_builder()` function
5. Add test fixtures: `test/good-fixtures/YourBuilder/{uuid}-entity.json`
6. Run tests with `USE_BUILDER_REGISTRY=1` to verify registry selection
7. Verify README describes when the builder is used (see "Imaging Data" section)

#### Legacy Approach (Maintenance Only)

1. Define assay constant in `assays.py` (e.g., `SEQFISH = "seqFish"`)
2. Create builder class in appropriate `builders/*_builders.py` file
3. Update `builder_factory.py:_get_builder_name()` decision tree with new hint combinations
4. Add builder name to `_lazy_import_builder()` function
5. Add test fixtures: `test/good-fixtures/YourBuilder/{uuid}-entity.json`
6. Verify README describes when the builder is used (see "Imaging Data" section)

**Note**: When adding new builders, update BOTH the registry configuration (for future) AND the legacy factory logic (for backward compatibility) until the registry becomes the default.

## Code Conventions

### Lazy Imports and Registry Architecture

To support the thin install, builder imports and registration are lazy:

```python
# builder_factory.py uses lazy imports
def _lazy_import_builder(builder_name):
    """Import builder class only when needed."""
    if builder_name == 'RNASeqAnnDataZarrViewConfBuilder':
        from .builders.anndata_builders import RNASeqAnnDataZarrViewConfBuilder
        return RNASeqAnnDataZarrViewConfBuilder
    # ... etc

# Registry approach: string-based registration (no imports)
def populate_legacy_registry():
    """Populate registry with declarative configuration."""
    builder_configs = [
        {
            "builder": "RNASeqAnnDataZarrViewConfBuilder",
            "description": "Generic RNA-seq with AnnData/Zarr",
            "required_hints": ["rna"],
            "priority": PRIORITY_FALLBACK + 5,
        },
        # ... more configs ...
    ]
    for config in builder_configs:
        builder_name = config.pop("builder")
        _ = config.pop("description", None)  # Documentation only
        _REGISTRY.register(builder_name, **config)

# Legacy factory: returns string names (no imports)
def _get_builder_name(entity, ...):
    """Pure Python logic - works in thin install."""
    return 'RNASeqAnnDataZarrViewConfBuilder'  # string, not class

# get_view_config_builder() combines them for full install
def get_view_config_builder(entity, ...):
    """Returns actual builder class (requires [full] install)."""
    if USE_BUILDER_REGISTRY:
        _ensure_registry_initialized()
        builder_name = _registry.find_builder(entity, parent, epic_uuid)
    else:
        builder_name = _get_builder_name(entity, ...)
    return _lazy_import_builder(builder_name)
```

**Why not decorators?** Decorator-based registration (@register_builder on classes) would require importing builder modules (which import heavy dependencies like vitessce/zarr) at registration time, breaking thin install. String-based registration allows lazy import of builders only when actually needed.

### Doctests

Inline doctests are mandatory for coverage. Use this pattern:

```python
def _build_assets_url(self, rel_path):
    """Create a url for an asset.
    >>> builder = _DocTestBuilder(
    ...   entity={"uuid": "uuid"}, groups_token='token',
    ...   assets_endpoint='https://example.com')
    >>> builder._build_assets_url("path/to/file.tiff")
    'https://example.com/uuid/path/to/file.tiff?token=token'
    """
```

### Error Handling

- Use `# pragma: no cover` for production-only code (e.g., Flask abort calls in `client.py`)
- Wrap builder errors in `ConfCells` with error message for graceful degradation in portal UI
- Log errors via `current_app.logger.error()` when Flask context available

## Critical Integration Points

### Portal-UI Integration

Called from `portal-ui/context/app/routes_browse.py`:

```python
from portal_visualization.builder_factory import get_view_config_builder
builder = get_view_config_builder(entity, get_entity_fn)
conf_cells = builder.get_conf_cells(marker=marker)
```

**Requires**: `pip install portal-visualization[full]`

### Search-API Integration

Similar usage but may specify `minimal=True` kwarg for lightweight configs

**Requires**: `pip install portal-visualization[full]`

### Environment-Specific URLs

`defaults.json` defines dev/prod endpoints:

- Assets: `https://assets.{dev.}hubmapconsortium.org`
- Entity API: `https://entity-api.{dev.}hubmapconsortium.org`
- Always use `assets_endpoint` parameter, never hardcode URLs

## Dependencies & Versioning

- **Primary dependency**: `vitessce==3.7.4` (pinned due to downstream conflicts)
- **Dependency structure**: Core has no dependencies; `[full]` extra includes all visualization dependencies
- **Release process**: Bump `VERSION.txt` → git tag → GitHub release → update `requirements.txt` in portal-ui and search-api with `[full]` extras
- **Python version**: Requires >=3.10 (see `pyproject.toml`)

## Common Pitfalls

1. **Forgetting token auth**: Non-public datasets require `groups_token` in URLs or request headers
2. **Image pyramid detection**: Use `get_found_images()` from `utils.py`, not custom regex (handles `separate/` exclusions)
3. **Physical size scaling**: When overlaying segmentation masks, retrieve metadata JSONs and compute scale via `get_image_scale()` in `utils.py`
4. **Registry vs. factory sync**: When adding new builders, update BOTH:
   - Registry: Add config dict to `builder_registry.py:populate_legacy_registry()`
   - Factory: Add logic to `builder_factory.py:_get_builder_name()` (backward compatibility)
5. **Hint processing**: Add new hints to BOTH `_get_builder_name()` return statements AND the `_lazy_import_builder()` function
6. **Import errors**: If adding new builder, must update `_lazy_import_builder()` with lazy import pattern
7. **Testing without full install**: Mark tests with `@pytest.mark.requires_full` if they need visualization dependencies
8. **Testing registry**: Always test with both `USE_BUILDER_REGISTRY=0` (default) and `USE_BUILDER_REGISTRY=1` to ensure parity

## Command Line Environment
Use `source .venv/bin/activate` to activate the virtual environment for development and testing.