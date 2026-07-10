# Contributing

## Project Structure

```text
signdata/
├── configs/
│   ├── experiments/            # Multi-job experiment configs
│   │   ├── baseline_youtube_asl.yaml
│   │   └── privacy_aware_slt.yaml
│   └── jobs/                   # Runnable job configs
│       ├── youtube_asl/
│       │   ├── mediapipe.yaml
│       │   ├── mmpose.yaml
│       │   └── video.yaml
│       └── how2sign/
│           ├── mediapipe.yaml
│           ├── mmpose.yaml
│           └── video.yaml
├── src/signdata/
│   ├── __main__.py             # CLI entry point
│   ├── cli.py                  # Argument parsing
│   ├── registry.py             # Component registry
│   ├── config/                 # YAML loading and Pydantic schema
│   ├── pipeline/               # Recipes, runner, and context
│   ├── datasets/               # Dataset adapters
│   ├── detection/              # Detection backends
│   ├── pose/                   # Pose extractors and variants
│   ├── processors/             # Pipeline processors
│   ├── presets/                # Keypoint presets
│   └── utils/                  # Video, file, and text utilities
├── resources/                  # Shipped model config assets
├── docs/                       # Documentation
├── assets/                     # Video ID lists and small assets
├── tests/
│   ├── integration/
│   └── unit/
└── requirements.txt
```

## Adding a New Dataset

All datasets must be packages now. Do not add new flat modules like
`src/signdata/datasets/my_dataset.py`.

Use this default layout:

```text
src/signdata/datasets/
├── _ingestion/               # Dataset-ingestion helpers only
│   ├── availability.py
│   ├── classmap.py
│   ├── media.py
│   ├── paths.py
│   └── youtube.py
└── my_dataset/
    ├── __init__.py
    ├── adapter.py
    ├── source.py
    └── manifest.py
```

### File Responsibilities

- `adapter.py`
  - Public dataset entrypoint only.
  - Contains `@register_dataset("my_dataset")` and the adapter class.
  - Keeps `download()` and `build_manifest()` thin by delegating to `source.py` and `manifest.py`.
- `source.py`
  - Owns `SourceConfig`, source path resolution, release discovery, validation, download, and preparation/materialization.
- `manifest.py`
  - Owns source metadata parsing and canonical manifest construction.
  - Handles split assignment, class-map joins, timing/bbox conversion, and TSV writing.

Add more files only when needed:

- `schema.py` for multiple typed row/config models
- `constants.py` for larger aliases or bundled filename tables
- `parsing.py` or `splits.py` when `manifest.py` becomes too dense

### Shared Helper Rule

- If a helper is used only by dataset ingestion or manifest building, put it in `src/signdata/datasets/_ingestion/`.
- If a helper is generic to the pipeline, processors, or outputs, keep it in `src/signdata/utils/`.

### Minimal Template

`src/signdata/datasets/my_dataset/__init__.py`

```python
from .adapter import MyDataset
```

`src/signdata/datasets/my_dataset/adapter.py`

```python
from pathlib import Path

from signdata.datasets.base import DatasetAdapter
from signdata.registry import register_dataset
from . import manifest as _manifest
from . import source as _source


@register_dataset("my_dataset")
class MyDataset(DatasetAdapter):
    name = "my_dataset"

    @classmethod
    def validate_config(cls, config) -> None:
        pass

    def download(self, config, context):
        source = _source.get_source_config(config)
        context.stats["dataset.download"] = _source.validate(source, config, self.logger)
        return context

    def build_manifest(self, config, context):
        source = _source.get_source_config(config)
        df = _manifest.build(config, source, self.logger)
        context.manifest_path = Path(config.paths.manifest)
        context.manifest_df = df
        return context
```

### Integration Steps

1. Create `src/signdata/datasets/my_dataset/` using the package layout above.
2. Export the adapter in `src/signdata/datasets/my_dataset/__init__.py`.
3. Re-export it from `src/signdata/datasets/__init__.py`.
4. Add a job YAML under `configs/jobs/my_dataset/`, for example `configs/jobs/my_dataset/mediapipe.yaml`. See [configuration reference](docs/configuration.md#minimal-working-config).
5. Add dataset documentation to `docs/datasets.md`.

## Adding a New Processor

1. Create a class in the appropriate `src/signdata/processors/<group>/` package decorated with `@register_processor`:

```python
from signdata.processors.base import BaseProcessor
from signdata.registry import register_processor


@register_processor("my_step")
class MyProcessor(BaseProcessor):
    name = "my_step"

    def run(self, context):
        return context

    def validate(self, context) -> bool:
        return True
```

2. Import it in the matching package `__init__.py`, such as `src/signdata/processors/pose/__init__.py` or `src/signdata/processors/video/__init__.py`. Re-export it from `src/signdata/processors/__init__.py` if the top-level package should expose it.

3. Add the stage to the relevant recipe in `src/signdata/pipeline/recipes.py`, or gate it through `config.stage_config` if it should remain optional.

## Adding a New Extractor

1. Create a class in `src/signdata/pose/` or one of its backend packages decorated with `@register_extractor`:

```python
from signdata.processors.pose.base import LandmarkExtractor
from signdata.registry import register_extractor


@register_extractor("my_extractor")
class MyExtractor(LandmarkExtractor):
    def process_frame(self, frame):
        ...

    def close(self):
        ...
```

2. Import it in `src/signdata/pose/__init__.py`.

3. Set `extractor.name: my_extractor` in your job YAML.

## Running Tests

```bash
python -m pytest -q tests
```

See [architecture docs](docs/architecture.md) for how the registry, recipes, and pipeline runner fit together.
