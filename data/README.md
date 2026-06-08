# Data Directory

This directory is reserved for small project-owned fixtures and local data notes.

Default project behavior:

- Documentation build does not require data downloads.
- Tests do not require network access or external datasets.
- Lightweight chapter experiments should use generated data or tiny fixtures.
- Large datasets are not committed.

Allowed to commit:

- Small project-owned fixtures.
- Small metadata files explaining fixture provenance and license.

Do not commit:

- Raw external datasets.
- Model weights.
- API outputs containing private data.
- Large generated caches.

Common local-only locations:

- `data/raw/`
- `data/cache/`
- `data/external/`
- `models/`

External datasets should be registered in `configs/datasets.yaml`.
External models and API-backed model access should be registered in `configs/models.yaml`.

