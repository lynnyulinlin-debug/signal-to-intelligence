# Resource Registries

This directory records external models and datasets used by the tutorial.

Rules:

- Registry files are metadata only. Do not put API keys, model weights, dataset files, or private URLs here.
- Default docs build, tests, and lightweight experiments must not require network access, API keys, GPU, external model weights, or large datasets.
- Optional API and local-model experiments should reference `models.yaml`.
- Optional external datasets should reference `datasets.yaml`.
- Large files belong in local cache directories such as `models/`, `data/raw/`, `data/cache/`, or provider-managed cache locations. They are not committed.

