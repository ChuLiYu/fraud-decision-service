# Agent guide

- Entry point: [docs/INDEX.md](docs/INDEX.md). Scope and acceptance live in [docs/product/scope.md](docs/product/scope.md); do not add features outside it without an updated scope or ADR.
- English documents are authoritative; `*.zh-TW.md` files are derived reading versions. Update the English source first, then sync or mark the reading version as pending.
- Never commit raw IEEE-CIS data, trained model binaries, or credentials. Data is fetched by script.
- Never state a target (throughput, latency, metric) as a result. Results need a linked report with hardware, data volume, and settings.
- Model failure or timeout must route to review, never approval ([ADR-003](docs/decisions/ADR-003-failure-routes-to-review.md)).
- Tool and library versions are chosen and pinned when each component is scaffolded, after checking current support; record them in that component's build files.
