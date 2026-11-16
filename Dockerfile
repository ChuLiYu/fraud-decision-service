# Decision service image (ADR-005). Builds demo bundles from synthetic rows
# (active, and a shadow model from another seed for S-7),
# so the image runs without the IEEE-CIS dataset. Mount a real bundle at
# /app/bundles/real and set BUNDLE_DIR to use it (never bake it into an image).
# Base image checked 2026-11-12: uv 0.12.24 with Python 3.14 on Debian trixie.
FROM astral/uv:0.12.24-python3.14-trixie-slim

# LightGBM needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY ml/pyproject.toml ml/uv.lock ml/params.yaml ml/
COPY ml/src ml/src
COPY service/pyproject.toml service/uv.lock service/
COPY service/src service/src
RUN cd service && uv sync --locked --no-dev

ENV PATH="/app/service/.venv/bin:$PATH"
RUN cd ml && python -m fraud_ml.bundle --demo --out /app/bundles/demo \
    && python -m fraud_ml.bundle --demo --seed 1 --out /app/bundles/shadow

RUN useradd --create-home app
USER app
# DATABASE_URL (PostgreSQL) is set by compose; the service refuses to start without it.
ENV BUNDLE_DIR=/app/bundles/demo
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "fraud_service.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
