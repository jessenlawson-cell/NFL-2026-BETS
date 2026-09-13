FROM python:3.13.7-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=America/Toronto \
    NFL_BETS_ROOT=/workspace \
    NFL_BETS_DATA_DIR=/workspace/data \
    NFL_BETS_CACHE_DIR=/workspace/data/cache/nflreadpy \
    NFL_BETS_RAW_DIR=/workspace/data/raw \
    NFL_BETS_CURATED_DIR=/workspace/data/curated \
    NFL_BETS_RUNTIME_DIR=/workspace/data/runtime \
    NFL_BETS_ARTIFACTS_DIR=/workspace/artifacts \
    NFL_BETS_REPORTS_DIR=/workspace/reports \
    NFL_BETS_DB_PATH=/workspace/data/runtime/nfl_bets.sqlite3

WORKDIR /workspace

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.lock pyproject.toml ./
COPY src ./src
RUN python -m pip install --upgrade pip==26.0.1 \
    && python -m pip install --requirement requirements.lock \
    && python -m pip install --no-deps --editable .

COPY . .

RUN mkdir -p \
    /workspace/data/cache/nflreadpy \
    /workspace/data/raw/nflverse \
    /workspace/data/raw/odds \
    /workspace/data/curated \
    /workspace/data/runtime \
    /workspace/artifacts/models \
    /workspace/reports \
    /workspace/logs

ENTRYPOINT ["nfl-bets"]
CMD ["--help"]
