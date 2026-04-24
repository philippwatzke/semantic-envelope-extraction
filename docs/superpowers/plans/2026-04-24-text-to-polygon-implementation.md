# Text-to-Polygon Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local text-to-polygon pipeline that extracts georeferenced vector polygons from DOP20 orthophotos in Bayern via zero-shot SAM 3.1 segmentation, driven by user-drawn BBox + free-text prompts, with async job management, review UI, and GeoPackage export.

**Architecture:** Two OS processes (FastAPI webserver + GPU worker), coordinated via SQLite (WAL-mode). Webserver never touches the GPU. Pipeline is layered: `app/` (HTTP), `worker/` (GPU orchestration), `pipeline/` (pure image processing), `jobs/` (DB + models). Concurrency is physically 1 (single worker process holds SAM 3.1). Worker self-restarts after `MAX_JOBS_PER_WORKER` to flush VRAM fragmentation.

**Tech Stack:** Python 3.12, FastAPI, uvicorn, SAM 3.1 (PyTorch 2.7+, CUDA 12.6+), rasterio, geopandas, shapely 2.x, pyproj, pydantic-settings, SQLite, Leaflet + vanilla JS.

**Design spec:** `docs/superpowers/specs/2026-04-22-text-to-polygon-design.md` — referenced throughout this plan as **Spec §N.M**. When a task says "see Spec §5.1 pt 1", the engineer MUST read that section before writing code; it contains production-critical nuances (FP traps, grid origins, axis order) that are not duplicated here.

**Order of execution:** Tasks are topologically ordered. Later tasks depend on earlier symbols, exact signatures, and exact config field names. Do not reorder without re-validating type consistency.

---

## File Structure

```
ki_geodaten/
├── __init__.py
├── config.py                     # pydantic-settings Settings singleton
├── models.py                     # Pydantic data models (Job, BBox, TileConfig, etc.)
├── pipeline/
│   ├── __init__.py
│   ├── geo_utils.py              # snap_floor/snap_ceil (Decimal), transform_bounds wrapper
│   ├── wcs_client.py             # DOP20 download, grid-snap, pagination, VRT build
│   ├── tiler.py                  # iter_tiles, TileConfig, NoData detection
│   ├── segmenter.py              # Sam3Segmenter, local mask NMS
│   ├── merger.py                 # keep_center_only, masks_to_polygons, extract_polygons
│   └── exporter.py               # two-layer GPKG export with AOI clip + empty-schema
├── jobs/
│   ├── __init__.py
│   ├── store.py                  # SQLite connection factory, schema init, CRUD
│   └── retention.py              # Daily cleanup of FAILED/EXPORTED jobs
├── worker/
│   ├── __init__.py
│   ├── loop.py                   # Poll loop, startup hook, disk zombie cleanup, restart
│   └── orchestrator.py           # Job runner: download → tile → infer → merge → persist
├── app/
│   ├── __init__.py
│   ├── main.py                   # FastAPI app factory, lifespan, executor
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── jobs.py               # POST /jobs, GET /jobs, GET /jobs/{id}, validate_bulk, export
│   │   └── geojson.py            # /polygons, /nodata with dedicated ProcessPoolExecutor
│   ├── serialization.py          # WKB → GeoJSON (runs in ProcessPoolExecutor)
│   ├── templates/
│   │   └── index.html            # Leaflet UI (single page)
│   └── static/
│       ├── app.js                # Vanilla JS: draw, submit, poll, validate, debounce
│       └── app.css
├── data/
│   ├── dop/                      # DOP chunks + VRTs per job (auto-cleaned)
│   ├── results/                  # {job_id}.gpkg files
│   └── jobs.db                   # SQLite (WAL)
├── models/
│   └── sam3.1_hiera_large.pt     # Downloaded manually (not versioned)
├── scripts/
│   ├── run-server.sh
│   └── run-worker.sh
├── tests/
│   ├── conftest.py               # Fixtures: tmp DB, synthetic GeoTIFF, stub segmenter
│   ├── pipeline/
│   │   ├── test_geo_utils.py
│   │   ├── test_wcs_client.py
│   │   ├── test_tiler.py
│   │   ├── test_segmenter.py
│   │   ├── test_merger.py
│   │   └── test_exporter.py
│   ├── jobs/
│   │   ├── test_store.py
│   │   └── test_retention.py
│   ├── worker/
│   │   └── test_loop.py
│   ├── app/
│   │   └── test_routes.py
│   └── test_end_to_end.py
├── pyproject.toml
├── .env.example
└── README.md
```

**Dependency direction:** `app/ → jobs/`; `worker/ → pipeline/` + `jobs/`; `pipeline/` has no upward deps. `app/` and `pipeline/` do not know each other.

---

## Task 1: Project Scaffolding & Dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `ki_geodaten/__init__.py`
- Create: `ki_geodaten/pipeline/__init__.py`, `ki_geodaten/jobs/__init__.py`, `ki_geodaten/worker/__init__.py`, `ki_geodaten/app/__init__.py`, `ki_geodaten/app/routes/__init__.py`
- Create: `tests/__init__.py`, `tests/pipeline/__init__.py`, `tests/jobs/__init__.py`, `tests/worker/__init__.py`, `tests/app/__init__.py`
- Create: `.gitignore`, `.env.example`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "ki-geodaten"
version = "0.1.0"
description = "Text-to-Polygon Pipeline — DOP20 + SAM 3.1"
requires-python = ">=3.12"
dependencies = [
    "torch>=2.7",
    "rasterio>=1.3",
    "geopandas>=1.0",
    "shapely>=2.0",
    "pyproj>=3.6",
    "fastapi",
    "uvicorn[standard]",
    "jinja2",
    "pydantic>=2",
    "pydantic-settings",
    "requests",
    "urllib3>=2",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-mock", "responses", "httpx"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --tb=short"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["ki_geodaten*"]
```

Note: `sam3` and `gdal` are not pip-installable cleanly; documented as "install via conda/source" in README.

- [ ] **Step 2: Create empty `__init__.py` files** (one per directory listed above — empty is fine).

- [ ] **Step 3: Create `.gitignore`**

```
__pycache__/
*.pyc
.venv/
data/dop/
data/results/
data/*.db*
models/*.pt
.env
.pytest_cache/
```

- [ ] **Step 4: Create `.env.example`**

```
WCS_URL=https://geoservices.bayern.de/wcs/v2/dop20
WCS_COVERAGE_ID=by_dop20rgb
SAM3_CHECKPOINT=models/sam3.1_hiera_large.pt
```

- [ ] **Step 5: Verify install**

Run: `pip install -e .[dev]`
Expected: install succeeds (ignore sam3/gdal for now).

- [ ] **Step 6: Commit**

```bash
git init
git add .
git commit -m "chore: project scaffolding and dependencies"
```

---

## Task 2: Configuration Module (`config.py`)

**Files:**
- Create: `ki_geodaten/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_config.py
from ki_geodaten.config import Settings

def test_settings_defaults():
    s = Settings()
    assert s.WCS_MAX_PIXELS == 4000
    assert s.TILE_SIZE == 1024
    assert s.DEFAULT_TILE_PRESET == "medium"
    assert s.MIN_POLYGON_AREA_M2 == 1.0
    assert s.LOCAL_MASK_NMS_IOU == 0.6
    assert s.LOCAL_MASK_CONTAINMENT_RATIO == 0.9
    assert s.SAFE_CENTER_NODATA_THRESHOLD == 0.0
    assert s.MAX_JOBS_PER_WORKER == 50
    assert s.WORKER_POLL_INTERVAL_SEC == 2.0
    assert s.MAX_BBOX_AREA_KM2 == 1.0
    assert s.MAX_PROMPT_CHARS == 240
    assert s.MAX_ENCODER_CONTEXT_TOKENS == 77
    assert s.MAX_CLIENT_BUFFER_UPDATES == 100
    assert s.BAYERN_BBOX_WGS84 == (8.9, 47.2, 13.9, 50.6)
    assert s.RETENTION_DAYS == 7
    assert s.WCS_GRID_ORIGIN_X == 0.0
    assert s.WCS_GRID_ORIGIN_Y == 0.0

def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("MAX_BBOX_AREA_KM2", "2.5")
    s = Settings()
    assert s.MAX_BBOX_AREA_KM2 == 2.5
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL (ModuleNotFoundError: ki_geodaten.config).

- [ ] **Step 3: Implement `config.py`**

```python
# ki_geodaten/config.py
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # WCS
    WCS_URL: str = "PLACEHOLDER_UNTIL_VERIFIED"
    WCS_COVERAGE_ID: str = "PLACEHOLDER_UNTIL_VERIFIED"
    WCS_MAX_PIXELS: int = 4000
    WCS_GRID_ORIGIN_X: float = 0.0
    WCS_GRID_ORIGIN_Y: float = 0.0

    # SAM 3.1
    SAM3_CHECKPOINT: Path = Path("models/sam3.1_hiera_large.pt")

    # Tiling
    TILE_SIZE: int = 1024
    DEFAULT_TILE_PRESET: str = "medium"

    # Filtering
    MIN_POLYGON_AREA_M2: float = 1.0
    LOCAL_MASK_NMS_IOU: float = 0.6
    LOCAL_MASK_CONTAINMENT_RATIO: float = 0.9
    SAFE_CENTER_NODATA_THRESHOLD: float = 0.0

    # Worker
    MAX_JOBS_PER_WORKER: int = 50
    WORKER_POLL_INTERVAL_SEC: float = 2.0

    # API limits
    MAX_BBOX_AREA_KM2: float = 1.0
    MAX_PROMPT_CHARS: int = 240
    MAX_ENCODER_CONTEXT_TOKENS: int = 77
    MAX_CLIENT_BUFFER_UPDATES: int = 100

    # Geographic Fence
    BAYERN_BBOX_WGS84: tuple[float, float, float, float] = (8.9, 47.2, 13.9, 50.6)

    # Retention
    RETENTION_DAYS: int = 7

    # Paths
    DATA_DIR: Path = Path("data")
    DOP_DIR: Path = Path("data/dop")
    RESULTS_DIR: Path = Path("data/results")
    DB_PATH: Path = Path("data/jobs.db")

settings = Settings()
```

- [ ] **Step 4: Run tests — expect pass**

Run: `pytest tests/test_config.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add ki_geodaten/config.py tests/test_config.py
git commit -m "feat(config): add Settings with pydantic-settings"
```

---

## Task 3: Pydantic Models (`models.py`)

**Files:**
- Create: `ki_geodaten/models.py`
- Test: `tests/test_models.py`

Defines: `BBox` (WGS84 or UTM), `TilePreset`, `JobStatus`, `ErrorReason`, `Validation`, `NoDataReason`, API request/response schemas.

- [ ] **Step 1: Write failing test**

```python
# tests/test_models.py
import pytest
from pydantic import ValidationError
from ki_geodaten.models import BBox, CreateJobRequest, TilePreset, ValidateBulkRequest, ValidationUpdate

def test_bbox_accepts_valid():
    b = BBox(minx=11.0, miny=48.0, maxx=11.1, maxy=48.1)
    assert b.as_tuple() == (11.0, 48.0, 11.1, 48.1)

def test_bbox_rejects_inverted():
    with pytest.raises(ValidationError):
        BBox(minx=11.1, miny=48.0, maxx=11.0, maxy=48.1)  # minx >= maxx
    with pytest.raises(ValidationError):
        BBox(minx=11.0, miny=48.1, maxx=11.1, maxy=48.0)  # miny >= maxy

def test_create_job_request_defaults_preset_to_medium():
    req = CreateJobRequest(prompt="building", bbox_wgs84=[11.0, 48.0, 11.1, 48.1])
    assert req.tile_preset == TilePreset.MEDIUM

def test_validate_bulk_request():
    r = ValidateBulkRequest(updates=[ValidationUpdate(pid=1, validation="REJECTED")])
    assert r.updates[0].pid == 1
    assert r.updates[0].validation == "REJECTED"

def test_validation_update_rejects_unknown_value():
    with pytest.raises(ValidationError):
        ValidationUpdate(pid=1, validation="MAYBE")
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_models.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement `models.py`**

```python
# ki_geodaten/models.py
from __future__ import annotations
from enum import StrEnum
from typing import Literal
from pydantic import BaseModel, Field, model_validator

class TilePreset(StrEnum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

class JobStatus(StrEnum):
    PENDING = "PENDING"
    DOWNLOADING = "DOWNLOADING"
    INFERRING = "INFERRING"
    READY_FOR_REVIEW = "READY_FOR_REVIEW"
    EXPORTED = "EXPORTED"
    FAILED = "FAILED"

class ErrorReason(StrEnum):
    WCS_TIMEOUT = "WCS_TIMEOUT"
    WCS_HTTP_ERROR = "WCS_HTTP_ERROR"
    OOM = "OOM"
    INFERENCE_ERROR = "INFERENCE_ERROR"
    WORKER_RESTARTED = "WORKER_RESTARTED"
    EXPORT_ERROR = "EXPORT_ERROR"
    INVALID_GEOMETRY = "INVALID_GEOMETRY"

class NoDataReason(StrEnum):
    OOM = "OOM"
    INFERENCE_ERROR = "INFERENCE_ERROR"
    INVALID_GEOMETRY = "INVALID_GEOMETRY"
    NODATA_PIXELS = "NODATA_PIXELS"

Validation = Literal["ACCEPTED", "REJECTED"]

class BBox(BaseModel):
    minx: float
    miny: float
    maxx: float
    maxy: float

    @model_validator(mode="after")
    def _check_ordered(self) -> "BBox":
        if self.minx >= self.maxx:
            raise ValueError("minx must be < maxx")
        if self.miny >= self.maxy:
            raise ValueError("miny must be < maxy")
        return self

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.minx, self.miny, self.maxx, self.maxy)

class CreateJobRequest(BaseModel):
    prompt: str = Field(min_length=1)
    bbox_wgs84: list[float] = Field(min_length=4, max_length=4)
    tile_preset: TilePreset = TilePreset.MEDIUM

class ValidationUpdate(BaseModel):
    pid: int
    validation: Validation

class ValidateBulkRequest(BaseModel):
    updates: list[ValidationUpdate]
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_models.py -v`
Expected: all 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add ki_geodaten/models.py tests/test_models.py
git commit -m "feat(models): add pydantic data models for jobs and API"
```

---

## Task 4: Geo Utilities (`pipeline/geo_utils.py`) — Grid Snapping & Transform

**Files:**
- Create: `ki_geodaten/pipeline/geo_utils.py`
- Test: `tests/pipeline/test_geo_utils.py`

**See Spec §5.1 pts 1, 6** — Decimal-based origin-sensitive snapping; `transform_bounds` with densify.

- [ ] **Step 1: Write failing test**

```python
# tests/pipeline/test_geo_utils.py
import pytest
from ki_geodaten.pipeline.geo_utils import (
    snap_floor, snap_ceil, transform_bbox_wgs84_to_utm, pixel_count,
)

def test_snap_floor_fp_edge_case():
    # 0.6 / 0.2 == 2.9999... in IEEE-754; naive floor would give 0.4
    assert snap_floor(0.6, origin=0.0, step=0.2) == pytest.approx(0.6)
    assert snap_floor(1.3, origin=0.0, step=0.2) == pytest.approx(1.2)
    assert snap_floor(600000.15, origin=0.0, step=0.2) == pytest.approx(600000.0)

def test_snap_ceil_fp_edge_case():
    assert snap_ceil(0.55, origin=0.0, step=0.2) == pytest.approx(0.6)
    assert snap_ceil(600000.01, origin=0.0, step=0.2) == pytest.approx(600000.2)

def test_snap_with_nonzero_origin():
    # Origin 0.1, step 0.2 → valid values are 0.1, 0.3, 0.5, ...
    assert snap_floor(0.45, origin=0.1, step=0.2) == pytest.approx(0.3)
    assert snap_ceil(0.45, origin=0.1, step=0.2) == pytest.approx(0.5)

def test_transform_bbox_uses_densify():
    # Munich-ish BBox
    minx, miny, maxx, maxy = transform_bbox_wgs84_to_utm(11.0, 48.0, 11.1, 48.1)
    # rough UTM32N Munich range: 640k–700k easting, 5.3M–5.4M northing
    assert 640_000 < minx < 700_000
    assert 5_300_000 < miny < 5_400_000
    assert maxx > minx
    assert maxy > miny

def test_pixel_count_round_not_int():
    # 300.0 / 0.2 == 1499.9999... in IEEE-754; round() yields 1500, int() yields 1499
    assert pixel_count(minx=0.0, maxx=300.0, step=0.2) == 1500
    assert pixel_count(minx=0.0, maxx=204.8, step=0.2) == 1024
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/pipeline/test_geo_utils.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `geo_utils.py`**

```python
# ki_geodaten/pipeline/geo_utils.py
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
from functools import cache
import pyproj

@cache
def _transformer_4326_to_25832() -> pyproj.Transformer:
    return pyproj.Transformer.from_crs(4326, 25832, always_xy=True)

@cache
def _transformer_25832_to_4326() -> pyproj.Transformer:
    return pyproj.Transformer.from_crs(25832, 4326, always_xy=True)

def transformer_25832_to_4326() -> pyproj.Transformer:
    return _transformer_25832_to_4326()

def snap_floor(x: float, origin: float, step: float = 0.2) -> float:
    d_origin = Decimal(str(origin))
    d_step = Decimal(str(step))
    d_x = Decimal(str(x))
    units = ((d_x - d_origin) / d_step).to_integral_value(rounding=ROUND_FLOOR)
    return float(units * d_step + d_origin)

def snap_ceil(x: float, origin: float, step: float = 0.2) -> float:
    d_origin = Decimal(str(origin))
    d_step = Decimal(str(step))
    d_x = Decimal(str(x))
    units = ((d_x - d_origin) / d_step).to_integral_value(rounding=ROUND_CEILING)
    return float(units * d_step + d_origin)

def transform_bbox_wgs84_to_utm(
    lon_min: float, lat_min: float, lon_max: float, lat_max: float,
    densify_pts: int = 21,
) -> tuple[float, float, float, float]:
    t = _transformer_4326_to_25832()
    return t.transform_bounds(lon_min, lat_min, lon_max, lat_max, densify_pts=densify_pts)

def transform_bbox_utm_to_wgs84(
    minx: float, miny: float, maxx: float, maxy: float,
    densify_pts: int = 21,
) -> tuple[float, float, float, float]:
    t = _transformer_25832_to_4326()
    return t.transform_bounds(minx, miny, maxx, maxy, densify_pts=densify_pts)

def pixel_count(minx: float, maxx: float, step: float = 0.2) -> int:
    """Robust against IEEE-754 FP: uses round() per Spec §5.1 pt 6."""
    return round((maxx - minx) / step)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/pipeline/test_geo_utils.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add ki_geodaten/pipeline/geo_utils.py tests/pipeline/test_geo_utils.py
git commit -m "feat(pipeline): add Decimal-based snap utilities and transform_bounds wrapper"
```

---

## Task 5: SQLite Store — Schema & Connection Factory

**Files:**
- Create: `ki_geodaten/jobs/store.py`
- Test: `tests/jobs/test_store.py`

**See Spec §7** — full schema, PRAGMAs, WKB rationale.

- [ ] **Step 1: Write failing test**

```python
# tests/jobs/test_store.py
import pytest
import sqlite3
from pathlib import Path
from ki_geodaten.jobs.store import connect, init_schema, insert_job, get_job, update_status
from ki_geodaten.models import JobStatus, TilePreset

def test_init_schema_creates_tables(tmp_path: Path):
    db = tmp_path / "t.db"
    init_schema(db)
    with connect(db) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    names = [r[0] for r in rows]
    assert "jobs" in names
    assert "polygons" in names
    assert "nodata_regions" in names

def test_connect_sets_wal_mode(tmp_path):
    db = tmp_path / "t.db"
    init_schema(db)
    with connect(db) as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    assert mode.lower() == "wal"
    assert fk == 1

def test_insert_and_get_job(tmp_path):
    db = tmp_path / "t.db"
    init_schema(db)
    jid = "00000000-0000-0000-0000-000000000001"
    insert_job(
        db, job_id=jid, prompt="building",
        bbox_wgs84=[11.0, 48.0, 11.1, 48.1],
        bbox_utm_snapped=[691000.0, 5335000.0, 692000.0, 5336000.0],
        tile_preset=TilePreset.MEDIUM,
    )
    job = get_job(db, jid)
    assert job["status"] == JobStatus.PENDING
    assert job["prompt"] == "building"
    assert job["validation_revision"] == 0
    assert job["exported_revision"] is None

def test_status_check_constraint_rejects_unknown(tmp_path):
    db = tmp_path / "t.db"
    init_schema(db)
    with connect(db) as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO jobs(id,prompt,bbox_wgs84,bbox_utm_snapped,tile_preset,status,created_at)"
                " VALUES ('x','p','[]','[]','medium','WEIRD','2026-01-01')"
            )

def test_cascade_delete(tmp_path):
    db = tmp_path / "t.db"
    init_schema(db)
    jid = "j1"
    insert_job(db, job_id=jid, prompt="p", bbox_wgs84=[0,0,1,1],
               bbox_utm_snapped=[0,0,1,1], tile_preset=TilePreset.MEDIUM)
    with connect(db) as conn:
        conn.execute(
            "INSERT INTO polygons(job_id,geometry_wkb,score,source_tile_row,source_tile_col)"
            " VALUES (?,?,?,?,?)", (jid, b"\x00", 0.5, 0, 0))
        conn.commit()
    with connect(db) as conn:
        conn.execute("DELETE FROM jobs WHERE id=?", (jid,))
        conn.commit()
        n = conn.execute("SELECT COUNT(*) FROM polygons").fetchone()[0]
    assert n == 0
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/jobs/test_store.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `store.py`**

```python
# ki_geodaten/jobs/store.py
from __future__ import annotations
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from ki_geodaten.models import JobStatus, TilePreset

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id                TEXT PRIMARY KEY,
    prompt            TEXT NOT NULL,
    bbox_wgs84        TEXT NOT NULL,
    bbox_utm_snapped  TEXT NOT NULL,
    tile_preset       TEXT NOT NULL CHECK (tile_preset IN ('small','medium','large')),
    status            TEXT NOT NULL CHECK (status IN (
                          'PENDING','DOWNLOADING','INFERRING',
                          'READY_FOR_REVIEW','EXPORTED','FAILED')),
    error_reason      TEXT CHECK (error_reason IS NULL OR error_reason IN (
                          'WCS_TIMEOUT','WCS_HTTP_ERROR','OOM',
                          'INFERENCE_ERROR','WORKER_RESTARTED',
                          'EXPORT_ERROR','INVALID_GEOMETRY')),
    error_message     TEXT,
    dop_vrt_path      TEXT,
    gpkg_path         TEXT,
    tile_total        INTEGER,
    tile_completed    INTEGER NOT NULL DEFAULT 0,
    tile_failed       INTEGER NOT NULL DEFAULT 0,
    validation_revision INTEGER NOT NULL DEFAULT 0,
    exported_revision   INTEGER,
    created_at        TEXT NOT NULL,
    started_at        TEXT,
    finished_at       TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at);

CREATE TABLE IF NOT EXISTS polygons (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id           TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    geometry_wkb     BLOB NOT NULL,
    score            REAL NOT NULL,
    source_tile_row  INTEGER NOT NULL,
    source_tile_col  INTEGER NOT NULL,
    validation       TEXT NOT NULL DEFAULT 'ACCEPTED' CHECK (validation IN ('ACCEPTED','REJECTED'))
);
CREATE INDEX IF NOT EXISTS idx_polygons_job ON polygons(job_id);

CREATE TABLE IF NOT EXISTS nodata_regions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id           TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    geometry_wkb     BLOB NOT NULL,
    tile_row         INTEGER NOT NULL,
    tile_col         INTEGER NOT NULL,
    reason           TEXT NOT NULL CHECK (reason IN (
                          'OOM','INFERENCE_ERROR','INVALID_GEOMETRY','NODATA_PIXELS'))
);
CREATE INDEX IF NOT EXISTS idx_nodata_job ON nodata_regions(job_id);
"""

def _apply_pragmas(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")

@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(db_path), isolation_level=None, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        _apply_pragmas(conn)
        yield conn
    finally:
        conn.close()

def init_schema(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)

def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def insert_job(
    db_path: Path, *, job_id: str, prompt: str,
    bbox_wgs84: list[float], bbox_utm_snapped: list[float],
    tile_preset: TilePreset,
) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO jobs(id,prompt,bbox_wgs84,bbox_utm_snapped,tile_preset,status,created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (job_id, prompt, json.dumps(bbox_wgs84), json.dumps(bbox_utm_snapped),
             str(tile_preset), JobStatus.PENDING, _utc_iso()),
        )

def get_job(db_path: Path, job_id: str) -> dict | None:
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return dict(row) if row else None

def update_status(
    db_path: Path, job_id: str, status: JobStatus,
    *, error_reason: str | None = None, error_message: str | None = None,
    dop_vrt_path: str | None = None, gpkg_path: str | None = None,
    tile_total: int | None = None, exported_revision: int | None = None,
    set_started: bool = False, set_finished: bool = False,
) -> None:
    fields = ["status = ?"]
    params: list = [str(status)]
    if error_reason is not None:
        fields.append("error_reason = ?"); params.append(error_reason)
    if error_message is not None:
        fields.append("error_message = ?"); params.append(error_message)
    if dop_vrt_path is not None:
        fields.append("dop_vrt_path = ?"); params.append(dop_vrt_path)
    if gpkg_path is not None:
        fields.append("gpkg_path = ?"); params.append(gpkg_path)
    if tile_total is not None:
        fields.append("tile_total = ?"); params.append(tile_total)
    if exported_revision is not None:
        fields.append("exported_revision = ?"); params.append(exported_revision)
    if set_started:
        fields.append("started_at = ?"); params.append(_utc_iso())
    if set_finished:
        fields.append("finished_at = ?"); params.append(_utc_iso())
    params.append(job_id)
    with connect(db_path) as conn:
        conn.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?", params)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/jobs/test_store.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add ki_geodaten/jobs/store.py tests/jobs/test_store.py
git commit -m "feat(jobs): add SQLite schema, WAL setup, and core CRUD"
```

---

## Task 6: Store — Polygon & NoData Persistence + Progress

**Files:**
- Modify: `ki_geodaten/jobs/store.py`
- Test: `tests/jobs/test_store.py` (append)

- [ ] **Step 1: Append failing tests**

```python
# tests/jobs/test_store.py (append)
from ki_geodaten.jobs.store import (
    insert_polygons, insert_nodata_region, increment_tile_completed,
    increment_tile_failed, get_polygons_for_job, get_nodata_for_job,
    validate_bulk,
)

def test_insert_polygons_and_fetch(tmp_path):
    db = tmp_path / "t.db"
    init_schema(db)
    jid = "j1"
    insert_job(db, job_id=jid, prompt="p", bbox_wgs84=[0,0,1,1],
               bbox_utm_snapped=[0,0,1,1], tile_preset=TilePreset.MEDIUM)
    insert_polygons(db, jid, [
        {"geometry_wkb": b"\x01\x02", "score": 0.9, "source_tile_row": 0, "source_tile_col": 0},
        {"geometry_wkb": b"\x03\x04", "score": 0.5, "source_tile_row": 0, "source_tile_col": 1},
    ])
    rows = get_polygons_for_job(db, jid)
    assert len(rows) == 2
    assert all(r["validation"] == "ACCEPTED" for r in rows)

def test_validate_bulk_updates_revision_and_ignores_unknown_pids(tmp_path):
    db = tmp_path / "t.db"
    init_schema(db)
    jid = "j1"
    insert_job(db, job_id=jid, prompt="p", bbox_wgs84=[0,0,1,1],
               bbox_utm_snapped=[0,0,1,1], tile_preset=TilePreset.MEDIUM)
    insert_polygons(db, jid, [
        {"geometry_wkb": b"a", "score": 0.9, "source_tile_row": 0, "source_tile_col": 0},
        {"geometry_wkb": b"b", "score": 0.8, "source_tile_row": 0, "source_tile_col": 1},
    ])
    updates = [{"pid": 1, "validation": "REJECTED"}, {"pid": 9999, "validation": "REJECTED"}]
    updated = validate_bulk(db, jid, updates)
    assert updated == 1
    job = get_job(db, jid)
    assert job["validation_revision"] == 1

def test_validate_bulk_handles_many_updates(tmp_path):
    # Regression: executemany must not hit SQLITE_MAX_VARIABLE_NUMBER
    db = tmp_path / "t.db"
    init_schema(db)
    jid = "j1"
    insert_job(db, job_id=jid, prompt="p", bbox_wgs84=[0,0,1,1],
               bbox_utm_snapped=[0,0,1,1], tile_preset=TilePreset.MEDIUM)
    insert_polygons(db, jid, [
        {"geometry_wkb": b"x", "score": 0.1, "source_tile_row": 0, "source_tile_col": 0}
        for _ in range(1500)
    ])
    updates = [{"pid": i + 1, "validation": "REJECTED"} for i in range(1500)]
    updated = validate_bulk(db, jid, updates)
    assert updated == 1500

def test_increment_tile_completed(tmp_path):
    db = tmp_path / "t.db"
    init_schema(db)
    jid = "j1"
    insert_job(db, job_id=jid, prompt="p", bbox_wgs84=[0,0,1,1],
               bbox_utm_snapped=[0,0,1,1], tile_preset=TilePreset.MEDIUM)
    increment_tile_completed(db, jid)
    increment_tile_completed(db, jid)
    increment_tile_failed(db, jid)
    job = get_job(db, jid)
    assert job["tile_completed"] == 2
    assert job["tile_failed"] == 1

def test_insert_nodata_region(tmp_path):
    db = tmp_path / "t.db"
    init_schema(db)
    jid = "j1"
    insert_job(db, job_id=jid, prompt="p", bbox_wgs84=[0,0,1,1],
               bbox_utm_snapped=[0,0,1,1], tile_preset=TilePreset.MEDIUM)
    insert_nodata_region(db, jid, geometry_wkb=b"\x00", tile_row=1, tile_col=2, reason="OOM")
    rows = get_nodata_for_job(db, jid)
    assert len(rows) == 1
    assert rows[0]["reason"] == "OOM"
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/jobs/test_store.py -v`
Expected: new tests FAIL (AttributeError: missing functions).

- [ ] **Step 3: Append implementation to `store.py`**

```python
# ki_geodaten/jobs/store.py (append)

def insert_polygons(db_path: Path, job_id: str, polys: list[dict]) -> None:
    if not polys:
        return
    with connect(db_path) as conn:
        conn.execute("BEGIN")
        conn.executemany(
            "INSERT INTO polygons(job_id,geometry_wkb,score,source_tile_row,source_tile_col)"
            " VALUES (?,?,?,?,?)",
            [(job_id, p["geometry_wkb"], p["score"],
              p["source_tile_row"], p["source_tile_col"]) for p in polys],
        )
        conn.execute("COMMIT")

def insert_nodata_region(
    db_path: Path, job_id: str, *, geometry_wkb: bytes,
    tile_row: int, tile_col: int, reason: str,
) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO nodata_regions(job_id,geometry_wkb,tile_row,tile_col,reason)"
            " VALUES (?,?,?,?,?)",
            (job_id, geometry_wkb, tile_row, tile_col, reason),
        )

def increment_tile_completed(db_path: Path, job_id: str) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE jobs SET tile_completed = tile_completed + 1 WHERE id = ?",
            (job_id,),
        )

def increment_tile_failed(db_path: Path, job_id: str) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE jobs SET tile_failed = tile_failed + 1 WHERE id = ?",
            (job_id,),
        )

def get_polygons_for_job(db_path: Path, job_id: str) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id,geometry_wkb,score,source_tile_row,source_tile_col,validation"
            " FROM polygons WHERE job_id = ?",
            (job_id,),
        ).fetchall()
    return [dict(r) for r in rows]

def get_nodata_for_job(db_path: Path, job_id: str) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id,geometry_wkb,tile_row,tile_col,reason"
            " FROM nodata_regions WHERE job_id = ?",
            (job_id,),
        ).fetchall()
    return [dict(r) for r in rows]

def validate_bulk(db_path: Path, job_id: str, updates: list[dict]) -> int:
    """executemany-based bulk update per Spec §8. Increments validation_revision."""
    if not updates:
        return 0
    with connect(db_path) as conn:
        conn.execute("BEGIN")
        cur = conn.cursor()
        cur.executemany(
            "UPDATE polygons SET validation = ? WHERE id = ? AND job_id = ?",
            [(u["validation"], u["pid"], job_id) for u in updates],
        )
        updated = cur.rowcount
        conn.execute(
            "UPDATE jobs SET validation_revision = validation_revision + 1 WHERE id = ?",
            (job_id,),
        )
        conn.execute("COMMIT")
    return max(updated, 0)

def list_jobs(db_path: Path) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]

def claim_next_pending_job(db_path: Path) -> dict | None:
    """Atomic PENDING→DOWNLOADING claim."""
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM jobs WHERE status='PENDING' ORDER BY created_at ASC LIMIT 1"
        ).fetchone()
        if row is None:
            conn.execute("COMMIT")
            return None
        conn.execute(
            "UPDATE jobs SET status='DOWNLOADING', started_at=? WHERE id=?",
            (_utc_iso(), row["id"]),
        )
        conn.execute("COMMIT")
    return dict(row)

def abort_incomplete_jobs_on_startup(db_path: Path) -> list[str]:
    """Per Spec §10.3 — mark DOWNLOADING/INFERRING as FAILED(WORKER_RESTARTED)."""
    with connect(db_path) as conn:
        conn.execute("BEGIN")
        rows = conn.execute(
            "SELECT id FROM jobs WHERE status IN ('DOWNLOADING','INFERRING')"
        ).fetchall()
        ids = [r["id"] for r in rows]
        conn.execute(
            "UPDATE jobs SET status='FAILED', error_reason='WORKER_RESTARTED', finished_at=?"
            " WHERE status IN ('DOWNLOADING','INFERRING')",
            (_utc_iso(),),
        )
        conn.execute("COMMIT")
    return ids
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/jobs/test_store.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add ki_geodaten/jobs/store.py tests/jobs/test_store.py
git commit -m "feat(jobs): polygon/nodata persistence, bulk validate, claim, startup cleanup"
```

---

## Task 7: WCS Client — BBox Prep (Snap + Margin + Min-Size)

**Files:**
- Create: `ki_geodaten/pipeline/wcs_client.py`
- Test: `tests/pipeline/test_wcs_client.py`

**See Spec §5.1 pts 1, 4, 5, 6** — snapping, margin expansion, minimum size, pixel-count via round().

Split of concerns: this task implements only the pure geometry math; HTTP comes next task.

- [ ] **Step 1: Write failing test**

```python
# tests/pipeline/test_wcs_client.py
import pytest
from ki_geodaten.pipeline.wcs_client import (
    prepare_download_bbox, WCSError, PreparedBBox,
)
from ki_geodaten.models import TilePreset

def test_prepare_expands_by_center_margin_medium():
    # 1 km box, aligned to 0.2m grid already
    result = prepare_download_bbox(
        minx=691000.0, miny=5335000.0, maxx=692000.0, maxy=5336000.0,
        preset=TilePreset.MEDIUM, origin_x=0.0, origin_y=0.0, step=0.2,
    )
    # medium = center_margin 320 px * 0.2 m = 64 m
    assert result.download_bbox == (690936.0, 5334936.0, 692064.0, 5336064.0)
    assert result.aoi_bbox == (691000.0, 5335000.0, 692000.0, 5336000.0)

def test_prepare_expands_by_center_margin_small():
    # small preset: center_margin 160 px * 0.2 m = 32 m
    result = prepare_download_bbox(
        minx=691000.0, miny=5335000.0, maxx=692000.0, maxy=5336000.0,
        preset=TilePreset.SMALL, origin_x=0.0, origin_y=0.0, step=0.2,
    )
    assert result.download_bbox == (690968.0, 5334968.0, 692032.0, 5336032.0)

def test_prepare_expands_by_center_margin_large():
    # large preset: center_margin 480 px * 0.2 m = 96 m
    result = prepare_download_bbox(
        minx=691000.0, miny=5335000.0, maxx=692000.0, maxy=5336000.0,
        preset=TilePreset.LARGE, origin_x=0.0, origin_y=0.0, step=0.2,
    )
    assert result.download_bbox == (690904.0, 5334904.0, 692096.0, 5336096.0)

def test_prepare_minimum_size_expands_to_204_8():
    # 50 m × 50 m tiny box → expand symmetrically to ≥ 204.8 m
    result = prepare_download_bbox(
        minx=691000.0, miny=5335000.0, maxx=691050.0, maxy=5335050.0,
        preset=TilePreset.MEDIUM, origin_x=0.0, origin_y=0.0, step=0.2,
    )
    dx = result.download_bbox[2] - result.download_bbox[0]
    dy = result.download_bbox[3] - result.download_bbox[1]
    assert dx >= 204.8
    assert dy >= 204.8

def test_prepare_snaps_unaligned_input():
    # Input 691000.15 → snap_floor to 691000.0, snap_ceil remains
    result = prepare_download_bbox(
        minx=691000.15, miny=5335000.07, maxx=692000.05, maxy=5336000.11,
        preset=TilePreset.MEDIUM, origin_x=0.0, origin_y=0.0, step=0.2,
    )
    # aoi is snapped (before expansion)
    minx, miny, maxx, maxy = result.aoi_bbox
    assert (minx * 10) % 2 == 0
    assert (maxy * 10) % 2 == 0

def test_prepare_with_nonzero_origin():
    result = prepare_download_bbox(
        minx=691000.1, miny=5335000.1, maxx=691500.1, maxy=5335500.1,
        preset=TilePreset.MEDIUM, origin_x=0.1, origin_y=0.1, step=0.2,
    )
    # After snapping all coords should be congruent to 0.1 mod 0.2
    for c in result.aoi_bbox:
        assert abs(((c - 0.1) / 0.2) - round((c - 0.1) / 0.2)) < 1e-9
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/pipeline/test_wcs_client.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `wcs_client.py` (geometry only; HTTP is Task 8)**

```python
# ki_geodaten/pipeline/wcs_client.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from ki_geodaten.models import TilePreset
from ki_geodaten.pipeline.geo_utils import snap_floor, snap_ceil, pixel_count

STEP = 0.2
TILE_SIZE = 1024  # must match config.TILE_SIZE
MIN_BBOX_SIDE_M = TILE_SIZE * STEP  # = 204.8

# Center-margin per preset (from Spec §5.2 table), in PIXELS:
CENTER_MARGIN_PX: dict[TilePreset, int] = {
    TilePreset.SMALL: 160,
    TilePreset.MEDIUM: 320,
    TilePreset.LARGE: 480,
}

class WCSError(Exception):
    pass

@dataclass(frozen=True)
class PreparedBBox:
    aoi_bbox: tuple[float, float, float, float]       # grid-snapped user AOI (persisted)
    download_bbox: tuple[float, float, float, float]  # aoi + margin + min-size (used for HTTP)

def _snap_bbox(
    minx: float, miny: float, maxx: float, maxy: float,
    origin_x: float, origin_y: float, step: float,
) -> tuple[float, float, float, float]:
    return (
        snap_floor(minx, origin_x, step),
        snap_floor(miny, origin_y, step),
        snap_ceil(maxx, origin_x, step),
        snap_ceil(maxy, origin_y, step),
    )

def _expand_symmetric(
    minx: float, miny: float, maxx: float, maxy: float,
    target_side: float, origin_x: float, origin_y: float, step: float,
) -> tuple[float, float, float, float]:
    dx = maxx - minx
    dy = maxy - miny
    if dx < target_side:
        extra = (target_side - dx) / 2
        minx = snap_floor(minx - extra, origin_x, step)
        maxx = snap_ceil(maxx + extra, origin_x, step)
    if dy < target_side:
        extra = (target_side - dy) / 2
        miny = snap_floor(miny - extra, origin_y, step)
        maxy = snap_ceil(maxy + extra, origin_y, step)
    return (minx, miny, maxx, maxy)

def prepare_download_bbox(
    minx: float, miny: float, maxx: float, maxy: float,
    *, preset: TilePreset, origin_x: float, origin_y: float, step: float = STEP,
) -> PreparedBBox:
    """Spec §5.1 pts 1, 4, 5. Output AOI is user-facing (persisted);
    download_bbox adds CENTER_MARGIN and enforces min side ≥ TILE_SIZE*step."""
    aoi = _snap_bbox(minx, miny, maxx, maxy, origin_x, origin_y, step)
    margin_m = CENTER_MARGIN_PX[preset] * step
    dmin_x = snap_floor(aoi[0] - margin_m, origin_x, step)
    dmin_y = snap_floor(aoi[1] - margin_m, origin_y, step)
    dmax_x = snap_ceil(aoi[2] + margin_m, origin_x, step)
    dmax_y = snap_ceil(aoi[3] + margin_m, origin_y, step)
    dmin_x, dmin_y, dmax_x, dmax_y = _expand_symmetric(
        dmin_x, dmin_y, dmax_x, dmax_y,
        MIN_BBOX_SIDE_M, origin_x, origin_y, step,
    )
    return PreparedBBox(aoi_bbox=aoi, download_bbox=(dmin_x, dmin_y, dmax_x, dmax_y))
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/pipeline/test_wcs_client.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add ki_geodaten/pipeline/wcs_client.py tests/pipeline/test_wcs_client.py
git commit -m "feat(wcs): BBox prep with grid-snap, margin expansion, min-size"
```

---

## Task 8: WCS Client — Chunk Grid & HTTP Download + VRT

**Files:**
- Modify: `ki_geodaten/pipeline/wcs_client.py`
- Test: `tests/pipeline/test_wcs_client.py` (append)

**See Spec §5.1 pts 2, 3, 6, 7, 8** — pagination grid, HTTP with retry, round() for pixels, contiguous edges, BuildVRT.

- [ ] **Step 1: Append failing tests**

```python
# tests/pipeline/test_wcs_client.py (append)
from ki_geodaten.pipeline.wcs_client import plan_chunk_grid, download_dop20

def test_chunk_grid_seamless_edges():
    # 2000m × 2000m box, MAX_WCS_PIXELS=4000 → chunk side 4000*0.2 = 800m
    chunks = plan_chunk_grid(
        minx=0.0, miny=0.0, maxx=2000.0, maxy=2000.0,
        max_pixels=4000, step=0.2, origin_x=0.0, origin_y=0.0,
    )
    # Expect 3×3 grid (800 + 800 + 400 remainder)
    assert len(chunks) == 9
    # All chunks aligned to 0.2 grid
    for c in chunks:
        for v in (c.minx, c.miny, c.maxx, c.maxy):
            assert abs(round(v / 0.2) * 0.2 - v) < 1e-9
    # Neighbours share identical edges (no gap, no overlap)
    by_row = {}
    for c in chunks:
        by_row.setdefault(c.row, []).append(c)
    for row_chunks in by_row.values():
        row_chunks.sort(key=lambda c: c.col)
        for a, b in zip(row_chunks, row_chunks[1:]):
            assert a.maxx == b.minx

def test_chunk_grid_small_bbox_single_chunk():
    chunks = plan_chunk_grid(
        minx=0.0, miny=0.0, maxx=204.8, maxy=204.8,
        max_pixels=4000, step=0.2, origin_x=0.0, origin_y=0.0,
    )
    assert len(chunks) == 1

def test_download_dop20_http_success(tmp_path, responses):
    # responses fixture via pytest-responses; mock WCS GetCoverage
    # Implementation uses requests, so responses activates automatic mocking.
    # We emit a valid minimal GeoTIFF blob per call.
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds

    def make_tif(bbox):
        path = tmp_path / "fake.tif"
        w = round((bbox[2] - bbox[0]) / 0.2)
        h = round((bbox[3] - bbox[1]) / 0.2)
        with rasterio.open(
            path, "w", driver="GTiff", height=h, width=w,
            count=3, dtype="uint8", crs="EPSG:25832",
            transform=from_bounds(*bbox, w, h),
        ) as dst:
            dst.write(np.zeros((3, h, w), dtype="uint8"))
        return path.read_bytes()

    bbox = (0.0, 0.0, 204.8, 204.8)
    responses.add(
        responses.GET,
        "http://example/wcs",
        body=make_tif(bbox),
        status=200,
        content_type="image/tiff",
    )

    vrt_path = download_dop20(
        bbox_utm=bbox, preset_center_margin_px=320, out_dir=tmp_path,
        wcs_url="http://example/wcs", coverage_id="cov1",
        max_pixels=4000, origin_x=0.0, origin_y=0.0,
    )
    assert vrt_path.exists()
    assert vrt_path.suffix == ".vrt"
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/pipeline/test_wcs_client.py -v`
Expected: new tests FAIL.

- [ ] **Step 3: Append implementation**

```python
# ki_geodaten/pipeline/wcs_client.py (append)
from dataclasses import dataclass
import logging
from osgeo import gdal
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ChunkPlan:
    row: int
    col: int
    minx: float
    miny: float
    maxx: float
    maxy: float

    def width_px(self, step: float = STEP) -> int:
        return pixel_count(self.minx, self.maxx, step)

    def height_px(self, step: float = STEP) -> int:
        return pixel_count(self.miny, self.maxy, step)

def plan_chunk_grid(
    minx: float, miny: float, maxx: float, maxy: float,
    *, max_pixels: int, step: float, origin_x: float, origin_y: float,
) -> list[ChunkPlan]:
    chunk_m = max_pixels * step
    chunks: list[ChunkPlan] = []
    y0 = miny
    row = 0
    while y0 < maxy:
        y1 = min(snap_ceil(y0 + chunk_m, origin_y, step), maxy)
        x0 = minx
        col = 0
        while x0 < maxx:
            x1 = min(snap_ceil(x0 + chunk_m, origin_x, step), maxx)
            chunks.append(ChunkPlan(row=row, col=col, minx=x0, miny=y0, maxx=x1, maxy=y1))
            x0 = x1
            col += 1
        y0 = y1
        row += 1
    return chunks

def _build_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=3, backoff_factor=2.0,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

def _fetch_chunk(
    session: requests.Session, wcs_url: str, coverage_id: str,
    chunk: ChunkPlan, out_path: Path,
) -> None:
    """WCS 2.0 GetCoverage; axis order per Spec §15 offene Punkte — verify at first call."""
    w = chunk.width_px()
    h = chunk.height_px()
    params = {
        "service": "WCS",
        "version": "2.0.1",
        "request": "GetCoverage",
        "coverageId": coverage_id,
        "format": "image/tiff",
        "subset": [
            f"E({chunk.minx},{chunk.maxx})",
            f"N({chunk.miny},{chunk.maxy})",
        ],
        "subsettingcrs": "http://www.opengis.net/def/crs/EPSG/0/25832",
        "outputcrs":    "http://www.opengis.net/def/crs/EPSG/0/25832",
        "size": [f"E({w})", f"N({h})"],
    }
    try:
        r = session.get(wcs_url, params=params, timeout=(10, 60))
        r.raise_for_status()
    except requests.exceptions.Timeout as e:
        raise WCSError("WCS_TIMEOUT") from e
    except requests.exceptions.RequestException as e:
        raise WCSError("WCS_HTTP_ERROR") from e
    out_path.write_bytes(r.content)

def download_dop20(
    bbox_utm: tuple[float, float, float, float],
    preset_center_margin_px: int, out_dir: Path,
    *, wcs_url: str, coverage_id: str, max_pixels: int,
    origin_x: float, origin_y: float,
) -> Path:
    """Spec §5.1 signature. bbox_utm MUST be the already-prepared download_bbox."""
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = plan_chunk_grid(
        *bbox_utm, max_pixels=max_pixels, step=STEP,
        origin_x=origin_x, origin_y=origin_y,
    )
    session = _build_session()
    chunk_files: list[str] = []
    for c in chunks:
        p = out_dir / f"chunk_{c.row}_{c.col}.tif"
        _fetch_chunk(session, wcs_url, coverage_id, c, p)
        chunk_files.append(str(p))
    vrt_path = out_dir / "out.vrt"
    vrt = gdal.BuildVRT(str(vrt_path), chunk_files)
    if vrt is None:
        raise WCSError("VRT_BUILD_FAILED")
    vrt.FlushCache()
    return vrt_path
```

Note: `responses` fixture requires `pytest-responses` — replace with `responses.activate` decorator if needed. If `responses` cannot register params, use callback-based matcher. The alternative test technique is to replace `_fetch_chunk` with a monkeypatch'd stub; the plan permits either, as long as VRT output is verified.

- [ ] **Step 4: Run tests**

Run: `pytest tests/pipeline/test_wcs_client.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add ki_geodaten/pipeline/wcs_client.py tests/pipeline/test_wcs_client.py
git commit -m "feat(wcs): paginated chunk grid, HTTP retry, VRT build"
```

---

## Task 9: Tiler — `TileConfig` & `Tile` Dataclass

**Files:**
- Create: `ki_geodaten/pipeline/tiler.py`
- Test: `tests/pipeline/test_tiler.py`

**See Spec §5.2** — preset table (OVERLAP=320/640/960, CENTER_MARGIN=160/320/480), invariants.

- [ ] **Step 1: Write failing test**

```python
# tests/pipeline/test_tiler.py
import pytest
from ki_geodaten.pipeline.tiler import TileConfig
from ki_geodaten.models import TilePreset

def test_tile_config_small():
    cfg = TileConfig.from_preset(TilePreset.SMALL)
    assert cfg.size == 1024 and cfg.overlap == 320 and cfg.center_margin == 160
    assert cfg.tile_step == 704
    assert cfg.safe_center_size == 704

def test_tile_config_medium():
    cfg = TileConfig.from_preset(TilePreset.MEDIUM)
    assert cfg.overlap == 640 and cfg.center_margin == 320
    assert cfg.tile_step == 384 and cfg.safe_center_size == 384

def test_tile_config_large():
    cfg = TileConfig.from_preset(TilePreset.LARGE)
    assert cfg.overlap == 960 and cfg.center_margin == 480
    assert cfg.tile_step == 64 and cfg.safe_center_size == 64

def test_tile_config_invariants():
    for preset in TilePreset:
        cfg = TileConfig.from_preset(preset)
        assert cfg.tile_step == cfg.safe_center_size
        assert cfg.center_margin * 2 == cfg.overlap
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/pipeline/test_tiler.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement**

```python
# ki_geodaten/pipeline/tiler.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal
import numpy as np
import rasterio
from rasterio.windows import Window
from affine import Affine

from ki_geodaten.models import TilePreset

_PRESET_PARAMS: dict[TilePreset, tuple[int, int]] = {
    TilePreset.SMALL:  (320, 160),
    TilePreset.MEDIUM: (640, 320),
    TilePreset.LARGE:  (960, 480),
}

@dataclass(frozen=True)
class TileConfig:
    size: int
    overlap: int
    center_margin: int

    @property
    def tile_step(self) -> int:
        return self.size - self.overlap

    @property
    def safe_center_size(self) -> int:
        return self.size - 2 * self.center_margin

    @classmethod
    def from_preset(cls, preset: TilePreset | str) -> "TileConfig":
        p = TilePreset(str(preset))
        overlap, margin = _PRESET_PARAMS[p]
        return cls(size=1024, overlap=overlap, center_margin=margin)

@dataclass(frozen=True)
class Tile:
    array: np.ndarray
    pixel_origin: tuple[int, int]
    size: int
    center_margin: int
    affine: Affine
    tile_row: int
    tile_col: int
    nodata_mask: np.ndarray
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/pipeline/test_tiler.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ki_geodaten/pipeline/tiler.py tests/pipeline/test_tiler.py
git commit -m "feat(tiler): TileConfig presets and Tile dataclass"
```

---

## Task 10: Tiler — `iter_grid` with Edge-Shift + Unique Indices

**Files:**
- Modify: `ki_geodaten/pipeline/tiler.py`
- Test: `tests/pipeline/test_tiler.py` (append)

**See Spec §5.2** — edge tile shifted inward; `tile_row`/`tile_col` are logical grid indices carried explicitly (NOT recomputed from offsets).

- [ ] **Step 1: Append failing tests**

```python
# tests/pipeline/test_tiler.py (append)
from ki_geodaten.pipeline.tiler import iter_grid

class FakeSrc:
    def __init__(self, width, height):
        self.width = width
        self.height = height

def test_iter_grid_exact_fit_medium():
    grid = list(iter_grid(FakeSrc(1408, 1408), TileConfig.from_preset(TilePreset.MEDIUM)))
    indices = {(r, c, ro, co) for r, c, ro, co in grid}
    assert len(indices) == 4

def test_iter_grid_edge_shifts_last_tile_inward():
    grid = list(iter_grid(FakeSrc(1500, 1024), TileConfig.from_preset(TilePreset.MEDIUM)))
    col_offsets = sorted({co for _, _, _, co in grid})
    assert col_offsets[-1] == 1500 - 1024
    last_tile = max(grid, key=lambda g: g[3])
    assert last_tile[3] + 1024 == 1500

def test_iter_grid_unique_logical_indices():
    grid = list(iter_grid(FakeSrc(1500, 1500), TileConfig.from_preset(TilePreset.MEDIUM)))
    indices = [(r, c) for r, c, _, _ in grid]
    assert len(indices) == len(set(indices))

def test_iter_grid_smaller_than_tile_single_tile():
    grid = list(iter_grid(FakeSrc(1024, 1024), TileConfig.from_preset(TilePreset.MEDIUM)))
    assert grid == [(0, 0, 0, 0)]
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/pipeline/test_tiler.py -v`
Expected: new tests FAIL.

- [ ] **Step 3: Append implementation**

```python
# ki_geodaten/pipeline/tiler.py (append)

def _axis_offsets(extent: int, size: int, step: int) -> list[int]:
    if extent <= size:
        return [0]
    offs = list(range(0, extent - size + 1, step))
    last = extent - size
    if offs[-1] != last:
        offs.append(last)
    return offs

def iter_grid(src, cfg: TileConfig) -> Iterator[tuple[int, int, int, int]]:
    """Yields (tile_row, tile_col, row_off, col_off). Logical indices are carried
    explicitly per Spec §5.2 so edge-shifted tiles retain a unique grid cell."""
    col_offsets = _axis_offsets(src.width, cfg.size, cfg.tile_step)
    row_offsets = _axis_offsets(src.height, cfg.size, cfg.tile_step)
    for r, ro in enumerate(row_offsets):
        for c, co in enumerate(col_offsets):
            yield (r, c, ro, co)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/pipeline/test_tiler.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add ki_geodaten/pipeline/tiler.py tests/pipeline/test_tiler.py
git commit -m "feat(tiler): iter_grid with edge-shift and unique logical indices"
```

---

## Task 11: Tiler — `iter_tiles` with Per-Tile Affine + NoData Detection

**Files:**
- Modify: `ki_geodaten/pipeline/tiler.py`
- Test: `tests/pipeline/test_tiler.py` (append)

**See Spec §5.2** — `src.window_transform(window)` zwingend, raster-mask-based NoData detection in safe center.

- [ ] **Step 1: Append failing tests**

```python
# tests/pipeline/test_tiler.py (append)
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from ki_geodaten.pipeline.tiler import iter_tiles, NodataTile

def _make_rgb_tif(path, w, h, bbox, nodata_mask=None, bands=3):
    with rasterio.open(
        path, "w", driver="GTiff", width=w, height=h, count=bands,
        dtype="uint8", crs="EPSG:25832", transform=from_bounds(*bbox, w, h),
        nodata=0,
    ) as dst:
        for b in range(1, bands + 1):
            arr = np.full((h, w), 128, dtype="uint8")
            if nodata_mask is not None:
                arr[nodata_mask] = 0
            dst.write(arr, b)
    return path

def test_iter_tiles_per_tile_affine(tmp_path):
    p = tmp_path / "fake.tif"
    _make_rgb_tif(p, 2048, 2048, (691000.0, 5335000.0, 691409.6, 5335409.6))
    cfg = TileConfig.from_preset(TilePreset.MEDIUM)
    tiles = list(iter_tiles(p, cfg))
    assert len(tiles) >= 1
    t00 = tiles[0]
    assert t00.affine.a == pytest.approx(0.2)
    assert t00.affine.c == pytest.approx(691000.0)
    assert t00.array.shape == (1024, 1024, 3)
    assert t00.array.dtype == np.uint8

def test_iter_tiles_reads_only_rgb_from_4band(tmp_path):
    p = tmp_path / "4band.tif"
    _make_rgb_tif(p, 1024, 1024, (0, 0, 204.8, 204.8), bands=4)
    cfg = TileConfig.from_preset(TilePreset.MEDIUM)
    tiles = list(iter_tiles(p, cfg))
    assert tiles[0].array.shape == (1024, 1024, 3)
    assert tiles[0].nodata_mask.shape == (1024, 1024)

def test_iter_tiles_flags_nodata_in_safe_center(tmp_path):
    w = h = 1024
    mask = np.zeros((h, w), dtype=bool)
    mask[500:520, 500:520] = True
    p = tmp_path / "nd.tif"
    _make_rgb_tif(p, w, h, (0, 0, 204.8, 204.8), nodata_mask=mask)
    cfg = TileConfig.from_preset(TilePreset.MEDIUM)
    tiles = list(iter_tiles(p, cfg))
    t = tiles[0]
    assert isinstance(t, NodataTile) or t.nodata_mask[500:520, 500:520].any()
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/pipeline/test_tiler.py -v`
Expected: new tests FAIL.

- [ ] **Step 3: Append implementation**

```python
# ki_geodaten/pipeline/tiler.py (append)

@dataclass(frozen=True)
class NodataTile:
    tile_row: int
    tile_col: int
    pixel_origin: tuple[int, int]
    size: int
    center_margin: int
    affine: Affine
    reason: Literal["NODATA_PIXELS"] = "NODATA_PIXELS"

def _safe_center_has_nodata(nodata_mask: np.ndarray, margin: int, size: int, threshold: float) -> bool:
    safe = nodata_mask[margin:size - margin, margin:size - margin]
    return bool(safe.mean() > threshold)

def iter_tiles(
    vrt_path: Path, cfg: TileConfig,
    *, safe_center_nodata_threshold: float = 0.0,
) -> Iterator["Tile | NodataTile"]:
    with rasterio.Env(GDAL_MAX_DATASET_POOL_SIZE=256):
        with rasterio.open(vrt_path) as src:
            for tile_row, tile_col, row_off, col_off in iter_grid(src, cfg):
                window = Window(col_off=col_off, row_off=row_off, width=cfg.size, height=cfg.size)
                tile_affine = src.window_transform(window)

                mask_band1 = src.read_masks(1, window=window, boundless=True, fill_value=0)
                nodata_mask = mask_band1 == 0

                if _safe_center_has_nodata(nodata_mask, cfg.center_margin, cfg.size,
                                           safe_center_nodata_threshold):
                    yield NodataTile(
                        tile_row=tile_row, tile_col=tile_col,
                        pixel_origin=(row_off, col_off),
                        size=cfg.size, center_margin=cfg.center_margin,
                        affine=tile_affine,
                    )
                    continue

                arr_chw = src.read(indexes=[1, 2, 3], window=window,
                                   boundless=True, fill_value=0)
                arr_hwc = arr_chw.transpose(1, 2, 0)
                yield Tile(
                    array=arr_hwc, pixel_origin=(row_off, col_off),
                    size=cfg.size, center_margin=cfg.center_margin,
                    affine=tile_affine, tile_row=tile_row, tile_col=tile_col,
                    nodata_mask=nodata_mask,
                )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/pipeline/test_tiler.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add ki_geodaten/pipeline/tiler.py tests/pipeline/test_tiler.py
git commit -m "feat(tiler): iter_tiles with per-tile affine and safe-center NoData detection"
```

---

## Task 12: Segmenter — `MaskResult` + Tile-Local NMS

**Files:**
- Create: `ki_geodaten/pipeline/segmenter.py`
- Test: `tests/pipeline/test_segmenter.py`

**See Spec §5.3** — tile-local dedup BEFORE polygonization: IoU ≥ `LOCAL_MASK_NMS_IOU` OR containment ≥ `LOCAL_MASK_CONTAINMENT_RATIO`.

Per Spec §11 we do NOT unit-test SAM 3.1 itself; only the NMS logic and the wrapper's API surface.

- [ ] **Step 1: Write failing test**

```python
# tests/pipeline/test_segmenter.py
import numpy as np
import pytest
from ki_geodaten.pipeline.segmenter import MaskResult, local_mask_nms

def _mask(bbox, shape=(100, 100)):
    m = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = bbox
    m[r0:r1, c0:c1] = True
    return m

def test_local_nms_drops_high_iou_duplicate():
    a = MaskResult(mask=_mask((10, 10, 50, 50)), score=0.9, box_pixel=(10, 10, 50, 50))
    b = MaskResult(mask=_mask((11, 11, 49, 49)), score=0.8, box_pixel=(11, 11, 49, 49))
    kept = local_mask_nms([a, b], iou_threshold=0.6, containment_ratio=0.9)
    assert kept == [a]

def test_local_nms_drops_contained_mask():
    a = MaskResult(mask=_mask((0, 0, 80, 80)), score=0.9, box_pixel=(0, 0, 80, 80))
    b = MaskResult(mask=_mask((10, 10, 30, 30)), score=0.8, box_pixel=(10, 10, 30, 30))
    kept = local_mask_nms([a, b], iou_threshold=0.6, containment_ratio=0.9)
    assert kept == [a]

def test_local_nms_keeps_spatially_separate():
    a = MaskResult(mask=_mask((0, 0, 20, 20)), score=0.9, box_pixel=(0, 0, 20, 20))
    b = MaskResult(mask=_mask((70, 70, 90, 90)), score=0.8, box_pixel=(70, 70, 90, 90))
    kept = local_mask_nms([a, b], iou_threshold=0.6, containment_ratio=0.9)
    assert sorted(kept, key=lambda x: x.score, reverse=True) == [a, b]

def test_local_nms_score_descending_priority():
    high = MaskResult(mask=_mask((0, 0, 50, 50)), score=0.9, box_pixel=(0, 0, 50, 50))
    low  = MaskResult(mask=_mask((0, 0, 50, 50)), score=0.3, box_pixel=(0, 0, 50, 50))
    kept = local_mask_nms([low, high], iou_threshold=0.6, containment_ratio=0.9)
    assert kept == [high]
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/pipeline/test_segmenter.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# ki_geodaten/pipeline/segmenter.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np

@dataclass(frozen=True)
class MaskResult:
    mask: np.ndarray
    score: float
    box_pixel: tuple[int, int, int, int]

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(a, b).sum())
    return inter / union

def _containment(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    min_area = min(int(a.sum()), int(b.sum()))
    if min_area == 0:
        return 0.0
    return inter / min_area

def local_mask_nms(
    masks: Iterable[MaskResult],
    *, iou_threshold: float, containment_ratio: float,
) -> list[MaskResult]:
    """Score-desc greedy dedup (Spec §5.3)."""
    sorted_masks = sorted(masks, key=lambda m: m.score, reverse=True)
    kept: list[MaskResult] = []
    for cand in sorted_masks:
        drop = False
        for keep in kept:
            if _iou(cand.mask, keep.mask) >= iou_threshold:
                drop = True; break
            if _containment(cand.mask, keep.mask) >= containment_ratio:
                drop = True; break
        if not drop:
            kept.append(cand)
    return kept

class Sam3Segmenter:
    """Thin wrapper; model IO is not unit-tested (Spec §11).
    Orchestrator stubs the whole class in tests."""
    def __init__(
        self, checkpoint: Path, *, device: str = "cuda",
        iou_threshold: float = 0.6, containment_ratio: float = 0.9,
    ) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.iou_threshold = iou_threshold
        self.containment_ratio = containment_ratio
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        if self._model is not None:
            return
        # Lazy import so tests don't require torch/sam3
        import torch  # noqa: F401
        from sam3 import build_sam3  # pragma: no cover
        self._model = build_sam3(str(self.checkpoint), device=self.device)
        self._tokenizer = self._model.text_tokenizer

    def predict(self, tile, prompt: str) -> list[MaskResult]:
        if self._model is None:
            self.load()
        raw = self._model.predict(tile.array, prompt)
        return local_mask_nms(
            raw, iou_threshold=self.iou_threshold, containment_ratio=self.containment_ratio,
        )

    def encoder_token_count(self, prompt: str) -> int:
        """Final templatized encoder sequence length (Spec §5.3 bullet 3)."""
        if self._tokenizer is None:
            self.load()
        return int(self._tokenizer(prompt, add_special_tokens=True).input_ids.shape[-1])
```

Note: `sam3` API is placeholder (Spec §13 lists `sam3` from `facebookresearch/sam3`). Real signature will be adjusted once the model is installed; adapter is isolated here.

- [ ] **Step 4: Run tests**

Run: `pytest tests/pipeline/test_segmenter.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ki_geodaten/pipeline/segmenter.py tests/pipeline/test_segmenter.py
git commit -m "feat(segmenter): MaskResult, local NMS, Sam3Segmenter scaffold"
```

---

## Task 13: Merger — `keep_center_only`

**Files:**
- Create: `ki_geodaten/pipeline/merger.py`
- Test: `tests/pipeline/test_merger.py`

**See Spec §5.4 step 1** — halfopen `[margin, size-margin)`, BBox-center (NOT polygon centroid).

- [ ] **Step 1: Write failing test**

```python
# tests/pipeline/test_merger.py
import numpy as np
import pytest
from affine import Affine
from ki_geodaten.pipeline.segmenter import MaskResult
from ki_geodaten.pipeline.tiler import Tile, TileConfig
from ki_geodaten.pipeline.merger import keep_center_only
from ki_geodaten.models import TilePreset

def _tile(preset=TilePreset.MEDIUM):
    cfg = TileConfig.from_preset(preset)
    return Tile(
        array=np.zeros((cfg.size, cfg.size, 3), dtype=np.uint8),
        pixel_origin=(0, 0), size=cfg.size, center_margin=cfg.center_margin,
        affine=Affine(0.2, 0, 0, 0, -0.2, cfg.size * 0.2),
        tile_row=0, tile_col=0,
        nodata_mask=np.zeros((cfg.size, cfg.size), dtype=bool),
    )

def _mr(box, shape=(1024, 1024)):
    m = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = box
    m[r0:r1, c0:c1] = True
    return MaskResult(mask=m, score=0.9, box_pixel=box)

def test_keep_when_center_in_safe_zone_medium():
    tile = _tile(TilePreset.MEDIUM)
    m = _mr((500, 500, 524, 524))
    assert keep_center_only([m], tile) == [m]

def test_drop_when_center_in_margin():
    tile = _tile(TilePreset.MEDIUM)
    m = _mr((50, 50, 100, 100))
    assert keep_center_only([m], tile) == []

def test_halfopen_right_edge_excluded():
    tile = _tile(TilePreset.MEDIUM)
    # size=1024, margin=320 → hi=704. center 704 → excluded
    m = _mr((703, 320, 705, 322))  # center (704.0, 321.0)
    assert keep_center_only([m], tile) == []

def test_halfopen_left_edge_included():
    tile = _tile(TilePreset.MEDIUM)
    m = _mr((319, 319, 321, 321))  # center (320.0, 320.0)
    assert keep_center_only([m], tile) == [m]

def test_all_presets_accept_midpoint():
    for preset in TilePreset:
        tile = _tile(preset)
        m = _mr((510, 510, 514, 514))
        assert keep_center_only([m], tile) == [m]

def test_uses_bbox_center_not_geometric_centroid():
    tile = _tile(TilePreset.MEDIUM)
    mask = np.zeros((1024, 1024), dtype=bool)
    # C-shape with centroid outside mask interior, but bbox center (512,512) still safe
    mask[500:524, 500:510] = True
    mask[500:524, 514:524] = True
    mask[500:504, 500:524] = True
    mr = MaskResult(mask=mask, score=0.9, box_pixel=(500, 500, 524, 524))
    assert keep_center_only([mr], tile) == [mr]
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/pipeline/test_merger.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# ki_geodaten/pipeline/merger.py
from __future__ import annotations
from typing import Iterable
from ki_geodaten.pipeline.segmenter import MaskResult
from ki_geodaten.pipeline.tiler import Tile

def keep_center_only(masks: Iterable[MaskResult], tile: Tile) -> list[MaskResult]:
    """Halfopen safe zone [margin, size-margin) per Spec §5.4 step 1."""
    margin = tile.center_margin
    lo = margin
    hi = tile.size - margin
    out: list[MaskResult] = []
    for m in masks:
        r0, c0, r1, c1 = m.box_pixel
        cr = (r0 + r1) / 2
        cc = (c0 + c1) / 2
        if lo <= cr < hi and lo <= cc < hi:
            out.append(m)
    return out
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/pipeline/test_merger.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ki_geodaten/pipeline/merger.py tests/pipeline/test_merger.py
git commit -m "feat(merger): keep_center_only with halfopen BBox-center rule"
```

---

## Task 14: Merger — Raster→Polygon with `connectivity=8` + Polygon-Only Extraction

**Files:**
- Modify: `ki_geodaten/pipeline/merger.py`
- Test: `tests/pipeline/test_merger.py` (append)

**See Spec §5.4 steps 2, 3** — `shapes(connectivity=8)`, exclude background (`value != 1`), `make_valid`, recursive polygon-only filter, `MIN_POLYGON_AREA_M2`.

- [ ] **Step 1: Append failing tests**

```python
# tests/pipeline/test_merger.py (append)
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, LineString, Point
from ki_geodaten.pipeline.merger import extract_polygons, masks_to_polygons

def test_extract_polygons_from_polygon():
    p = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    assert extract_polygons(p) == [p]

def test_extract_polygons_from_multipolygon():
    mp = MultiPolygon([
        Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]),
        Polygon([(10, 10), (15, 10), (15, 15), (10, 15)]),
    ])
    assert len(extract_polygons(mp)) == 2

def test_extract_polygons_drops_non_polygon_from_collection():
    gc = GeometryCollection([
        Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]),
        LineString([(0, 0), (10, 10)]),
        Point(5, 5),
    ])
    polys = extract_polygons(gc)
    assert len(polys) == 1
    assert isinstance(polys[0], Polygon)

def test_extract_polygons_drops_empty():
    assert extract_polygons(Polygon()) == []

def test_masks_to_polygons_connectivity_8_joins_diagonal():
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:30, 10:30] = True
    mask[30:50, 30:50] = True  # diagonally touches at (30,30)
    mr = MaskResult(mask=mask, score=0.95, box_pixel=(10, 10, 50, 50))
    tile = _tile()
    gdf = masks_to_polygons([mr], tile, min_area_m2=0.01)
    assert len(gdf) == 1  # connectivity=8 → single polygon

def test_masks_to_polygons_excludes_background():
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:30, 10:30] = True
    mr = MaskResult(mask=mask, score=0.9, box_pixel=(10, 10, 30, 30))
    tile = _tile()
    gdf = masks_to_polygons([mr], tile, min_area_m2=0.01)
    assert len(gdf) == 1

def test_masks_to_polygons_area_filter():
    mask = np.zeros((100, 100), dtype=bool)
    mask[0:2, 0:2] = True  # 0.16 m²
    mr = MaskResult(mask=mask, score=0.9, box_pixel=(0, 0, 2, 2))
    tile = _tile()
    gdf = masks_to_polygons([mr], tile, min_area_m2=1.0)
    assert len(gdf) == 0

def test_masks_to_polygons_metadata():
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:30, 10:30] = True
    mr = MaskResult(mask=mask, score=0.77, box_pixel=(10, 10, 30, 30))
    tile = _tile()
    gdf = masks_to_polygons([mr], tile, min_area_m2=0.01)
    assert gdf.iloc[0]["score"] == 0.77
    assert gdf.iloc[0]["source_tile_row"] == 0
    assert gdf.iloc[0]["source_tile_col"] == 0
    assert str(gdf.crs) == "EPSG:25832"
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/pipeline/test_merger.py -v`
Expected: new tests FAIL.

- [ ] **Step 3: Append implementation**

```python
# ki_geodaten/pipeline/merger.py (append)
import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape as shapely_shape
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.validation import make_valid

def extract_polygons(geom) -> list[Polygon]:
    """Recursive polygon-only filter (Spec §5.4 step 3)."""
    if isinstance(geom, Polygon):
        return [geom] if not geom.is_empty else []
    if isinstance(geom, MultiPolygon):
        return [p for p in geom.geoms if not p.is_empty]
    if isinstance(geom, GeometryCollection):
        out: list[Polygon] = []
        for sub in geom.geoms:
            out.extend(extract_polygons(sub))
        return out
    return []

def masks_to_polygons(
    masks: list[MaskResult], tile: Tile, *, min_area_m2: float,
) -> gpd.GeoDataFrame:
    """Spec §5.4 step 2,3: connectivity=8, exclude background, make_valid, area filter."""
    records: list[dict] = []
    geoms: list[Polygon] = []
    for mr in masks:
        mask_u8 = mr.mask.astype("uint8")
        for geom_dict, value in shapes(mask_u8, mask=mr.mask, transform=tile.affine, connectivity=8):
            if value != 1:
                continue
            geom = make_valid(shapely_shape(geom_dict))
            for poly in extract_polygons(geom):
                if poly.area < min_area_m2:
                    continue
                geoms.append(poly)
                records.append({
                    "score": mr.score,
                    "source_tile_row": tile.tile_row,
                    "source_tile_col": tile.tile_col,
                })
    return gpd.GeoDataFrame(records, geometry=geoms, crs="EPSG:25832")
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/pipeline/test_merger.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add ki_geodaten/pipeline/merger.py tests/pipeline/test_merger.py
git commit -m "feat(merger): raster→polygon with connectivity=8, polygon-only, area filter"
```

---
