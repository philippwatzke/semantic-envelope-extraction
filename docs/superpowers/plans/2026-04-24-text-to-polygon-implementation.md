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
│   ├── dop_client.py             # DOP20 WMS download, meter-snap, pagination, PNG→GeoTIFF wrap, VRT build
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
│   │   ├── test_dop_client.py
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

## Task 0: WMS Capabilities Verification Gate

**Files:**
- Create: `docs/wms-verification.md` (replaces obsolete `docs/wcs-verification.md`)

**Must complete before Task 7/8 implementation.** The DOP client uses LDBV's OpenData WMS (`https://geoservices.bayern.de/od/wms/dop/v1/dop20`, CC BY 4.0, no auth). The WMS has no native pixel grid — pixel-accurate mosaicking depends on the server consistently sampling its source raster when the client requests BBoxes aligned to a 0.2 m meter grid. This task verifies that assumption empirically before writing the client.

- [ ] **Step 1: Delete stale WCS verification doc, create WMS one**

```bash
git rm docs/wcs-verification.md
```

Write `docs/wms-verification.md`:

```markdown
# LDBV DOP20 WMS Verification

Verified on: YYYY-MM-DD

- WMS_URL:           https://geoservices.bayern.de/od/wms/dop/v1/dop20
- WMS_VERSION:       1.1.1
- WMS_LAYER:         by_dop20c
- WMS_CRS:           EPSG:25832
- WMS_FORMAT:        image/png
- WMS_MAX_PIXELS:    6000
- PNG band count:    4 (RGBA) — Alpha is NoData mask
- Pixel resolution:  0.2 m native
- Coverage bbox (EPSG:25832): minx=497000 miny=5234000 maxx=857000 maxy=5604000

## Evidence

- GetCapabilities URL: https://geoservices.bayern.de/od/wms/dop/v1/dop20?SERVICE=WMS&REQUEST=GetCapabilities
- Supported formats (verbatim from <Format> tags):
- Supported CRS (verbatim from <SRS> tags):
- Max pixels clause (from <Abstract>):
- Tested GetMap URL (Munich-area, 500×500 px):
- PIL-verified band count of response:
- Adjacent-chunk edge test — URL A:
- Adjacent-chunk edge test — URL B:
- Edge test result: pixel at column 0 of chunk B matches pixel at column (width-1) of chunk A? (yes/no, numerical diff)

## Notes
```

- [ ] **Step 2: Perform the adjacent-chunk edge test**

The WMS has no server-side grid. Verify empirically that two BBoxes snapped to the 0.2 m meter grid with `next_minx == prev_maxx` produce pixel-aligned mosaics.

Pick a Munich-area AOI. For example:
- Chunk A: `BBOX=690000,5334000,690100,5334100` (100 m × 100 m) → `WIDTH=500&HEIGHT=500` (exact 0.2 m/px)
- Chunk B: `BBOX=690100,5334000,690200,5334100` (adjacent east) → same WIDTH/HEIGHT

Request both with `VERSION=1.1.1`, `LAYERS=by_dop20c`, `SRS=EPSG:25832`, `FORMAT=image/png`, `TRANSPARENT=TRUE`.

Decode both PNGs (Pillow / numpy), compare:
- Chunk A column `499` (rightmost) vs Chunk B column `0` (leftmost).
- Within a geographically smooth area, these should be effectively identical (pixel difference ≤ 1-2 values per channel due to JPEG-style internal compression artefacts, not structural drift).
- Check for structural drift: if corresponding row features (e.g., a road edge) appear at different row positions, the server is resampling inconsistently → WMS pivot is not viable, fall back to WCS with credentials.

Record the diff numerically in `docs/wms-verification.md` Evidence section.

- [ ] **Step 3: Propagate verified values into config defaults**

Only after Step 1/2 pass, set `WMS_URL`, `WMS_LAYER`, `WMS_VERSION`, `WMS_FORMAT`, `WMS_CRS`, `WMS_MAX_PIXELS` in `.env.example` / `config.py`. Unlike WCS, there is no grid origin — the meter-snap origin is fixed at `(0.0, 0.0)` in EPSG:25832 by convention (any origin on a 0.2 m grid works; `(0, 0)` is the simplest and matches integer snapping).

- [ ] **Step 4: Commit**

```bash
git add docs/wms-verification.md
git commit -m "docs: verify LDBV DOP20 WMS capabilities and pixel alignment"
```

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
WMS_URL=https://geoservices.bayern.de/od/wms/dop/v1/dop20
WMS_LAYER=by_dop20c
WMS_VERSION=1.1.1
WMS_FORMAT=image/png
WMS_CRS=EPSG:25832
WMS_MAX_PIXELS=6000
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
    assert s.WMS_URL == "https://geoservices.bayern.de/od/wms/dop/v1/dop20"
    assert s.WMS_LAYER == "by_dop20c"
    assert s.WMS_VERSION == "1.1.1"
    assert s.WMS_FORMAT == "image/png"
    assert s.WMS_CRS == "EPSG:25832"
    assert s.WMS_MAX_PIXELS == 6000
    assert s.WMS_GRID_ORIGIN_X == 0.0
    assert s.WMS_GRID_ORIGIN_Y == 0.0
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

    # WMS (LDBV OpenData DOP20 — Task 0 verified)
    WMS_URL: str = "https://geoservices.bayern.de/od/wms/dop/v1/dop20"
    WMS_LAYER: str = "by_dop20c"
    WMS_VERSION: str = "1.1.1"
    WMS_FORMAT: str = "image/png"
    WMS_CRS: str = "EPSG:25832"
    WMS_MAX_PIXELS: int = 6000
    # WMS has no native server-side grid; we snap client-side to this origin.
    # (0.0, 0.0) + 0.2 m step is the natural integer-meter raster for DOP20.
    WMS_GRID_ORIGIN_X: float = 0.0
    WMS_GRID_ORIGIN_Y: float = 0.0

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
    DOP_TIMEOUT = "DOP_TIMEOUT"
    DOP_HTTP_ERROR = "DOP_HTTP_ERROR"
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
                          'DOP_TIMEOUT','DOP_HTTP_ERROR','OOM',
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
    """executemany-based bulk update per Spec §8. Increments validation_revision.

    Note: we rely on sqlite3.Cursor.rowcount being cumulative across an
    executemany() call — this is only guaranteed on Python 3.12+. The
    pyproject.toml requires-python setting enforces that; do NOT downgrade.
    """
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

## Task 7: DOP Client — BBox Prep (Snap + Margin + Min-Size)

**Files:**
- Create: `ki_geodaten/pipeline/dop_client.py`
- Test: `tests/pipeline/test_dop_client.py`

**See Spec §5.1 pts 1, 4, 5, 6** — snapping, margin expansion, minimum size, pixel-count via round().

**Provider:** LDBV OpenData WMS (no auth, CC BY 4.0). Because WMS has no server-advertised grid, the snap origin is fixed at `(0.0, 0.0)` — the natural integer-meter raster of EPSG:25832. Task 0's edge test verified that the server samples consistently when the client requests BBoxes aligned to the 0.2 m meter grid, so the same snap-to-grid contract that would apply to WCS holds here.

**Resolved design decision:** Margin expansion is `CENTER_MARGIN_PX[preset] * 0.2m` (= 32 m / 64 m / 96 m for small / medium / large). The larger values 64 m / 128 m / 192 m are the max-object diameters (`2 * CENTER_MARGIN`) and are not the download expansion. Spec §5.1 pt 4 has been aligned with this formula.

Split of concerns: this task implements only the pure geometry math; HTTP comes next task.

- [ ] **Step 1: Write failing test**

```python
# tests/pipeline/test_dop_client.py
import pytest
from ki_geodaten.pipeline.dop_client import (
    prepare_download_bbox, DopDownloadError, PreparedBBox,
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

Run: `pytest tests/pipeline/test_dop_client.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `dop_client.py` (geometry only; HTTP is Task 8)**

```python
# ki_geodaten/pipeline/dop_client.py
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

class DopDownloadError(Exception):
    """Transport or mosaic error from the DOP20 WMS client (Task 8)."""
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

Run: `pytest tests/pipeline/test_dop_client.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add ki_geodaten/pipeline/dop_client.py tests/pipeline/test_dop_client.py
git commit -m "feat(dop): BBox prep with meter-snap, margin expansion, min-size"
```

---

## Task 8: DOP Client — Chunk Grid & WMS HTTP Download + VRT

**Files:**
- Modify: `ki_geodaten/pipeline/dop_client.py`
- Test: `tests/pipeline/test_dop_client.py` (append)

**See Spec §5.1 pts 2, 3, 6, 7, 8 and Task 0** — pagination grid, HTTP with retry, `round()` for pixels, contiguous edges, BuildVRT. The provider is the LDBV OpenData WMS (`GetMap`), not WCS. Key differences from a WCS client:
- No server-side coverage axis labels / scale-size params — just `BBOX`, `WIDTH`, `HEIGHT` in WMS 1.1.1 (`SRS=EPSG:25832`, X/Y order).
- WMS returns a raw raster without embedded georeferencing. We wrap the PNG bytes into a GeoTIFF with a client-constructed affine transform, then `gdal.BuildVRT` over the GeoTIFFs.
- PNG comes as 4-band RGBA. Band 4 (alpha) is the NoData mask for areas outside the orthophoto coverage — it MUST be preserved into the wrapped GeoTIFF so that downstream `rasterio.DatasetReader.dataset_mask()` works (Task 11).

- [ ] **Step 1: Append failing tests**

```python
# tests/pipeline/test_dop_client.py (append)
from ki_geodaten.pipeline.dop_client import plan_chunk_grid, download_dop20

def test_chunk_grid_seamless_edges():
    # 2000m × 2000m box, WMS_MAX_PIXELS=6000 → chunk side 6000*0.2 = 1200m
    chunks = plan_chunk_grid(
        minx=0.0, miny=0.0, maxx=2000.0, maxy=2000.0,
        max_pixels=6000, step=0.2, origin_x=0.0, origin_y=0.0,
    )
    # Expect 2×2 grid (1200 + 800 remainder)
    assert len(chunks) == 4
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
        max_pixels=6000, step=0.2, origin_x=0.0, origin_y=0.0,
    )
    assert len(chunks) == 1

def test_download_dop20_http_success(tmp_path, responses):
    # Mock WMS GetMap. Implementation uses requests; `responses` intercepts it.
    # We emit a valid minimal PNG RGBA blob per call — the client is responsible
    # for wrapping it into a georeferenced GeoTIFF.
    import io
    import numpy as np
    from PIL import Image
    import rasterio
    from rasterio.transform import from_bounds

    def make_png(w: int, h: int) -> bytes:
        arr = np.zeros((h, w, 4), dtype=np.uint8)
        arr[..., 3] = 255  # opaque alpha
        buf = io.BytesIO()
        Image.fromarray(arr, mode="RGBA").save(buf, format="PNG")
        return buf.getvalue()

    bbox = (0.0, 0.0, 204.8, 204.8)
    w = round((bbox[2] - bbox[0]) / 0.2)
    h = round((bbox[3] - bbox[1]) / 0.2)
    responses.add(
        responses.GET,
        "http://example/wms",
        body=make_png(w, h),
        status=200,
        content_type="image/png",
    )

    vrt_path = download_dop20(
        bbox_utm=bbox, out_dir=tmp_path,
        wms_url="http://example/wms", layer="by_dop20c",
        wms_version="1.1.1", fmt="image/png", crs="EPSG:25832",
        max_pixels=6000, origin_x=0.0, origin_y=0.0,
    )
    assert vrt_path.exists()
    assert vrt_path.suffix == ".vrt"
    # Verify the chunk GeoTIFF is actually georeferenced and preserves alpha
    tif = next(tmp_path.glob("chunk_*.tif"))
    with rasterio.open(tif) as src:
        assert src.crs.to_string() == "EPSG:25832"
        assert src.count == 4  # RGBA preserved
        assert src.bounds.left == 0.0
        assert src.bounds.right == 204.8
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/pipeline/test_dop_client.py -v`
Expected: new tests FAIL.

- [ ] **Step 3: Append implementation**

```python
# ki_geodaten/pipeline/dop_client.py (append)
import io
import logging
from dataclasses import dataclass

import numpy as np
import rasterio
import requests
from osgeo import gdal
from PIL import Image
from rasterio.transform import from_bounds
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

def _png_bytes_to_array(payload: bytes) -> np.ndarray:
    """Decode WMS PNG response to HxWxC uint8 ndarray (RGBA expected)."""
    img = Image.open(io.BytesIO(payload))
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return np.asarray(img)

def _write_geotiff(
    out_path: Path, arr: np.ndarray,
    bbox: tuple[float, float, float, float], crs: str,
) -> None:
    """Wrap an HxWxC uint8 RGBA ndarray into a 4-band GeoTIFF.
    Band 4 is written as the alpha/NoData mask so rasterio.dataset_mask()
    returns it downstream (Task 11)."""
    h, w, c = arr.shape
    if c != 4:
        raise DopDownloadError(f"Expected RGBA from WMS, got {c} bands")
    transform = from_bounds(*bbox, w, h)
    with rasterio.open(
        out_path, "w", driver="GTiff",
        height=h, width=w, count=4, dtype="uint8",
        crs=crs, transform=transform,
        photometric="RGB",
    ) as dst:
        # (H, W, C) → (C, H, W) for rasterio band order
        for i in range(4):
            dst.write(arr[..., i], i + 1)
        # Declare band 4 as alpha so dataset_mask() finds it
        dst.colorinterp = (
            rasterio.enums.ColorInterp.red,
            rasterio.enums.ColorInterp.green,
            rasterio.enums.ColorInterp.blue,
            rasterio.enums.ColorInterp.alpha,
        )

def _fetch_chunk(
    session: requests.Session, wms_url: str, layer: str,
    chunk: ChunkPlan, out_path: Path, *,
    wms_version: str, fmt: str, crs: str,
) -> None:
    """WMS 1.1.1 GetMap → decode PNG RGBA → wrap as georeferenced GeoTIFF."""
    w = chunk.width_px()
    h = chunk.height_px()
    crs_param = "SRS" if wms_version.startswith("1.1") else "CRS"
    params = {
        "SERVICE": "WMS",
        "VERSION": wms_version,
        "REQUEST": "GetMap",
        "LAYERS": layer,
        "STYLES": "",
        crs_param: crs,
        # WMS 1.1.1 EPSG:25832: X,Y (easting, northing) — same order as our chunk
        "BBOX": f"{chunk.minx},{chunk.miny},{chunk.maxx},{chunk.maxy}",
        "WIDTH": str(w),
        "HEIGHT": str(h),
        "FORMAT": fmt,
        "TRANSPARENT": "TRUE",
    }
    try:
        r = session.get(wms_url, params=params, timeout=(10, 60))
        r.raise_for_status()
    except requests.exceptions.Timeout as e:
        raise DopDownloadError("DOP_TIMEOUT") from e
    except requests.exceptions.RequestException as e:
        raise DopDownloadError("DOP_HTTP_ERROR") from e
    if not r.headers.get("Content-Type", "").startswith("image/"):
        # WMS ExceptionReport comes back as text/xml with HTTP 200
        raise DopDownloadError(f"DOP_HTTP_ERROR: non-image response {r.text[:200]}")
    arr = _png_bytes_to_array(r.content)
    _write_geotiff(
        out_path, arr,
        bbox=(chunk.minx, chunk.miny, chunk.maxx, chunk.maxy),
        crs=crs,
    )

def download_dop20(
    bbox_utm: tuple[float, float, float, float],
    out_dir: Path,
    *, wms_url: str, layer: str, wms_version: str, fmt: str, crs: str,
    max_pixels: int, origin_x: float, origin_y: float,
) -> Path:
    """Spec §5.1 signature. bbox_utm MUST be the already-prepared download_bbox.
    Downloads the AOI as a set of WMS GetMap chunks, wraps each in GeoTIFF, and
    combines them via gdal.BuildVRT for seamless downstream reading."""
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = plan_chunk_grid(
        *bbox_utm, max_pixels=max_pixels, step=STEP,
        origin_x=origin_x, origin_y=origin_y,
    )
    session = _build_session()
    chunk_files: list[str] = []
    for c in chunks:
        p = out_dir / f"chunk_{c.row}_{c.col}.tif"
        _fetch_chunk(
            session, wms_url, layer, c, p,
            wms_version=wms_version, fmt=fmt, crs=crs,
        )
        chunk_files.append(str(p))
    vrt_path = out_dir / "out.vrt"
    vrt = gdal.BuildVRT(str(vrt_path), chunk_files)
    if vrt is None:
        raise DopDownloadError("VRT_BUILD_FAILED")
    vrt.FlushCache()
    return vrt_path
```

Note: `responses` fixture requires `pytest-responses` — replace with `responses.activate` decorator if needed. If `responses` cannot register params, use callback-based matcher. The alternative test technique is to replace `_fetch_chunk` with a monkeypatch'd stub; the plan permits either, as long as VRT output (with georeferenced RGBA GeoTIFF chunks) is verified.

- [ ] **Step 4: Run tests**

Run: `pytest tests/pipeline/test_dop_client.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add ki_geodaten/pipeline/dop_client.py tests/pipeline/test_dop_client.py
git commit -m "feat(dop): paginated WMS chunk grid, PNG→GeoTIFF wrap, VRT build"
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

def test_iter_tiles_black_rgb_without_nodata_mask_is_not_nodata(tmp_path):
    p = tmp_path / "black.tif"
    with rasterio.open(
        p, "w", driver="GTiff", width=1024, height=1024, count=3,
        dtype="uint8", crs="EPSG:25832",
        transform=from_bounds(0, 0, 204.8, 204.8, 1024, 1024),
    ) as dst:
        dst.write(np.zeros((3, 1024, 1024), dtype="uint8"))
    cfg = TileConfig.from_preset(TilePreset.MEDIUM)
    tiles = list(iter_tiles(p, cfg))
    assert not isinstance(tiles[0], NodataTile)

def test_iter_tiles_alpha_mask_marks_nodata(tmp_path):
    from rasterio.enums import ColorInterp
    p = tmp_path / "alpha.tif"
    with rasterio.open(
        p, "w", driver="GTiff", width=1024, height=1024, count=4,
        dtype="uint8", crs="EPSG:25832",
        transform=from_bounds(0, 0, 204.8, 204.8, 1024, 1024),
    ) as dst:
        dst.write(np.full((3, 1024, 1024), 128, dtype="uint8"), indexes=[1, 2, 3])
        alpha = np.full((1024, 1024), 255, dtype="uint8")
        alpha[500:520, 500:520] = 0
        dst.write(alpha, 4)
        dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha)
    cfg = TileConfig.from_preset(TilePreset.MEDIUM)
    tiles = list(iter_tiles(p, cfg))
    assert isinstance(tiles[0], NodataTile)
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

                dataset_mask = src.dataset_mask(window=window, boundless=True, fill_value=0)
                nodata_mask = dataset_mask == 0

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

def test_local_nms_bbox_prefilter_skips_disjoint_dense_masks(monkeypatch):
    # Regression: disjoint candidates must not run full 1024x1024 logical_and
    # comparisons for every pair.
    a = MaskResult(mask=_mask((0, 0, 20, 20)), score=0.9, box_pixel=(0, 0, 20, 20))
    b = MaskResult(mask=_mask((70, 70, 90, 90)), score=0.8, box_pixel=(70, 70, 90, 90))

    def fail_iou(*args, **kwargs):
        raise AssertionError("dense IoU should not run for disjoint boxes")

    import ki_geodaten.pipeline.segmenter as segmenter
    monkeypatch.setattr(segmenter, "_iou", fail_iou)
    monkeypatch.setattr(segmenter, "_containment", fail_iou)
    assert local_mask_nms([a, b], iou_threshold=0.6, containment_ratio=0.9) == [a, b]
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

def _boxes_intersect(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return min(ar1, br1) > max(ar0, br0) and min(ac1, bc1) > max(ac0, bc0)

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
            # Cheap bbox gate first; only overlapping boxes can have non-zero
            # mask IoU/containment. This avoids O(N^2) dense 1024x1024 scans.
            if not _boxes_intersect(cand.box_pixel, keep.box_pixel):
                continue
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
        raise NotImplementedError(
            "Task 12A must verify the real SAM 3.1 prediction API and implement "
            "tensor-to-CPU NumPy normalization before production inference is wired."
        )

    def encoder_token_count(self, prompt: str) -> int:
        """Final templatized encoder sequence length (Spec §5.3 bullet 3)."""
        if self._tokenizer is None:
            self.load()
        return int(self._tokenizer(prompt, add_special_tokens=True).input_ids.shape[-1])

class Sam3TextTokenCounter:
    """Tokenizer-only adapter for the FastAPI process.

    This class MUST NOT build or load the SAM vision model. Task 12A replaces
    the placeholder import with the verified upstream tokenizer-only API.
    """
    def __init__(self, checkpoint: Path) -> None:
        self.checkpoint = checkpoint
        self._tokenizer = None

    def load(self) -> None:
        if self._tokenizer is not None:
            return
        try:
            from sam3.text import build_text_tokenizer  # pragma: no cover - verified in Task 12A
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "SAM3 tokenizer-only API is not verified yet; complete Task 12A"
            ) from exc
        self._tokenizer = build_text_tokenizer(str(self.checkpoint))

    def __call__(self, prompt: str) -> int:
        if self._tokenizer is None:
            self.load()
        return int(self._tokenizer(prompt, add_special_tokens=True).input_ids.shape[-1])
```

Note: `sam3` model and tokenizer APIs are placeholders until Task 12A is completed. `Sam3Segmenter.predict()` intentionally raises until the real adapter is implemented. Do not implement Worker/API integration against this placeholder.

- [ ] **Step 4: Run tests**

Run: `pytest tests/pipeline/test_segmenter.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ki_geodaten/pipeline/segmenter.py tests/pipeline/test_segmenter.py
git commit -m "feat(segmenter): MaskResult, local NMS, Sam3Segmenter scaffold"
```

---

## Task 12A: Verify Real SAM 3.1 Adapter API

**Files:**
- Modify: `ki_geodaten/pipeline/segmenter.py`
- Test: `tests/pipeline/test_sam3_adapter_smoke.py`

**Must complete before Task 18/21 production wiring.** Task 12 intentionally scaffolds the SAM boundary, but the exact upstream `facebookresearch/sam3` API must be verified in the real environment. This task removes the placeholder risk.

- [ ] **Step 1: Install / expose upstream SAM 3.1 in the target environment**

Verify imports, checkpoint loading, text prompt encoding, and one image prediction against a tiny RGB tile. Record any required environment variables in `README.md`.

- [ ] **Step 2: Replace placeholder imports and normalization**

Update `Sam3Segmenter.load()` and `Sam3Segmenter.predict()` so they convert the actual upstream output into:

```python
MaskResult(mask=np.ndarray[bool], score=float, box_pixel=(row0, col0, row1, col1))
```

Required adapter behavior:
- `mask` is tile-local shape `(1024, 1024)` and boolean.
- `box_pixel` uses `(row0, col0, row1, col1)` coordinates matching `keep_center_only`.
- Scores are floats in `[0, 1]`.
- No `torch.Tensor` may reach `local_mask_nms`; masks/scores/boxes must be detached, moved to CPU, and converted to isolated NumPy/Python values first. Mask conversion must use `.detach().cpu().numpy().copy()` so returned NumPy arrays do not keep PyTorch tensor storage alive.
- Per-tile raw SAM outputs and temporary tensors must be dropped (`del raw`, `del tensor_refs`, etc.) before returning so the orchestrator's per-tile `torch.cuda.empty_cache()` can actually release VRAM.
- Raw model outputs are passed through `local_mask_nms`.
- Device placement never loads a second GPU model in the FastAPI process.

- [ ] **Step 3: Replace `Sam3TextTokenCounter` with verified tokenizer-only loading**

The webserver token counter must load only tokenizer/text-encoder state needed for token counting. It must not instantiate the full vision model or allocate CUDA memory.

- [ ] **Step 4: Add smoke tests**

```python
# tests/pipeline/test_sam3_adapter_smoke.py
import os
import pytest

pytest.importorskip("sam3")

def test_sam3_token_counter_does_not_require_cuda():
    from ki_geodaten.pipeline.segmenter import Sam3TextTokenCounter
    checkpoint = os.environ.get("SAM3_CHECKPOINT")
    if not checkpoint:
        pytest.skip("SAM3_CHECKPOINT not set")
    counter = Sam3TextTokenCounter(checkpoint)
    assert counter("building") > 0

def test_sam3_segmenter_smoke_prediction_shape():
    import numpy as np
    from affine import Affine
    from ki_geodaten.models import TilePreset
    from ki_geodaten.pipeline.tiler import Tile, TileConfig
    from ki_geodaten.pipeline.segmenter import Sam3Segmenter, MaskResult

    checkpoint = os.environ.get("SAM3_CHECKPOINT")
    if not checkpoint:
        pytest.skip("SAM3_CHECKPOINT not set")
    cfg = TileConfig.from_preset(TilePreset.MEDIUM)
    tile = Tile(
        array=np.zeros((1024, 1024, 3), dtype=np.uint8),
        pixel_origin=(0, 0), size=1024, center_margin=cfg.center_margin,
        affine=Affine(0.2, 0, 0, 0, -0.2, 204.8),
        tile_row=0, tile_col=0,
        nodata_mask=np.zeros((1024, 1024), dtype=bool),
    )
    seg = Sam3Segmenter(checkpoint)
    out = seg.predict(tile, "building")
    assert isinstance(out, list)
    for item in out:
        assert isinstance(item, MaskResult)
        assert item.mask.shape == (1024, 1024)
        assert item.mask.dtype == bool
```

- [ ] **Step 5: Commit**

```bash
git add ki_geodaten/pipeline/segmenter.py tests/pipeline/test_sam3_adapter_smoke.py README.md
git commit -m "feat(segmenter): verify real SAM 3.1 adapter and tokenizer-only counter"
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

## Task 15: Exporter — Two-Layer GPKG with AOI Clip + Empty-Schema Handling

**Files:**
- Create: `ki_geodaten/pipeline/exporter.py`
- Test: `tests/pipeline/test_exporter.py`

**See Spec §5.5** — AOI clip (Clip-Window-Semantik), re-export overwrite, empty layers with explicit schema AND explicit CRS, `nodata_regions` carries safe-center footprints (caller responsibility). The clipped export is the user-facing AOI product; for metrics, reference data must be clipped the same way. Do not switch this to an `intersects(aoi)` full-object filter unless the product semantics are explicitly changed.

- [ ] **Step 1: Write failing test**

```python
# tests/pipeline/test_exporter.py
import geopandas as gpd
import pytest
from pathlib import Path
from shapely.geometry import Polygon, box
import fiona

from ki_geodaten.pipeline.exporter import export_two_layer_gpkg

def _aoi():
    return box(691000.0, 5335000.0, 692000.0, 5336000.0)

def _detected_gdf(geoms, scores=None):
    scores = scores or [0.9] * len(geoms)
    return gpd.GeoDataFrame(
        {"score": scores,
         "source_tile_row": [0] * len(geoms),
         "source_tile_col": [0] * len(geoms)},
        geometry=list(geoms), crs="EPSG:25832",
    )

def _nodata_gdf(geoms, reasons=None):
    reasons = reasons or ["NODATA_PIXELS"] * len(geoms)
    return gpd.GeoDataFrame(
        {"reason": reasons}, geometry=list(geoms), crs="EPSG:25832",
    )

def test_export_writes_two_layers(tmp_path: Path):
    out = tmp_path / "j.gpkg"
    inside = Polygon([(691100, 5335100), (691200, 5335100),
                      (691200, 5335200), (691100, 5335200)])
    export_two_layer_gpkg(
        detected_gdf=_detected_gdf([inside]),
        nodata_gdf=_nodata_gdf([]),
        requested_bbox=_aoi(),
        out_path=out,
    )
    assert out.exists()
    layers = fiona.listlayers(str(out))
    assert "detected_objects" in layers
    assert "nodata_regions" in layers

def test_export_clips_crossing_polygon_to_aoi(tmp_path: Path):
    out = tmp_path / "j.gpkg"
    # Polygon crosses east edge of AOI (maxx=692000)
    crossing = Polygon([(691900, 5335500), (692100, 5335500),
                        (692100, 5335600), (691900, 5335600)])
    export_two_layer_gpkg(
        detected_gdf=_detected_gdf([crossing]),
        nodata_gdf=_nodata_gdf([]),
        requested_bbox=_aoi(),
        out_path=out,
    )
    gdf = gpd.read_file(out, layer="detected_objects")
    assert len(gdf) == 1
    xs = [pt[0] for pt in gdf.geometry.iloc[0].exterior.coords]
    assert max(xs) <= 692000.0 + 1e-6

def test_export_drops_polygon_fully_outside_aoi(tmp_path: Path):
    out = tmp_path / "j.gpkg"
    far = Polygon([(700000, 5400000), (700100, 5400000),
                   (700100, 5400100), (700000, 5400100)])
    export_two_layer_gpkg(
        detected_gdf=_detected_gdf([far]),
        nodata_gdf=_nodata_gdf([]),
        requested_bbox=_aoi(),
        out_path=out,
    )
    gdf = gpd.read_file(out, layer="detected_objects")
    assert len(gdf) == 0
    assert str(gdf.crs) == "EPSG:25832"

def test_export_empty_detected_has_explicit_crs(tmp_path: Path):
    out = tmp_path / "j.gpkg"
    export_two_layer_gpkg(
        detected_gdf=_detected_gdf([]),
        nodata_gdf=_nodata_gdf([]),
        requested_bbox=_aoi(),
        out_path=out,
    )
    gdf = gpd.read_file(out, layer="detected_objects")
    assert len(gdf) == 0
    assert str(gdf.crs) == "EPSG:25832"
    # schema properties exist
    with fiona.open(str(out), layer="detected_objects") as src:
        props = src.schema["properties"]
        assert "score" in props
        assert "source_tile_row" in props
        assert "source_tile_col" in props

def test_export_empty_nodata_has_explicit_schema(tmp_path: Path):
    out = tmp_path / "j.gpkg"
    export_two_layer_gpkg(
        detected_gdf=_detected_gdf([]),
        nodata_gdf=_nodata_gdf([]),
        requested_bbox=_aoi(),
        out_path=out,
    )
    gdf = gpd.read_file(out, layer="nodata_regions")
    assert len(gdf) == 0
    assert str(gdf.crs) == "EPSG:25832"
    with fiona.open(str(out), layer="nodata_regions") as src:
        props = src.schema["properties"]
        assert "reason" in props

def test_export_overwrites_existing_file(tmp_path: Path):
    out = tmp_path / "j.gpkg"
    p1 = Polygon([(691100, 5335100), (691200, 5335100),
                  (691200, 5335200), (691100, 5335200)])
    p2 = Polygon([(691300, 5335300), (691400, 5335300),
                  (691400, 5335400), (691300, 5335400)])
    export_two_layer_gpkg(_detected_gdf([p1]), _nodata_gdf([]), _aoi(), out)
    export_two_layer_gpkg(_detected_gdf([p2]), _nodata_gdf([]), _aoi(), out)
    gdf = gpd.read_file(out, layer="detected_objects")
    assert len(gdf) == 1   # only p2, not p1 + p2

def test_export_overwrite_removes_sqlite_sidecars(tmp_path: Path):
    out = tmp_path / "j.gpkg"
    out.write_bytes(b"old")
    out.with_suffix(".gpkg-wal").write_bytes(b"old wal")
    out.with_suffix(".gpkg-shm").write_bytes(b"old shm")
    inside = Polygon([(691100, 5335100), (691200, 5335100),
                      (691200, 5335200), (691100, 5335200)])
    export_two_layer_gpkg(_detected_gdf([inside]), _nodata_gdf([]), _aoi(), out)
    assert out.exists()
    assert not out.with_suffix(".gpkg-wal").exists()
    assert not out.with_suffix(".gpkg-shm").exists()
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/pipeline/test_exporter.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement**

```python
# ki_geodaten/pipeline/exporter.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd
from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid

from ki_geodaten.pipeline.merger import extract_polygons

_DETECTED_LAYER = "detected_objects"
_NODATA_LAYER = "nodata_regions"
_CRS = "EPSG:25832"

_EMPTY_DETECTED_COLS = ("score", "source_tile_row", "source_tile_col")
_EMPTY_NODATA_COLS = ("reason",)
_DETECTED_SCHEMA = {
    "geometry": "Polygon",
    "properties": {
        "score": "float",
        "source_tile_row": "int",
        "source_tile_col": "int",
    },
}
_NODATA_SCHEMA = {
    "geometry": "Polygon",
    "properties": {"reason": "str"},
}

def _clip_and_normalize(
    gdf: gpd.GeoDataFrame, aoi: BaseGeometry,
) -> gpd.GeoDataFrame:
    """Spec §5.5: Clip-Window-Semantik. After intersection, geometries may
    become MultiPolygon/GeometryCollection — re-run polygon-only pipeline."""
    if len(gdf) == 0:
        return gdf
    clipped = gdf.copy()
    clipped["geometry"] = clipped.geometry.intersection(aoi)
    clipped = clipped[~clipped.geometry.is_empty]
    # Explode to polygon-only components
    rows: list[dict] = []
    geoms: list = []
    for _, rec in clipped.iterrows():
        geom = make_valid(rec.geometry)
        for poly in extract_polygons(geom):
            payload = {k: rec[k] for k in rec.index if k != "geometry"}
            rows.append(payload)
            geoms.append(poly)
    if not geoms:
        return gpd.GeoDataFrame(
            {c: [] for c in gdf.columns if c != "geometry"},
            geometry=[], crs=_CRS,
        )
    return gpd.GeoDataFrame(rows, geometry=geoms, crs=_CRS)

def _empty_with_schema(cols: Iterable[str]) -> gpd.GeoDataFrame:
    """Spec §5.5 final — empty layer must still carry CRS and property columns."""
    return gpd.GeoDataFrame(
        {c: [] for c in cols}, geometry=[], crs=_CRS,
    )

def _unlink_gpkg_family(out_path: Path) -> None:
    """Delete GeoPackage plus SQLite sidecars before re-export.

    GeoPackage is SQLite. If a previous writer left `*.gpkg-wal`/`*.gpkg-shm`
    beside the main file, deleting only the main DB can corrupt the next fresh
    file created under the same name.
    """
    out_path.unlink(missing_ok=True)
    out_path.with_suffix(".gpkg-wal").unlink(missing_ok=True)
    out_path.with_suffix(".gpkg-shm").unlink(missing_ok=True)

def export_two_layer_gpkg(
    detected_gdf: gpd.GeoDataFrame,
    nodata_gdf: gpd.GeoDataFrame,
    requested_bbox: BaseGeometry,
    out_path: Path,
) -> None:
    """Spec §5.5 signature. Overwrites out_path unconditionally."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _unlink_gpkg_family(out_path)

    det = _clip_and_normalize(detected_gdf, requested_bbox)
    nod = _clip_and_normalize(nodata_gdf, requested_bbox)

    if len(det) == 0:
        det = _empty_with_schema(_EMPTY_DETECTED_COLS)
    if len(nod) == 0:
        nod = _empty_with_schema(_EMPTY_NODATA_COLS)

    # Explicit schema is required for empty layers; use Fiona because pyogrio
    # ignores/does not support the schema kwarg in common GeoPandas versions.
    det.to_file(
        out_path, layer=_DETECTED_LAYER, driver="GPKG",
        schema=_DETECTED_SCHEMA, engine="fiona",
    )
    nod.to_file(
        out_path, layer=_NODATA_LAYER, driver="GPKG",
        schema=_NODATA_SCHEMA, engine="fiona",
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/pipeline/test_exporter.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add ki_geodaten/pipeline/exporter.py tests/pipeline/test_exporter.py
git commit -m "feat(exporter): two-layer GPKG with AOI clip, overwrite, and empty-schema"
```

---

## Task 16: Serialization — WKB → GeoJSON (AOI-Clipped, Transformed, Precision-Reduced)

**Files:**
- Create: `ki_geodaten/app/serialization.py`
- Test: `tests/app/test_serialization.py`

**See Spec §8** — transform EPSG:25832 → EPSG:4326, `shapely.set_precision(grid_size=1e-6)`, AOI clip uses `bbox_utm_snapped` in UTM **before** transform, and the ProcessPool target returns a finished JSON string so FastAPI does not serialize large FeatureCollections on the event loop. The AOI clip is intentional Clip-Window-Semantik; metrics must clip reference features equivalently.

This module is intended to run inside a `ProcessPoolExecutor` (Task 21). No HTTP state here; only pure serialization.

- [ ] **Step 1: Write failing test**

```python
# tests/app/test_serialization.py
import json
import pytest
from shapely.geometry import Polygon, box
from shapely.wkb import dumps as wkb_dumps

from ki_geodaten.app.serialization import (
    build_polygons_feature_collection,
    build_nodata_feature_collection,
    build_polygons_geojson,
)

def _aoi_utm():
    return (691000.0, 5335000.0, 692000.0, 5336000.0)

def _poly_wkb(geom):
    return wkb_dumps(geom)

def test_polygons_geojson_has_4326_bounds_inside_bayern():
    poly = Polygon([(691100, 5335100), (691200, 5335100),
                    (691200, 5335200), (691100, 5335200)])
    rows = [{"id": 1, "geometry_wkb": _poly_wkb(poly),
             "score": 0.88, "validation": "ACCEPTED"}]
    fc = build_polygons_feature_collection(rows, aoi_utm=_aoi_utm())
    feat = fc["features"][0]
    assert feat["geometry"]["type"] == "Polygon"
    xs = [c[0] for c in feat["geometry"]["coordinates"][0]]
    assert 8.9 <= min(xs) <= max(xs) <= 13.9
    assert feat["properties"] == {"id": 1, "score": 0.88, "validation": "ACCEPTED"}

def test_polygons_precision_capped_at_1e_6():
    poly = Polygon([(691123.456789, 5335111.222333),
                    (691200, 5335100), (691200, 5335200),
                    (691100, 5335200)])
    rows = [{"id": 1, "geometry_wkb": _poly_wkb(poly),
             "score": 0.5, "validation": "ACCEPTED"}]
    fc = build_polygons_feature_collection(rows, aoi_utm=_aoi_utm())
    for coord in fc["features"][0]["geometry"]["coordinates"][0]:
        for v in coord:
            frac = abs(v - round(v, 6))
            assert frac < 1e-9, f"coord {v} exceeds 6 decimal precision"

def test_polygons_clip_by_aoi():
    poly = Polygon([(691900, 5335500), (692200, 5335500),
                    (692200, 5335600), (691900, 5335600)])
    rows = [{"id": 1, "geometry_wkb": _poly_wkb(poly),
             "score": 0.8, "validation": "ACCEPTED"}]
    fc = build_polygons_feature_collection(rows, aoi_utm=_aoi_utm())
    # Should still yield a feature (polygon intersects AOI)
    assert len(fc["features"]) == 1

def test_polygons_drop_fully_outside():
    poly = Polygon([(700000, 5400000), (700100, 5400000),
                    (700100, 5400100), (700000, 5400100)])
    rows = [{"id": 1, "geometry_wkb": _poly_wkb(poly),
             "score": 0.8, "validation": "ACCEPTED"}]
    fc = build_polygons_feature_collection(rows, aoi_utm=_aoi_utm())
    assert fc["features"] == []

def test_nodata_geojson_carries_reason():
    poly = Polygon([(691100, 5335100), (691200, 5335100),
                    (691200, 5335200), (691100, 5335200)])
    rows = [{"id": 9, "geometry_wkb": _poly_wkb(poly),
             "reason": "OOM"}]
    fc = build_nodata_feature_collection(rows, aoi_utm=_aoi_utm())
    assert fc["features"][0]["properties"] == {"reason": "OOM"}

def test_feature_collection_is_json_serializable():
    poly = Polygon([(691100, 5335100), (691200, 5335100),
                    (691200, 5335200), (691100, 5335200)])
    rows = [{"id": 1, "geometry_wkb": _poly_wkb(poly),
             "score": 0.5, "validation": "ACCEPTED"}]
    fc = build_polygons_feature_collection(rows, aoi_utm=_aoi_utm())
    json.dumps(fc)  # must not raise

def test_processpool_builder_returns_json_string():
    poly = Polygon([(691100, 5335100), (691200, 5335100),
                    (691200, 5335200), (691100, 5335200)])
    rows = [{"id": 1, "geometry_wkb": _poly_wkb(poly),
             "score": 0.5, "validation": "ACCEPTED"}]
    payload = build_polygons_geojson(rows, aoi_utm=_aoi_utm())
    assert isinstance(payload, str)
    assert json.loads(payload)["type"] == "FeatureCollection"
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/app/test_serialization.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# ki_geodaten/app/serialization.py
from __future__ import annotations
import json
from typing import Iterable
import shapely
from shapely.geometry import box, mapping
from shapely.ops import transform as shapely_transform
from shapely.wkb import loads as wkb_loads
from shapely.validation import make_valid

from ki_geodaten.pipeline.geo_utils import transformer_25832_to_4326
from ki_geodaten.pipeline.merger import extract_polygons

_PRECISION_DEG = 1e-6

def _transform_to_4326(geom):
    t = transformer_25832_to_4326()
    return shapely_transform(lambda x, y, z=None: t.transform(x, y), geom)

def _clip_transform_precision(geom, aoi_utm_polygon):
    """Clip in UTM, transform to 4326, set precision."""
    clipped = geom.intersection(aoi_utm_polygon)
    if clipped.is_empty:
        return []
    clipped = make_valid(clipped)
    polys = extract_polygons(clipped)
    out = []
    for poly in polys:
        p_4326 = _transform_to_4326(poly)
        p_4326 = shapely.set_precision(p_4326, grid_size=_PRECISION_DEG)
        # Precision reduction can collapse/sliver polygons; validate and run
        # the polygon-only filter again after rounding.
        p_4326 = make_valid(p_4326)
        out.extend(extract_polygons(p_4326))
    return out

def _feature_collection(features: list[dict]) -> dict:
    return {"type": "FeatureCollection", "features": features}

def build_polygons_feature_collection(
    rows: Iterable[dict], *, aoi_utm: tuple[float, float, float, float],
) -> dict:
    """Spec §8: /jobs/{id}/polygons payload. Clipped to aoi_utm, EPSG:4326."""
    aoi = box(*aoi_utm)
    features: list[dict] = []
    for r in rows:
        geom = wkb_loads(r["geometry_wkb"])
        for g4326 in _clip_transform_precision(geom, aoi):
            features.append({
                "type": "Feature",
                "geometry": mapping(g4326),
                "properties": {
                    "id": r["id"],
                    "score": r["score"],
                    "validation": r["validation"],
                },
            })
    return _feature_collection(features)

def build_polygons_geojson(
    rows: Iterable[dict], *, aoi_utm: tuple[float, float, float, float],
) -> str:
    """ProcessPool target: returns already-serialized JSON to avoid blocking
    FastAPI's event loop with a large json.dumps in the route."""
    return json.dumps(build_polygons_feature_collection(rows, aoi_utm=aoi_utm))

def build_nodata_feature_collection(
    rows: Iterable[dict], *, aoi_utm: tuple[float, float, float, float],
) -> dict:
    """Spec §8: /jobs/{id}/nodata payload. Clipped to aoi_utm, EPSG:4326."""
    aoi = box(*aoi_utm)
    features: list[dict] = []
    for r in rows:
        geom = wkb_loads(r["geometry_wkb"])
        for g4326 in _clip_transform_precision(geom, aoi):
            features.append({
                "type": "Feature",
                "geometry": mapping(g4326),
                "properties": {"reason": r["reason"]},
            })
    return _feature_collection(features)

def build_nodata_geojson(
    rows: Iterable[dict], *, aoi_utm: tuple[float, float, float, float],
) -> str:
    """ProcessPool target: returns already-serialized JSON."""
    return json.dumps(build_nodata_feature_collection(rows, aoi_utm=aoi_utm))
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/app/test_serialization.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add ki_geodaten/app/serialization.py tests/app/test_serialization.py
git commit -m "feat(app): WKB→GeoJSON with AOI clip, 25832→4326 transform, 1e-6 precision"
```

---

## Task 17: Safe-Center Polygon Helper

**Files:**
- Modify: `ki_geodaten/pipeline/tiler.py` (append)
- Test: `tests/pipeline/test_tiler.py` (append)

**See Spec §5.5** — NoData geometry is the **safe-center footprint** (size-2·margin square), NOT the full tile footprint. The orchestrator needs a single helper to derive that polygon from a `Tile` or `NodataTile`.

- [ ] **Step 1: Append failing tests**

```python
# tests/pipeline/test_tiler.py (append)
from shapely.geometry import Polygon
from ki_geodaten.pipeline.tiler import safe_center_polygon

def test_safe_center_polygon_medium():
    cfg = TileConfig.from_preset(TilePreset.MEDIUM)
    from affine import Affine
    # Tile at UTM (691000, 5336204.8) with 0.2 m px, top-left origin
    affine = Affine(0.2, 0, 691000.0, 0, -0.2, 5336204.8)
    t = Tile(
        array=np.zeros((cfg.size, cfg.size, 3), dtype=np.uint8),
        pixel_origin=(0, 0), size=cfg.size, center_margin=cfg.center_margin,
        affine=affine, tile_row=0, tile_col=0,
        nodata_mask=np.zeros((cfg.size, cfg.size), dtype=bool),
    )
    poly = safe_center_polygon(t)
    assert isinstance(poly, Polygon)
    minx, miny, maxx, maxy = poly.bounds
    # Margin = 320 px * 0.2 m = 64 m; safe center side = 384 px * 0.2 m = 76.8 m
    assert maxx - minx == pytest.approx(76.8)
    assert maxy - miny == pytest.approx(76.8)
    # Lower-left corner: col=320, row=704 → x = 691000 + 64 = 691064; y = 5336204.8 - 704*0.2 = 5336064
    assert minx == pytest.approx(691064.0)
    assert maxy == pytest.approx(5336204.8 - 64.0)
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/pipeline/test_tiler.py -v`
Expected: new test FAILs.

- [ ] **Step 3: Append implementation**

```python
# ki_geodaten/pipeline/tiler.py (append)
from shapely.geometry import Polygon as _ShpPolygon

def safe_center_polygon(tile: "Tile | NodataTile") -> _ShpPolygon:
    """Spec §5.5: return UTM polygon of the safe-center square (not full tile)."""
    m = tile.center_margin
    size = tile.size
    # Corners in pixel-space (col, row) for the safe-center square:
    corners_px = [
        (m, m),
        (size - m, m),
        (size - m, size - m),
        (m, size - m),
    ]
    corners_utm = [tile.affine * c for c in corners_px]
    return _ShpPolygon(corners_utm)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/pipeline/test_tiler.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add ki_geodaten/pipeline/tiler.py tests/pipeline/test_tiler.py
git commit -m "feat(tiler): safe_center_polygon helper for NoData persistence"
```

---

## Task 18: Worker Orchestrator — Per-Tile Commits & Error Handling

**Files:**
- Create: `ki_geodaten/worker/orchestrator.py`
- Test: `tests/worker/test_orchestrator.py`

**See Spec §6, §10** — per-tile try/except (`OOM` / `INFERENCE_ERROR` → `nodata_regions` with safe-center geometry), per-job try/except (status FAILED with stacktrace tail), per-tile commit to avoid long-running write transactions.

- [ ] **Step 1: Write failing test**

```python
# tests/worker/test_orchestrator.py
import numpy as np
import pytest
from affine import Affine
from pathlib import Path
from shapely.geometry import Polygon, box
from shapely.wkb import dumps as wkb_dumps

from ki_geodaten.jobs.store import (
    init_schema, insert_job, get_job, get_polygons_for_job, get_nodata_for_job,
)
from ki_geodaten.models import JobStatus, TilePreset
from ki_geodaten.pipeline.segmenter import MaskResult
from ki_geodaten.pipeline.tiler import Tile, TileConfig, NodataTile
from ki_geodaten.worker.orchestrator import run_job

class _StubSegmenter:
    def __init__(self, behaviours):
        self._behaviours = list(behaviours)
        self.encoder_token_count = lambda s: len(s.split())
    def predict(self, tile, prompt):
        b = self._behaviours.pop(0)
        if isinstance(b, BaseException):
            raise b
        return b

def _tile(row, col):
    cfg = TileConfig.from_preset(TilePreset.MEDIUM)
    affine = Affine(0.2, 0, 691000.0 + col * cfg.tile_step * 0.2,
                    0, -0.2, 5336204.8 - row * cfg.tile_step * 0.2)
    return Tile(
        array=np.zeros((cfg.size, cfg.size, 3), dtype=np.uint8),
        pixel_origin=(row * cfg.tile_step, col * cfg.tile_step),
        size=cfg.size, center_margin=cfg.center_margin,
        affine=affine, tile_row=row, tile_col=col,
        nodata_mask=np.zeros((cfg.size, cfg.size), dtype=bool),
    )

def _nodata_tile(row, col):
    cfg = TileConfig.from_preset(TilePreset.MEDIUM)
    affine = Affine(0.2, 0, 691000.0 + col * cfg.tile_step * 0.2,
                    0, -0.2, 5336204.8 - row * cfg.tile_step * 0.2)
    return NodataTile(
        tile_row=row, tile_col=col,
        pixel_origin=(row * cfg.tile_step, col * cfg.tile_step),
        size=cfg.size, center_margin=cfg.center_margin,
        affine=affine,
    )

def _setup_job(tmp_path: Path) -> Path:
    db = tmp_path / "j.db"
    init_schema(db)
    insert_job(
        db, job_id="j1", prompt="building",
        bbox_wgs84=[11.0, 48.0, 11.01, 48.01],
        bbox_utm_snapped=[691000.0, 5335000.0, 692000.0, 5336000.0],
        tile_preset=TilePreset.MEDIUM,
    )
    return db

def test_orchestrator_happy_path_marks_ready(tmp_path, monkeypatch):
    db = _setup_job(tmp_path)
    mask = np.zeros((1024, 1024), dtype=bool)
    mask[500:524, 500:524] = True
    mr = MaskResult(mask=mask, score=0.9, box_pixel=(500, 500, 524, 524))
    seg = _StubSegmenter([[mr]])
    tiles = [_tile(0, 0)]

    def fake_download(*a, **kw):
        return Path("fake.vrt")
    def fake_iter_tiles(*a, **kw):
        return iter(tiles)

    monkeypatch.setattr("ki_geodaten.worker.orchestrator.download_dop20", fake_download)
    monkeypatch.setattr("ki_geodaten.worker.orchestrator.iter_tiles", fake_iter_tiles)

    run_job(db, job_id="j1", segmenter=seg, data_root=tmp_path,
            wms_url="", layer="by_dop20c", max_pixels=6000,
            wms_version="1.1.1", fmt="image/png", crs="EPSG:25832",
            origin_x=0.0, origin_y=0.0, min_polygon_area_m2=0.01,
            safe_center_nodata_threshold=0.0)

    job = get_job(db, "j1")
    assert job["status"] == JobStatus.READY_FOR_REVIEW
    assert job["tile_total"] == 1
    assert job["tile_completed"] == 1
    assert len(get_polygons_for_job(db, "j1")) == 1

def test_orchestrator_records_oom_as_nodata(tmp_path, monkeypatch):
    # Use a dedicated stub exception and unconditionally override _is_cuda_oom,
    # so the test is deterministic whether torch+CUDA is available or not.
    db = _setup_job(tmp_path)
    class FakeOOM(RuntimeError): pass
    monkeypatch.setattr(
        "ki_geodaten.worker.orchestrator._is_cuda_oom",
        lambda e: isinstance(e, FakeOOM),
    )
    oom = FakeOOM("simulated")
    seg = _StubSegmenter([oom])
    tiles = [_tile(0, 0)]
    monkeypatch.setattr("ki_geodaten.worker.orchestrator.download_dop20",
                        lambda *a, **kw: Path("fake.vrt"))
    monkeypatch.setattr("ki_geodaten.worker.orchestrator.iter_tiles",
                        lambda *a, **kw: iter(tiles))

    run_job(db, job_id="j1", segmenter=seg, data_root=tmp_path,
            wms_url="", layer="by_dop20c", max_pixels=6000,
            wms_version="1.1.1", fmt="image/png", crs="EPSG:25832",
            origin_x=0.0, origin_y=0.0, min_polygon_area_m2=0.01,
            safe_center_nodata_threshold=0.0)

    nd = get_nodata_for_job(db, "j1")
    assert len(nd) == 1
    assert nd[0]["reason"] == "OOM"
    assert get_job(db, "j1")["status"] == JobStatus.READY_FOR_REVIEW

def test_orchestrator_records_inference_error_as_nodata(tmp_path, monkeypatch):
    db = _setup_job(tmp_path)
    seg = _StubSegmenter([ValueError("boom")])
    tiles = [_tile(0, 0)]
    monkeypatch.setattr("ki_geodaten.worker.orchestrator.download_dop20",
                        lambda *a, **kw: Path("fake.vrt"))
    monkeypatch.setattr("ki_geodaten.worker.orchestrator.iter_tiles",
                        lambda *a, **kw: iter(tiles))

    run_job(db, job_id="j1", segmenter=seg, data_root=tmp_path,
            wms_url="", layer="by_dop20c", max_pixels=6000,
            wms_version="1.1.1", fmt="image/png", crs="EPSG:25832",
            origin_x=0.0, origin_y=0.0, min_polygon_area_m2=0.01,
            safe_center_nodata_threshold=0.0)

    nd = get_nodata_for_job(db, "j1")
    assert nd[0]["reason"] == "INFERENCE_ERROR"

def test_orchestrator_records_nodata_tile_without_invoking_segmenter(tmp_path, monkeypatch):
    db = _setup_job(tmp_path)
    seg = _StubSegmenter([])  # should NEVER be called
    tiles = [_nodata_tile(0, 0)]
    monkeypatch.setattr("ki_geodaten.worker.orchestrator.download_dop20",
                        lambda *a, **kw: Path("fake.vrt"))
    monkeypatch.setattr("ki_geodaten.worker.orchestrator.iter_tiles",
                        lambda *a, **kw: iter(tiles))

    run_job(db, job_id="j1", segmenter=seg, data_root=tmp_path,
            wms_url="", layer="by_dop20c", max_pixels=6000,
            wms_version="1.1.1", fmt="image/png", crs="EPSG:25832",
            origin_x=0.0, origin_y=0.0, min_polygon_area_m2=0.01,
            safe_center_nodata_threshold=0.0)

    nd = get_nodata_for_job(db, "j1")
    assert nd[0]["reason"] == "NODATA_PIXELS"

def test_orchestrator_catches_download_error_marks_failed(tmp_path, monkeypatch):
    from ki_geodaten.pipeline.dop_client import DopDownloadError
    db = _setup_job(tmp_path)
    def fail_download(*a, **kw):
        raise DopDownloadError("DOP_TIMEOUT")
    monkeypatch.setattr("ki_geodaten.worker.orchestrator.download_dop20", fail_download)

    run_job(db, job_id="j1", segmenter=_StubSegmenter([]), data_root=tmp_path,
            wms_url="", layer="by_dop20c", max_pixels=6000,
            wms_version="1.1.1", fmt="image/png", crs="EPSG:25832",
            origin_x=0.0, origin_y=0.0, min_polygon_area_m2=0.01,
            safe_center_nodata_threshold=0.0)

    job = get_job(db, "j1")
    assert job["status"] == JobStatus.FAILED
    assert job["error_reason"] == "DOP_TIMEOUT"
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/worker/test_orchestrator.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# ki_geodaten/worker/orchestrator.py
from __future__ import annotations
import json
import logging
import shutil
import traceback
from pathlib import Path

import geopandas as gpd
from shapely.wkb import dumps as wkb_dumps

from ki_geodaten.jobs.store import (
    update_status, insert_polygons, insert_nodata_region,
    increment_tile_completed, increment_tile_failed,
)
from ki_geodaten.models import JobStatus, TilePreset, ErrorReason, NoDataReason
from ki_geodaten.pipeline.dop_client import download_dop20, DopDownloadError
from ki_geodaten.pipeline.tiler import (
    TileConfig, Tile, NodataTile, iter_tiles, safe_center_polygon,
)
from ki_geodaten.pipeline.merger import keep_center_only, masks_to_polygons

logger = logging.getLogger(__name__)

def _is_cuda_oom(exc: BaseException) -> bool:
    try:
        import torch
    except Exception:
        return False
    return isinstance(exc, torch.cuda.OutOfMemoryError)

def _empty_cuda_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

def _clear_exception_context(exc: BaseException) -> None:
    """Break traceback/frame references before CUDA cache flush.

    In Python 3, `except Exception as exc` keeps `exc.__traceback__` alive until
    the except block exits. If the failing SAM frame owns temporary GPU tensors,
    `torch.cuda.empty_cache()` cannot release that memory while the traceback is
    still referenced.
    """
    exc.__traceback__ = None
    exc.__context__ = None
    exc.__cause__ = None

def _traceback_tail(exc: BaseException, n: int = 20) -> str:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    lines = tb.splitlines()
    return "\n".join(lines[-n:])

def _persist_polygons_for_tile(
    db_path: Path, job_id: str, gdf: gpd.GeoDataFrame,
) -> None:
    if gdf is None or len(gdf) == 0:
        return
    rows = [
        {
            "geometry_wkb": wkb_dumps(row.geometry),
            "score": float(row["score"]),
            "source_tile_row": int(row["source_tile_row"]),
            "source_tile_col": int(row["source_tile_col"]),
        }
        for _, row in gdf.iterrows()
    ]
    insert_polygons(db_path, job_id, rows)

def _persist_safe_center_nodata(
    db_path: Path, job_id: str, tile, reason: NoDataReason,
) -> None:
    poly = safe_center_polygon(tile)
    insert_nodata_region(
        db_path, job_id,
        geometry_wkb=wkb_dumps(poly),
        tile_row=tile.tile_row, tile_col=tile.tile_col,
        reason=str(reason),
    )

def run_job(
    db_path: Path, *, job_id: str, segmenter, data_root: Path,
    wms_url: str, layer: str, max_pixels: int,
    wms_version: str, fmt: str, crs: str,
    origin_x: float, origin_y: float, min_polygon_area_m2: float,
    safe_center_nodata_threshold: float,
) -> None:
    """Spec §6: DOWNLOADING → INFERRING → READY_FOR_REVIEW (or FAILED).
    Per-tile commits; segmenter is injected so tests can stub it."""
    from ki_geodaten.jobs.store import get_job as _get_job
    job = _get_job(db_path, job_id)
    if job is None:
        return
    prompt = job["prompt"]
    preset = TilePreset(job["tile_preset"])
    cfg = TileConfig.from_preset(preset)
    out_dir = data_root / "dop" / job_id

    try:
        update_status(db_path, job_id, JobStatus.DOWNLOADING, set_started=True)
        aoi_utm = tuple(json.loads(job["bbox_utm_snapped"]))
        # Download bbox = aoi + margin (caller had already prepared via prepare_download_bbox
        # when the job was created; we re-apply here for robustness).
        from ki_geodaten.pipeline.dop_client import prepare_download_bbox
        prepared = prepare_download_bbox(
            *aoi_utm, preset=preset,
            origin_x=origin_x, origin_y=origin_y,
        )
        vrt_path = download_dop20(
            prepared.download_bbox, out_dir=out_dir,
            wms_url=wms_url, layer=layer, wms_version=wms_version,
            fmt=fmt, crs=crs, max_pixels=max_pixels,
            origin_x=origin_x, origin_y=origin_y,
        )
        update_status(db_path, job_id, JobStatus.INFERRING, dop_vrt_path=str(vrt_path))

        # Spec §5.2 requires lazy tile iteration. Counting via iter_grid() is
        # pure arithmetic on src.width/src.height — no pixel reads.
        import rasterio as _rio
        from ki_geodaten.pipeline.tiler import iter_grid as _iter_grid
        with _rio.open(vrt_path) as _src:
            tile_total = sum(1 for _ in _iter_grid(_src, cfg))
        update_status(db_path, job_id, JobStatus.INFERRING, tile_total=tile_total)

        for tile in iter_tiles(
            vrt_path, cfg,
            safe_center_nodata_threshold=safe_center_nodata_threshold,
        ):
            if isinstance(tile, NodataTile):
                _persist_safe_center_nodata(db_path, job_id, tile,
                                            NoDataReason.NODATA_PIXELS)
                increment_tile_failed(db_path, job_id)
                continue
            try:
                masks = segmenter.predict(tile, prompt)
            except Exception as exc:   # noqa: BLE001 — we intentionally cover all predict failures
                reason = NoDataReason.OOM if _is_cuda_oom(exc) else NoDataReason.INFERENCE_ERROR
                _persist_safe_center_nodata(db_path, job_id, tile, reason)
                increment_tile_failed(db_path, job_id)
                _clear_exception_context(exc)
                del exc
                _empty_cuda_cache()
                continue

            try:
                kept = keep_center_only(masks, tile)
                gdf = masks_to_polygons(kept, tile, min_area_m2=min_polygon_area_m2)
                _persist_polygons_for_tile(db_path, job_id, gdf)
                increment_tile_completed(db_path, job_id)
            except Exception as exc:
                logger.exception("INVALID_GEOMETRY tile=%s/%s", tile.tile_row, tile.tile_col)
                _persist_safe_center_nodata(db_path, job_id, tile,
                                            NoDataReason.INVALID_GEOMETRY)
                increment_tile_failed(db_path, job_id)
                _clear_exception_context(exc)
                del exc
            finally:
                _empty_cuda_cache()

        update_status(db_path, job_id, JobStatus.READY_FOR_REVIEW, set_finished=True)

    except DopDownloadError as exc:
        reason = str(exc) if str(exc) in ("DOP_TIMEOUT", "DOP_HTTP_ERROR") else ErrorReason.DOP_HTTP_ERROR
        update_status(
            db_path, job_id, JobStatus.FAILED,
            error_reason=str(reason), error_message=_traceback_tail(exc),
            set_finished=True,
        )
        shutil.rmtree(out_dir, ignore_errors=True)
    except Exception as exc:
        update_status(
            db_path, job_id, JobStatus.FAILED,
            error_reason=str(ErrorReason.INFERENCE_ERROR),
            error_message=_traceback_tail(exc), set_finished=True,
        )
        shutil.rmtree(out_dir, ignore_errors=True)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/worker/test_orchestrator.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add ki_geodaten/worker/orchestrator.py tests/worker/test_orchestrator.py
git commit -m "feat(worker): orchestrator with per-tile commits and three error layers"
```

---

## Task 19: Worker Loop — Poll, Startup Cleanup, Self-Restart

**Files:**
- Create: `ki_geodaten/worker/loop.py`
- Test: `tests/worker/test_loop.py`

**See Spec §10.3** — startup hook marks DOWNLOADING/INFERRING jobs as FAILED, rigorously rmtree's orphan dirs under `data/dop/`. Worker exits cleanly after `MAX_JOBS_PER_WORKER` so the supervisor script re-launches it.

- [ ] **Step 1: Write failing test**

```python
# tests/worker/test_loop.py
import pytest
from pathlib import Path
from ki_geodaten.jobs.store import init_schema, insert_job, get_job, update_status
from ki_geodaten.models import JobStatus, TilePreset
from ki_geodaten.worker.loop import startup_cleanup, run_forever

def test_startup_marks_incomplete_jobs_failed(tmp_path):
    db = tmp_path / "j.db"
    init_schema(db)
    insert_job(db, job_id="j1", prompt="p", bbox_wgs84=[0,0,1,1],
               bbox_utm_snapped=[0,0,1,1], tile_preset=TilePreset.MEDIUM)
    update_status(db, "j1", JobStatus.DOWNLOADING, set_started=True)
    startup_cleanup(db, data_root=tmp_path)
    j = get_job(db, "j1")
    assert j["status"] == JobStatus.FAILED
    assert j["error_reason"] == "WORKER_RESTARTED"

def test_startup_rmtree_removes_orphan_dirs(tmp_path):
    db = tmp_path / "j.db"
    init_schema(db)
    # Job that does NOT exist → its dop-dir is an orphan
    orphan = tmp_path / "dop" / "ghost-job"
    orphan.mkdir(parents=True)
    (orphan / "chunk_0_0.tif").write_bytes(b"x")
    # Active job dir must be preserved
    insert_job(db, job_id="j_active", prompt="p", bbox_wgs84=[0,0,1,1],
               bbox_utm_snapped=[0,0,1,1], tile_preset=TilePreset.MEDIUM)
    update_status(db, "j_active", JobStatus.DOWNLOADING, set_started=True)
    active_dir = tmp_path / "dop" / "j_active"
    active_dir.mkdir(parents=True)
    (active_dir / "chunk_0_0.tif").write_bytes(b"y")

    # First call re-marks j_active as FAILED; next pass removes its dir because it's now inactive
    startup_cleanup(db, data_root=tmp_path)
    # After the DB-pass, j_active is FAILED. Re-run the disk pass (simulating after restart):
    startup_cleanup(db, data_root=tmp_path)
    assert not orphan.exists()
    assert not active_dir.exists()

def test_run_forever_exits_after_max_jobs(tmp_path, monkeypatch):
    db = tmp_path / "j.db"
    init_schema(db)
    insert_job(db, job_id="j1", prompt="p", bbox_wgs84=[0,0,1,1],
               bbox_utm_snapped=[0,0,1,1], tile_preset=TilePreset.MEDIUM)
    insert_job(db, job_id="j2", prompt="p", bbox_wgs84=[0,0,1,1],
               bbox_utm_snapped=[0,0,1,1], tile_preset=TilePreset.MEDIUM)
    calls: list[str] = []

    def fake_run_job(*a, **kw):
        calls.append(kw["job_id"])
        update_status(db, kw["job_id"], JobStatus.READY_FOR_REVIEW, set_finished=True)

    monkeypatch.setattr("ki_geodaten.worker.loop.run_job", fake_run_job)

    class _StubSegmenter: ...
    run_forever(
        db_path=db, data_root=tmp_path,
        segmenter_factory=lambda: _StubSegmenter(),
        wms_url="", layer="by_dop20c", max_pixels=6000,
        wms_version="1.1.1", fmt="image/png", crs="EPSG:25832",
        origin_x=0.0, origin_y=0.0,
        min_polygon_area_m2=1.0, safe_center_nodata_threshold=0.0,
        max_jobs=2, poll_interval=0.01, idle_exit_after=2,
    )
    assert calls == ["j1", "j2"]
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/worker/test_loop.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# ki_geodaten/worker/loop.py
from __future__ import annotations
import logging
import shutil
import time
from pathlib import Path
from typing import Callable

from ki_geodaten.jobs.store import (
    abort_incomplete_jobs_on_startup, claim_next_pending_job,
    connect,
)
from ki_geodaten.worker.orchestrator import run_job

logger = logging.getLogger(__name__)

def _active_job_ids(db_path: Path) -> set[str]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id FROM jobs WHERE status IN ('PENDING','DOWNLOADING','INFERRING')"
        ).fetchall()
    return {r["id"] for r in rows}

def startup_cleanup(db_path: Path, *, data_root: Path) -> None:
    """Spec §10.3: 1) abort DOWNLOADING/INFERRING 2) rmtree orphan disk dirs."""
    abort_incomplete_jobs_on_startup(db_path)
    active = _active_job_ids(db_path)
    dop_root = data_root / "dop"
    if not dop_root.exists():
        return
    for child in dop_root.iterdir():
        if not child.is_dir():
            continue
        if child.name not in active:
            shutil.rmtree(child, ignore_errors=True)

def run_forever(
    *, db_path: Path, data_root: Path,
    segmenter_factory: Callable[[], object],
    wms_url: str, layer: str, max_pixels: int,
    wms_version: str, fmt: str, crs: str,
    origin_x: float, origin_y: float,
    min_polygon_area_m2: float, safe_center_nodata_threshold: float,
    max_jobs: int, poll_interval: float,
    idle_exit_after: int | None = None,
) -> None:
    """Poll loop. Exits cleanly after max_jobs (supervisor restarts us).
    idle_exit_after: exit after N consecutive empty polls (for tests)."""
    startup_cleanup(db_path, data_root=data_root)
    segmenter = segmenter_factory()
    processed = 0
    idle_polls = 0
    while processed < max_jobs:
        job = claim_next_pending_job(db_path)
        if job is None:
            idle_polls += 1
            if idle_exit_after is not None and idle_polls >= idle_exit_after:
                return
            time.sleep(poll_interval)
            continue
        idle_polls = 0
        try:
            run_job(
                db_path, job_id=job["id"], segmenter=segmenter,
                data_root=data_root,
                wms_url=wms_url, layer=layer, max_pixels=max_pixels,
                wms_version=wms_version, fmt=fmt, crs=crs,
                origin_x=origin_x, origin_y=origin_y,
                min_polygon_area_m2=min_polygon_area_m2,
                safe_center_nodata_threshold=safe_center_nodata_threshold,
            )
        except Exception:
            logger.exception("job crashed outside orchestrator try/except: %s", job["id"])
        processed += 1

def main() -> None:   # pragma: no cover
    import os
    from ki_geodaten.config import Settings
    from ki_geodaten.pipeline.segmenter import Sam3Segmenter
    settings = Settings()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    db_path = Path("data/jobs.db")
    run_forever(
        db_path=db_path, data_root=Path("data"),
        segmenter_factory=lambda: Sam3Segmenter(
            settings.SAM3_CHECKPOINT,
            iou_threshold=settings.LOCAL_MASK_NMS_IOU,
            containment_ratio=settings.LOCAL_MASK_CONTAINMENT_RATIO,
        ),
        wms_url=settings.WMS_URL, layer=settings.WMS_LAYER,
        max_pixels=settings.WMS_MAX_PIXELS,
        wms_version=settings.WMS_VERSION,
        fmt=settings.WMS_FORMAT, crs=settings.WMS_CRS,
        origin_x=settings.WMS_GRID_ORIGIN_X, origin_y=settings.WMS_GRID_ORIGIN_Y,
        min_polygon_area_m2=settings.MIN_POLYGON_AREA_M2,
        safe_center_nodata_threshold=settings.SAFE_CENTER_NODATA_THRESHOLD,
        max_jobs=settings.MAX_JOBS_PER_WORKER,
        poll_interval=settings.WORKER_POLL_INTERVAL_SEC,
    )

if __name__ == "__main__":   # pragma: no cover
    main()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/worker/test_loop.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add ki_geodaten/worker/loop.py tests/worker/test_loop.py
git commit -m "feat(worker): poll loop, startup DB+disk cleanup, self-restart after N jobs"
```

---

## Task 20: Retention Cleanup (`jobs/retention.py`)

**Files:**
- Create: `ki_geodaten/jobs/retention.py`
- Test: `tests/jobs/test_retention.py`

**See Spec §NFR-7** — DELETE FAILED/EXPORTED jobs older than RETENTION_DAYS; CASCADE drops polygons + nodata_regions; unlink `.gpkg` files; `VACUUM`.

- [ ] **Step 1: Write failing test**

```python
# tests/jobs/test_retention.py
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest
from ki_geodaten.jobs.store import (
    connect, init_schema, insert_job, insert_polygons, update_status, get_job,
)
from ki_geodaten.jobs.retention import cleanup_old_jobs
from ki_geodaten.models import JobStatus, TilePreset

def _set_finished_at(db_path: Path, job_id: str, when: datetime) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE jobs SET finished_at = ? WHERE id = ?",
            (when.isoformat(), job_id),
        )

def test_cleanup_deletes_old_failed_and_exported(tmp_path):
    db = tmp_path / "j.db"
    results = tmp_path / "results"
    results.mkdir()
    init_schema(db)
    for jid in ("old_failed", "old_exported", "recent_exported", "active"):
        insert_job(db, job_id=jid, prompt="p",
                   bbox_wgs84=[0,0,1,1], bbox_utm_snapped=[0,0,1,1],
                   tile_preset=TilePreset.MEDIUM)
    update_status(db, "old_failed", JobStatus.FAILED,
                  error_reason="DOP_TIMEOUT", set_finished=True)
    update_status(db, "old_exported", JobStatus.EXPORTED, set_finished=True,
                  gpkg_path=str(results / "old_exported.gpkg"))
    (results / "old_exported.gpkg").write_bytes(b"gpkg")
    update_status(db, "recent_exported", JobStatus.EXPORTED, set_finished=True,
                  gpkg_path=str(results / "recent_exported.gpkg"))
    (results / "recent_exported.gpkg").write_bytes(b"gpkg")

    now = datetime.now(timezone.utc)
    _set_finished_at(db, "old_failed", now - timedelta(days=10))
    _set_finished_at(db, "old_exported", now - timedelta(days=10))
    _set_finished_at(db, "recent_exported", now - timedelta(days=1))

    deleted = cleanup_old_jobs(db, results_dir=results, retention_days=7)
    assert set(deleted) == {"old_failed", "old_exported"}
    assert get_job(db, "old_failed") is None
    assert get_job(db, "old_exported") is None
    assert get_job(db, "recent_exported") is not None
    assert get_job(db, "active") is not None
    assert not (results / "old_exported.gpkg").exists()
    assert (results / "recent_exported.gpkg").exists()

def test_cleanup_cascades_to_polygons(tmp_path):
    db = tmp_path / "j.db"
    results = tmp_path / "results"
    results.mkdir()
    init_schema(db)
    insert_job(db, job_id="old", prompt="p",
               bbox_wgs84=[0,0,1,1], bbox_utm_snapped=[0,0,1,1],
               tile_preset=TilePreset.MEDIUM)
    insert_polygons(db, "old", [
        {"geometry_wkb": b"a", "score": 0.9, "source_tile_row": 0, "source_tile_col": 0}
    ])
    update_status(db, "old", JobStatus.FAILED, error_reason="OOM", set_finished=True)
    _set_finished_at(db, "old", datetime.now(timezone.utc) - timedelta(days=10))
    cleanup_old_jobs(db, results_dir=results, retention_days=7)
    with connect(db) as conn:
        n = conn.execute("SELECT COUNT(*) FROM polygons WHERE job_id='old'").fetchone()[0]
    assert n == 0

def test_cleanup_batches_large_delete_set(tmp_path):
    # Regression: a single IN (?, ?, ...) can exceed SQLITE_MAX_VARIABLE_NUMBER
    db = tmp_path / "j.db"
    results = tmp_path / "results"
    results.mkdir()
    init_schema(db)
    old_when = datetime.now(timezone.utc) - timedelta(days=10)
    for i in range(1050):
        jid = f"old_{i}"
        insert_job(db, job_id=jid, prompt="p",
                   bbox_wgs84=[0,0,1,1], bbox_utm_snapped=[0,0,1,1],
                   tile_preset=TilePreset.MEDIUM)
        update_status(db, jid, JobStatus.FAILED,
                      error_reason="DOP_TIMEOUT", set_finished=True)
        _set_finished_at(db, jid, old_when)
    deleted = cleanup_old_jobs(db, results_dir=results, retention_days=7)
    assert len(deleted) == 1050
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/jobs/test_retention.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# ki_geodaten/jobs/retention.py
from __future__ import annotations
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ki_geodaten.jobs.store import connect

logger = logging.getLogger(__name__)
_DELETE_BATCH_SIZE = 900

def cleanup_old_jobs(
    db_path: Path, *, results_dir: Path, retention_days: int,
) -> list[str]:
    """Spec §NFR-7. Returns list of deleted job IDs."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
    with connect(db_path) as conn:
        conn.execute("BEGIN")
        rows = conn.execute(
            "SELECT id, gpkg_path FROM jobs"
            " WHERE status IN ('FAILED','EXPORTED') AND finished_at IS NOT NULL"
            " AND finished_at < ?",
            (cutoff,),
        ).fetchall()
        ids = [r["id"] for r in rows]
        gpkg_paths = [r["gpkg_path"] for r in rows if r["gpkg_path"]]
        for i in range(0, len(ids), _DELETE_BATCH_SIZE):
            batch = ids[i:i + _DELETE_BATCH_SIZE]
            placeholders = ",".join("?" * len(batch))
            conn.execute(f"DELETE FROM jobs WHERE id IN ({placeholders})", batch)
        conn.execute("COMMIT")
        conn.execute("VACUUM")
    for path_str in gpkg_paths:
        p = Path(path_str)
        try:
            p.unlink(missing_ok=True)
        except OSError:
            logger.warning("retention: failed to unlink %s", p)
    return ids
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/jobs/test_retention.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add ki_geodaten/jobs/retention.py tests/jobs/test_retention.py
git commit -m "feat(jobs): retention cleanup for FAILED/EXPORTED jobs"
```

---

## Task 21: FastAPI App Factory & Process Pool Executor

**Files:**
- Create: `ki_geodaten/app/main.py`
- Test: `tests/app/test_main.py`

**See Spec §8** — dedicated `ProcessPoolExecutor` (small; 2 workers) for GeoJSON serialization to keep Event-Loop *and* Web-Threadpool free. Lifespan initializes DB schema and starts/stops the executor.

**Injection hooks:** `create_app(executor_factory=..., token_counter=...)` both accept optional overrides so tests can substitute a `ThreadPoolExecutor` (avoiding slow subprocess spawn on every test) and a deterministic tokenizer stub. Production leaves them at defaults.

**Token-counter in production:** per Spec §5.3/§8 the limit must be checked on the **real, templatized encoder sequence**, not whitespace-split chars. The app process must not build the SAM vision model and must not allocate CUDA memory. It uses `Sam3TextTokenCounter` from Task 12A, which loads only tokenizer/text-encoding state. If loading fails (e.g. sam3 not installed yet during development), it falls back to whitespace split **and logs a warning** — *never* silent, because the spec invariant is that an over-long prompt must reach HTTP 422.

- [ ] **Step 1: Write failing test**

```python
# tests/app/test_main.py
from fastapi.testclient import TestClient
from pathlib import Path
from ki_geodaten.app.main import create_app

def test_app_starts_and_serves_index(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = create_app()
    with TestClient(app) as c:
        r = c.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

def test_app_lifespan_initializes_db(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = create_app()
    with TestClient(app) as c:
        pass
    assert (tmp_path / "data" / "jobs.db").exists()

def test_app_has_geojson_executor(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = create_app()
    with TestClient(app) as c:
        pass
        assert hasattr(app.state, "geojson_executor")

def test_executor_factory_override_used(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    monkeypatch.chdir(tmp_path)
    sentinel = ThreadPoolExecutor(max_workers=1)
    app = create_app(executor_factory=lambda: sentinel)
    with TestClient(app) as c:
        assert app.state.geojson_executor is sentinel

def test_token_counter_override_used(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = create_app(token_counter=lambda s: 42)
    with TestClient(app):
        assert app.state.token_counter("anything") == 42
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/app/test_main.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# ki_geodaten/app/main.py
from __future__ import annotations
import logging
import threading
from concurrent.futures import Executor, ProcessPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.responses import HTMLResponse

from ki_geodaten.config import Settings
from ki_geodaten.jobs.store import init_schema

logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).parent

TokenCounter = Callable[[str], int]
ExecutorFactory = Callable[[], Executor]

def _data_paths(root: Path) -> tuple[Path, Path, Path]:
    data_root = root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    (data_root / "dop").mkdir(exist_ok=True)
    (data_root / "results").mkdir(exist_ok=True)
    return data_root, data_root / "jobs.db", data_root / "results"

def _default_executor_factory() -> Executor:
    return ProcessPoolExecutor(max_workers=2)

def _default_token_counter(settings: Settings) -> TokenCounter:
    """Spec §5.3/§8: token limit must be checked on the FINAL encoder sequence.
    Prefer loading the real SAM text tokenizer on CPU. Fall back with a loud
    warning — we must NEVER silently under-count tokens."""
    try:
        from ki_geodaten.pipeline.segmenter import Sam3TextTokenCounter
        counter = Sam3TextTokenCounter(settings.SAM3_CHECKPOINT)
        counter.load()
        return counter
    except Exception as exc:   # noqa: BLE001 — covers missing sam3 in dev
        logger.warning(
            "Falling back to whitespace-split token counter "
            "(real encoder unavailable: %s). Spec §5.3 invariant NOT enforced.",
            exc,
        )
        return lambda s: len(s.split())

def lifespan_factory(
    executor_factory: ExecutorFactory,
    token_counter: TokenCounter | None,
):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings: Settings = app.state.settings
        data_root, db_path, results_dir = _data_paths(Path.cwd())
        init_schema(db_path)
        app.state.db_path = db_path
        app.state.data_root = data_root
        app.state.results_dir = results_dir
        app.state.export_lock = threading.Lock()
        app.state.geojson_cache = {}             # (job_id, revision, target) -> dict
        app.state.geojson_executor = executor_factory()
        app.state.token_counter = (
            token_counter if token_counter is not None
            else _default_token_counter(settings)
        )
        try:
            yield
        finally:
            app.state.geojson_executor.shutdown(wait=False, cancel_futures=True)
    return lifespan

def create_app(
    settings: Settings | None = None,
    *,
    executor_factory: ExecutorFactory = _default_executor_factory,
    token_counter: TokenCounter | None = None,
) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(
        title="Text-to-Polygon Pipeline",
        lifespan=lifespan_factory(executor_factory, token_counter),
    )
    app.state.settings = settings

    templates = Jinja2Templates(directory=str(_BASE_DIR / "templates"))
    app.state.templates = templates
    app.mount(
        "/static",
        StaticFiles(directory=str(_BASE_DIR / "static")),
        name="static",
    )

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    from ki_geodaten.app.routes.jobs import router as jobs_router
    from ki_geodaten.app.routes.geojson import router as geojson_router
    app.include_router(jobs_router)
    app.include_router(geojson_router)
    return app

app = create_app()   # uvicorn entrypoint
```

Also create a minimal placeholder template now (will be expanded in Task 27):

```html
<!-- ki_geodaten/app/templates/index.html -->
<!doctype html><title>Text-to-Polygon</title><h1>ki-geodaten</h1>
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/app/test_main.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ki_geodaten/app/main.py ki_geodaten/app/templates/index.html tests/app/test_main.py
git commit -m "feat(app): FastAPI factory, lifespan, ProcessPoolExecutor for GeoJSON"
```

---

## Task 22: `POST /jobs` — Bayern Fence + UTM Area Check + Token Limit

**Files:**
- Create: `ki_geodaten/app/routes/jobs.py`
- Test: `tests/app/test_routes_jobs.py`

**See Spec §8** — geographic fence on `BAYERN_BBOX_WGS84`, area check on **EPSG:25832 bbox** (NOT WGS84 degrees), prompt length validated on final templatized encoder sequence.

Token count checking reads `app.state.token_counter`, which is always set by `lifespan` (production: real SAM tokenizer on CPU, with a warning-logged whitespace fallback if sam3 isn't installed; tests: inject via `create_app(token_counter=...)`, see Task 21).

- [ ] **Step 1: Write failing test**

```python
# tests/app/test_routes_jobs.py
import uuid
import pytest
from fastapi.testclient import TestClient
from ki_geodaten.app.main import create_app
from ki_geodaten.jobs.store import get_job

@pytest.fixture
def client(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    monkeypatch.chdir(tmp_path)
    # Use a ThreadPoolExecutor in tests (ProcessPool subprocess spawn is slow
    # on Windows and the executor doesn't need process isolation here).
    # Inject a deterministic 1-token-per-word counter for predictable limits.
    app = create_app(
        executor_factory=lambda: ThreadPoolExecutor(max_workers=2),
        token_counter=lambda s: len(s.split()),
    )
    with TestClient(app) as c:
        yield c, app

def _munich_bbox():
    return [11.55, 48.13, 11.56, 48.14]  # ~800m × ~1.1km; just under 1 km²

def _too_big_bbox():
    return [11.55, 48.13, 11.60, 48.20]  # way over 1 km²

def _outside_bayern_bbox():
    return [13.30, 52.40, 13.31, 52.41]  # Berlin

def test_post_jobs_accepts_valid(client):
    c, app = client
    r = c.post("/jobs", json={"prompt": "building", "bbox_wgs84": _munich_bbox()})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "PENDING"
    uuid.UUID(body["id"])
    assert get_job(app.state.db_path, body["id"]) is not None

def test_post_jobs_defaults_preset_medium(client):
    c, app = client
    r = c.post("/jobs", json={"prompt": "car", "bbox_wgs84": _munich_bbox()})
    jid = r.json()["id"]
    job = get_job(app.state.db_path, jid)
    assert job["tile_preset"] == "medium"

def test_post_jobs_rejects_outside_bayern(client):
    c, _ = client
    r = c.post("/jobs", json={"prompt": "building", "bbox_wgs84": _outside_bayern_bbox()})
    assert r.status_code == 422

def test_post_jobs_rejects_area_over_1sqkm(client):
    c, _ = client
    r = c.post("/jobs", json={"prompt": "building", "bbox_wgs84": _too_big_bbox()})
    assert r.status_code == 422

def test_post_jobs_rejects_empty_prompt_after_strip(client):
    c, _ = client
    r = c.post("/jobs", json={"prompt": "   ", "bbox_wgs84": _munich_bbox()})
    assert r.status_code == 422

def test_post_jobs_rejects_token_count_over_limit(client):
    c, app = client
    app.state.token_counter = lambda s: 999
    r = c.post("/jobs", json={"prompt": "solar panel", "bbox_wgs84": _munich_bbox()})
    assert r.status_code == 422

def test_post_jobs_rejects_inverted_bbox(client):
    c, _ = client
    r = c.post("/jobs", json={"prompt": "building", "bbox_wgs84": [11.6, 48.2, 11.5, 48.1]})
    assert r.status_code == 422
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/app/test_routes_jobs.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# ki_geodaten/app/routes/jobs.py
from __future__ import annotations
import json
import uuid
from fastapi import APIRouter, HTTPException, Request

from ki_geodaten.config import Settings
from ki_geodaten.jobs.store import insert_job
from ki_geodaten.models import CreateJobRequest, TilePreset
from ki_geodaten.pipeline.geo_utils import transform_bbox_wgs84_to_utm
from ki_geodaten.pipeline.dop_client import prepare_download_bbox

router = APIRouter()

def _within_bayern(
    bbox: list[float], bayern: tuple[float, float, float, float],
) -> bool:
    lon_min, lat_min, lon_max, lat_max = bayern
    return (
        bbox[0] >= lon_min and bbox[2] <= lon_max
        and bbox[1] >= lat_min and bbox[3] <= lat_max
    )

def _utm_area_km2(bbox_wgs84: list[float]) -> float:
    minx, miny, maxx, maxy = transform_bbox_wgs84_to_utm(*bbox_wgs84)
    return (maxx - minx) * (maxy - miny) / 1_000_000.0

@router.post("/jobs")
async def create_job(req: CreateJobRequest, request: Request):
    settings: Settings = request.app.state.settings
    bbox = req.bbox_wgs84

    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise HTTPException(422, detail="bbox_wgs84 must have minx<maxx and miny<maxy")

    if not _within_bayern(bbox, settings.BAYERN_BBOX_WGS84):
        raise HTTPException(
            422, detail=f"bbox must be within Bayern WGS84 bounds "
                        f"{settings.BAYERN_BBOX_WGS84}")

    area_km2 = _utm_area_km2(bbox)
    if area_km2 > settings.MAX_BBOX_AREA_KM2:
        raise HTTPException(
            422, detail=f"bbox area {area_km2:.3f} km² exceeds limit "
                        f"{settings.MAX_BBOX_AREA_KM2} km²")

    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(422, detail="prompt must be non-empty after strip()")
    if len(prompt) > settings.MAX_PROMPT_CHARS:
        raise HTTPException(
            422, detail=f"prompt exceeds {settings.MAX_PROMPT_CHARS} chars")

    counter = request.app.state.token_counter
    token_count = counter(prompt)
    if token_count > settings.MAX_ENCODER_CONTEXT_TOKENS:
        raise HTTPException(
            422,
            detail=f"prompt encodes to {token_count} tokens, exceeds "
                   f"{settings.MAX_ENCODER_CONTEXT_TOKENS}; use a shorter noun phrase",
        )

    # Grid-snap AOI to UTM (NOT expanded; expansion happens inside worker)
    utm_bounds = transform_bbox_wgs84_to_utm(*bbox)
    prepared = prepare_download_bbox(
        *utm_bounds, preset=req.tile_preset,
        origin_x=settings.WMS_GRID_ORIGIN_X, origin_y=settings.WMS_GRID_ORIGIN_Y,
    )

    job_id = str(uuid.uuid4())
    insert_job(
        request.app.state.db_path,
        job_id=job_id, prompt=prompt,
        bbox_wgs84=list(bbox),
        bbox_utm_snapped=list(prepared.aoi_bbox),
        tile_preset=req.tile_preset,
    )
    return {"id": job_id, "status": "PENDING"}
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/app/test_routes_jobs.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add ki_geodaten/app/routes/jobs.py tests/app/test_routes_jobs.py
git commit -m "feat(api): POST /jobs with Bayern fence, UTM area check, encoder token limit"
```

---

## Task 23: `GET /jobs`, `GET /jobs/{id}`, `GET /jobs/{id}/export.gpkg`

**Files:**
- Modify: `ki_geodaten/app/routes/jobs.py` (append)
- Test: `tests/app/test_routes_jobs.py` (append)

**See Spec §8** — list + detail expose `validation_revision`, `exported_revision`, derived `export_stale`. Download streams the `.gpkg` file.

- [ ] **Step 1: Append failing tests**

```python
# tests/app/test_routes_jobs.py (append)
from ki_geodaten.jobs.store import update_status
from ki_geodaten.models import JobStatus

def test_list_jobs_returns_revision_fields(client):
    c, app = client
    jid = c.post("/jobs", json={"prompt": "building", "bbox_wgs84": _munich_bbox()}).json()["id"]
    r = c.get("/jobs")
    assert r.status_code == 200
    body = r.json()
    entry = next(e for e in body if e["id"] == jid)
    assert entry["validation_revision"] == 0
    assert entry["exported_revision"] is None
    assert entry["export_stale"] is True
    # bbox_wgs84 must be a parsed array, not a JSON string
    assert entry["bbox_wgs84"] == _munich_bbox()

def test_get_job_detail(client):
    c, app = client
    jid = c.post("/jobs", json={"prompt": "building", "bbox_wgs84": _munich_bbox()}).json()["id"]
    r = c.get(f"/jobs/{jid}")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == jid
    assert body["status"] == "PENDING"
    assert "tile_total" in body

def test_get_job_404(client):
    c, _ = client
    r = c.get("/jobs/nonexistent")
    assert r.status_code == 404

def test_export_gpkg_download(client, tmp_path):
    c, app = client
    jid = c.post("/jobs", json={"prompt": "b", "bbox_wgs84": _munich_bbox()}).json()["id"]
    gpkg = app.state.results_dir / f"{jid}.gpkg"
    gpkg.write_bytes(b"PRAGMA")
    update_status(app.state.db_path, jid, JobStatus.EXPORTED,
                  gpkg_path=str(gpkg), set_finished=True)
    r = c.get(f"/jobs/{jid}/export.gpkg")
    assert r.status_code == 200
    assert r.content == b"PRAGMA"

def test_export_gpkg_missing_file_404(client):
    c, app = client
    jid = c.post("/jobs", json={"prompt": "b", "bbox_wgs84": _munich_bbox()}).json()["id"]
    r = c.get(f"/jobs/{jid}/export.gpkg")
    assert r.status_code == 404
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/app/test_routes_jobs.py -v`
Expected: new tests FAIL.

- [ ] **Step 3: Append implementation**

```python
# ki_geodaten/app/routes/jobs.py (append)
from fastapi.responses import FileResponse
from pathlib import Path
from ki_geodaten.jobs.store import list_jobs, get_job

_JOB_VIEW_FIELDS = (
    "id", "prompt", "tile_preset", "status",
    "error_reason", "error_message",
    "tile_completed", "tile_total", "tile_failed",
    "validation_revision", "exported_revision",
    "created_at", "started_at", "finished_at",
    "bbox_wgs84",
)

def _job_view(job: dict) -> dict:
    view = {k: job.get(k) for k in _JOB_VIEW_FIELDS}
    # bbox_wgs84 is stored as a JSON string in SQLite; UI expects an array.
    if view.get("bbox_wgs84"):
        view["bbox_wgs84"] = json.loads(view["bbox_wgs84"])
    exported = job.get("exported_revision")
    validation = job.get("validation_revision") or 0
    view["export_stale"] = exported is None or exported < validation
    return view

@router.get("/jobs")
async def list_jobs_endpoint(request: Request):
    rows = list_jobs(request.app.state.db_path)
    return [_job_view(j) for j in rows]

@router.get("/jobs/{job_id}")
async def get_job_endpoint(job_id: str, request: Request):
    job = get_job(request.app.state.db_path, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return _job_view(job)

@router.get("/jobs/{job_id}/export.gpkg")
async def download_gpkg(job_id: str, request: Request):
    job = get_job(request.app.state.db_path, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    path = job.get("gpkg_path")
    if not path or not Path(path).exists():
        raise HTTPException(404, "gpkg not yet generated")
    return FileResponse(
        path, media_type="application/geopackage+sqlite3",
        filename=f"{job_id}.gpkg",
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/app/test_routes_jobs.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add ki_geodaten/app/routes/jobs.py tests/app/test_routes_jobs.py
git commit -m "feat(api): GET /jobs, /jobs/{id}, /jobs/{id}/export.gpkg"
```

---

## Task 24: `POST /jobs/{id}/polygons/validate_bulk`

**Files:**
- Modify: `ki_geodaten/app/routes/jobs.py` (append)
- Test: `tests/app/test_routes_jobs.py` (append)

**See Spec §8** — status gate (`READY_FOR_REVIEW` or `EXPORTED` only), `executemany`, increments `validation_revision`.

**Critical:** declare this endpoint as `def`, not `async def`. It performs synchronous SQLite writes and can wait on WAL writer locks; FastAPI should run it in the threadpool so the event loop remains responsive to status polling and GeoJSON requests.

- [ ] **Step 1: Append failing tests**

```python
# tests/app/test_routes_jobs.py (append)
from ki_geodaten.jobs.store import insert_polygons

def _make_reviewable_job(app, c, n_polys=3, status=JobStatus.READY_FOR_REVIEW) -> str:
    jid = c.post("/jobs", json={"prompt": "b", "bbox_wgs84": _munich_bbox()}).json()["id"]
    insert_polygons(app.state.db_path, jid, [
        {"geometry_wkb": b"x", "score": 0.9, "source_tile_row": 0, "source_tile_col": i}
        for i in range(n_polys)
    ])
    update_status(app.state.db_path, jid, status, set_finished=True)
    return jid

def test_validate_bulk_happy_path(client):
    c, app = client
    jid = _make_reviewable_job(app, c, n_polys=3)
    r = c.post(
        f"/jobs/{jid}/polygons/validate_bulk",
        json={"updates": [{"pid": 1, "validation": "REJECTED"},
                          {"pid": 2, "validation": "REJECTED"}]},
    )
    assert r.status_code == 200
    assert r.json()["updated"] == 2
    assert c.get(f"/jobs/{jid}").json()["validation_revision"] == 1

def test_validate_bulk_rejects_pending_job(client):
    c, app = client
    jid = c.post("/jobs", json={"prompt": "b", "bbox_wgs84": _munich_bbox()}).json()["id"]
    r = c.post(
        f"/jobs/{jid}/polygons/validate_bulk",
        json={"updates": [{"pid": 1, "validation": "REJECTED"}]},
    )
    assert r.status_code == 409

def test_validate_bulk_allowed_on_exported_job(client):
    c, app = client
    jid = _make_reviewable_job(app, c, status=JobStatus.EXPORTED)
    r = c.post(
        f"/jobs/{jid}/polygons/validate_bulk",
        json={"updates": [{"pid": 1, "validation": "REJECTED"}]},
    )
    assert r.status_code == 200

def test_validate_bulk_job_not_found(client):
    c, _ = client
    r = c.post(
        "/jobs/nope/polygons/validate_bulk",
        json={"updates": [{"pid": 1, "validation": "REJECTED"}]},
    )
    assert r.status_code == 404
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/app/test_routes_jobs.py -v`
Expected: new tests FAIL.

- [ ] **Step 3: Append implementation**

```python
# ki_geodaten/app/routes/jobs.py (append)
from ki_geodaten.models import ValidateBulkRequest
from ki_geodaten.jobs.store import validate_bulk

_REVIEWABLE = {"READY_FOR_REVIEW", "EXPORTED"}

@router.post("/jobs/{job_id}/polygons/validate_bulk")
def validate_bulk_endpoint(
    job_id: str, req: ValidateBulkRequest, request: Request,
):
    job = get_job(request.app.state.db_path, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job["status"] not in _REVIEWABLE:
        raise HTTPException(
            409, f"job status {job['status']!s} does not allow validation",
        )
    updates = [u.model_dump() for u in req.updates]
    updated = validate_bulk(request.app.state.db_path, job_id, updates)
    # Invalidate polygon GeoJSON cache for this job
    cache = request.app.state.geojson_cache
    for key in list(cache.keys()):
        if key[0] == job_id and key[2] == "polygons":
            cache.pop(key, None)
    return {"ok": True, "updated": updated}
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/app/test_routes_jobs.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add ki_geodaten/app/routes/jobs.py tests/app/test_routes_jobs.py
git commit -m "feat(api): validate_bulk with status gate and revision bump"
```

---

## Task 25: `POST /jobs/{id}/export` — Serialized Lock, exported_revision, Chunk Cleanup

**Files:**
- Modify: `ki_geodaten/app/routes/jobs.py` (append)
- Test: `tests/app/test_routes_jobs.py` (append)

**See Spec §8** — module-wide `threading.Lock` around `export_two_layer_gpkg` (GDAL/Fiona are not thread-safe). Re-export allowed from `EXPORTED`. On success: `status='EXPORTED'`, `exported_revision = validation_revision`, `finished_at` is refreshed to the export time for retention semantics, and `data/dop/{job_id}` is removed.

**Critical:** this endpoint is declared as `def` (not `async def`). FastAPI routes sync functions to its threadpool — essential, because `export_two_layer_gpkg` holds the GIL-releasing GDAL/Fiona calls for multiple seconds. If it were `async def`, the blocking work would freeze the event loop and block ALL other requests (status polls, GeoJSON fetches) until export finishes. The `threading.Lock` then correctly serializes concurrent threadpool workers.

- [ ] **Step 1: Append failing tests**

```python
# tests/app/test_routes_jobs.py (append)
import threading
from pathlib import Path
import rasterio
from shapely.geometry import Polygon
from shapely.wkb import dumps as wkb_dumps

def _insert_real_polygon(app, jid):
    from ki_geodaten.jobs.store import insert_polygons
    poly = Polygon([(691100, 5335100), (691200, 5335100),
                    (691200, 5335200), (691100, 5335200)])
    insert_polygons(app.state.db_path, jid, [
        {"geometry_wkb": wkb_dumps(poly), "score": 0.9,
         "source_tile_row": 0, "source_tile_col": 0}
    ])
    update_status(app.state.db_path, jid, JobStatus.READY_FOR_REVIEW,
                  set_finished=True)

def _post_real_bbox(c):
    # UTM 691000..692000 / 5335000..5336000 is inside Munich-range wgs84 bbox
    return c.post("/jobs", json={
        "prompt": "building", "bbox_wgs84": [11.55, 48.13, 11.56, 48.14],
    }).json()["id"]

def test_export_happy_path_sets_revision_and_file(client, tmp_path):
    c, app = client
    jid = _post_real_bbox(c)
    _insert_real_polygon(app, jid)
    r = c.post(f"/jobs/{jid}/export")
    assert r.status_code == 200
    gpkg_path = Path(r.json()["gpkg_path"])
    assert gpkg_path.exists()
    detail = c.get(f"/jobs/{jid}").json()
    assert detail["status"] == "EXPORTED"
    assert detail["exported_revision"] == detail["validation_revision"]
    assert detail["export_stale"] is False

def test_export_rejects_pending_job(client):
    c, _ = client
    jid = c.post("/jobs", json={"prompt": "b", "bbox_wgs84": _munich_bbox()}).json()["id"]
    r = c.post(f"/jobs/{jid}/export")
    assert r.status_code == 409

def test_re_export_from_exported_status_allowed(client, tmp_path):
    c, app = client
    jid = _post_real_bbox(c)
    _insert_real_polygon(app, jid)
    c.post(f"/jobs/{jid}/export")
    r2 = c.post(f"/jobs/{jid}/export")
    assert r2.status_code == 200

def test_export_cleans_dop_directory(client, tmp_path):
    c, app = client
    jid = _post_real_bbox(c)
    dop_dir = app.state.data_root / "dop" / jid
    dop_dir.mkdir(parents=True)
    (dop_dir / "chunk_0_0.tif").write_bytes(b"junk")
    _insert_real_polygon(app, jid)
    c.post(f"/jobs/{jid}/export")
    assert not dop_dir.exists()

def test_export_lock_serializes_parallel_calls(client):
    """Two concurrent POSTs must both succeed and never crash."""
    c, app = client
    jid = _post_real_bbox(c)
    _insert_real_polygon(app, jid)
    results = []
    def go():
        results.append(c.post(f"/jobs/{jid}/export").status_code)
    threads = [threading.Thread(target=go) for _ in range(2)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert sorted(results) == [200, 200]
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/app/test_routes_jobs.py -v`
Expected: new tests FAIL.

- [ ] **Step 3: Append implementation**

```python
# ki_geodaten/app/routes/jobs.py (append)
import shutil
from shapely.geometry import box as shp_box
from shapely.wkb import loads as wkb_loads
import geopandas as gpd
from ki_geodaten.jobs.store import (
    get_polygons_for_job, get_nodata_for_job, update_status,
)
from ki_geodaten.pipeline.exporter import export_two_layer_gpkg
from ki_geodaten.models import JobStatus, ErrorReason

_EXPORTABLE = {"READY_FOR_REVIEW", "EXPORTED"}

def _detected_gdf(rows: list[dict]) -> gpd.GeoDataFrame:
    accepted = [r for r in rows if r["validation"] == "ACCEPTED"]
    if not accepted:
        return gpd.GeoDataFrame(
            {"score": [], "source_tile_row": [], "source_tile_col": []},
            geometry=[], crs="EPSG:25832",
        )
    geoms = [wkb_loads(r["geometry_wkb"]) for r in accepted]
    return gpd.GeoDataFrame(
        {
            "score": [r["score"] for r in accepted],
            "source_tile_row": [r["source_tile_row"] for r in accepted],
            "source_tile_col": [r["source_tile_col"] for r in accepted],
        },
        geometry=geoms, crs="EPSG:25832",
    )

def _nodata_gdf(rows: list[dict]) -> gpd.GeoDataFrame:
    if not rows:
        return gpd.GeoDataFrame({"reason": []}, geometry=[], crs="EPSG:25832")
    geoms = [wkb_loads(r["geometry_wkb"]) for r in rows]
    return gpd.GeoDataFrame(
        {"reason": [r["reason"] for r in rows]},
        geometry=geoms, crs="EPSG:25832",
    )

# NOTE: `def` (not `async def`) — FastAPI runs this in its threadpool, so the
# blocking GDAL/Fiona write below doesn't freeze the event loop. The
# threading.Lock then correctly serializes concurrent workers.
@router.post("/jobs/{job_id}/export")
def export_job(job_id: str, request: Request):
    db = request.app.state.db_path
    job = get_job(db, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job["status"] not in _EXPORTABLE:
        raise HTTPException(409, f"job status {job['status']!s} not exportable")

    aoi = shp_box(*json.loads(job["bbox_utm_snapped"]))
    out_path = request.app.state.results_dir / f"{job_id}.gpkg"

    lock = request.app.state.export_lock
    with lock:
        polys = get_polygons_for_job(db, job_id)
        nods = get_nodata_for_job(db, job_id)
        det_gdf = _detected_gdf(polys)
        nod_gdf = _nodata_gdf(nods)
        try:
            export_two_layer_gpkg(det_gdf, nod_gdf, aoi, out_path)
        except Exception as exc:
            update_status(
                db, job_id, JobStatus.FAILED,
                error_reason=str(ErrorReason.EXPORT_ERROR),
                error_message=str(exc), set_finished=True,
            )
            raise HTTPException(500, f"export failed: {exc}") from exc

        update_status(
            db, job_id, JobStatus.EXPORTED,
            gpkg_path=str(out_path),
            exported_revision=int(job["validation_revision"]),
            set_finished=True,
        )

    # Disk-Hygiene after successful export (Spec §7)
    shutil.rmtree(request.app.state.data_root / "dop" / job_id, ignore_errors=True)
    return {"gpkg_path": str(out_path)}
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/app/test_routes_jobs.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add ki_geodaten/app/routes/jobs.py tests/app/test_routes_jobs.py
git commit -m "feat(api): export endpoint with lock, exported_revision, chunk cleanup"
```

---

## Task 26: `GET /jobs/{id}/polygons` & `GET /jobs/{id}/nodata` with ProcessPoolExecutor + Cache

**Files:**
- Create: `ki_geodaten/app/routes/geojson.py`
- Test: `tests/app/test_routes_geojson.py`

**See Spec §8** — runs WKB→GeoJSON and `json.dumps(...)` in `app.state.geojson_executor`, key cache by `(job_id, validation_revision, target)`. The route returns a raw `Response` with the already-serialized JSON string so the FastAPI event loop does not serialize multi-MB dictionaries.

Because `ProcessPoolExecutor` submits picklable callables, the function invoked in the pool must be a module-level function that takes plain Python data (already in `ki_geodaten/app/serialization.py`).

- [ ] **Step 1: Write failing test**

```python
# tests/app/test_routes_geojson.py
import pytest
from fastapi.testclient import TestClient
from shapely.geometry import Polygon
from shapely.wkb import dumps as wkb_dumps

from ki_geodaten.app.main import create_app
from ki_geodaten.jobs.store import insert_polygons, insert_nodata_region, update_status
from ki_geodaten.models import JobStatus

@pytest.fixture
def client(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    monkeypatch.chdir(tmp_path)
    # ThreadPoolExecutor avoids subprocess spawn; functools.partial in the
    # route still works because it's picklable, but for threads it's just a
    # callable — either way, the same route code is exercised.
    app = create_app(
        executor_factory=lambda: ThreadPoolExecutor(max_workers=2),
        token_counter=lambda s: len(s.split()),
    )
    with TestClient(app) as c:
        yield c, app

def _setup(app, c):
    jid = c.post("/jobs", json={
        "prompt": "b", "bbox_wgs84": [11.55, 48.13, 11.56, 48.14],
    }).json()["id"]
    poly = Polygon([(691100, 5335100), (691200, 5335100),
                    (691200, 5335200), (691100, 5335200)])
    insert_polygons(app.state.db_path, jid, [
        {"geometry_wkb": wkb_dumps(poly), "score": 0.9,
         "source_tile_row": 0, "source_tile_col": 0}
    ])
    update_status(app.state.db_path, jid, JobStatus.READY_FOR_REVIEW,
                  set_finished=True)
    return jid, poly

def test_get_polygons_returns_feature_collection(client):
    c, app = client
    jid, _ = _setup(app, c)
    r = c.get(f"/jobs/{jid}/polygons")
    assert r.status_code == 200
    fc = r.json()
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 1
    props = fc["features"][0]["properties"]
    assert {"id", "score", "validation"}.issubset(props.keys())

def test_get_nodata_returns_feature_collection(client):
    c, app = client
    jid, poly = _setup(app, c)
    insert_nodata_region(
        app.state.db_path, jid,
        geometry_wkb=wkb_dumps(poly), tile_row=0, tile_col=0, reason="OOM",
    )
    r = c.get(f"/jobs/{jid}/nodata")
    fc = r.json()
    assert fc["features"][0]["properties"] == {"reason": "OOM"}

def test_polygons_cache_invalidated_on_validate_bulk(client):
    c, app = client
    jid, _ = _setup(app, c)
    fc1 = c.get(f"/jobs/{jid}/polygons").json()
    assert fc1["features"][0]["properties"]["validation"] == "ACCEPTED"
    c.post(
        f"/jobs/{jid}/polygons/validate_bulk",
        json={"updates": [{"pid": 1, "validation": "REJECTED"}]},
    )
    fc2 = c.get(f"/jobs/{jid}/polygons").json()
    assert fc2["features"][0]["properties"]["validation"] == "REJECTED"

def test_get_polygons_404(client):
    c, _ = client
    r = c.get("/jobs/nope/polygons")
    assert r.status_code == 404
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/app/test_routes_geojson.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# ki_geodaten/app/routes/geojson.py
from __future__ import annotations
import asyncio
import functools
import json
from fastapi import APIRouter, HTTPException, Request, Response

from ki_geodaten.app.serialization import (
    build_polygons_geojson, build_nodata_geojson,
)
from ki_geodaten.jobs.store import (
    get_job, get_polygons_for_job, get_nodata_for_job,
)

router = APIRouter()

async def _run_in_executor(app, fn, /, *args, **kwargs):
    """Submit to the dedicated GeoJSON executor.

    We use functools.partial, NOT a lambda: ProcessPoolExecutor pickles the
    callable when dispatching to a worker process; lambdas are not picklable
    and would raise PicklingError at runtime. functools.partial of a
    module-level function with picklable args IS picklable.
    """
    loop = asyncio.get_running_loop()
    bound = functools.partial(fn, *args, **kwargs)
    return await loop.run_in_executor(app.state.geojson_executor, bound)

def _aoi(job: dict) -> tuple[float, float, float, float]:
    return tuple(json.loads(job["bbox_utm_snapped"]))

@router.get("/jobs/{job_id}/polygons")
async def get_polygons(job_id: str, request: Request):
    job = get_job(request.app.state.db_path, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    revision = int(job["validation_revision"] or 0)
    key = (job_id, revision, "polygons")
    cache = request.app.state.geojson_cache
    if key in cache:
        return Response(content=cache[key], media_type="application/json")
    rows = get_polygons_for_job(request.app.state.db_path, job_id)
    payload = await _run_in_executor(
        request.app, build_polygons_geojson,
        rows, aoi_utm=_aoi(job),
    )
    cache[key] = payload
    return Response(content=payload, media_type="application/json")

@router.get("/jobs/{job_id}/nodata")
async def get_nodata(job_id: str, request: Request):
    job = get_job(request.app.state.db_path, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    revision = int(job["validation_revision"] or 0)
    key = (job_id, revision, "nodata")
    cache = request.app.state.geojson_cache
    if key in cache:
        return Response(content=cache[key], media_type="application/json")
    rows = get_nodata_for_job(request.app.state.db_path, job_id)
    payload = await _run_in_executor(
        request.app, build_nodata_geojson,
        rows, aoi_utm=_aoi(job),
    )
    cache[key] = payload
    return Response(content=payload, media_type="application/json")
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/app/test_routes_geojson.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add ki_geodaten/app/routes/geojson.py tests/app/test_routes_geojson.py
git commit -m "feat(api): /polygons and /nodata via ProcessPoolExecutor with revision cache"
```

---

## Task 27: Leaflet HTML Template + Static Assets

**Files:**
- Modify: `ki_geodaten/app/templates/index.html`
- Create: `ki_geodaten/app/static/app.js`
- Create: `ki_geodaten/app/static/app.css`

**See Spec §9** — single page: map + prompt panel + job history. Leaflet forced `renderer: L.canvas()`; preset dropdown with tooltip. NoData overlay is hatched.

Because this is client code driving a user-facing UI, there is no unit test layer; Task 30 (end-to-end) verifies the round-trip.

- [ ] **Step 1: Rewrite `index.html`**

```html
<!-- ki_geodaten/app/templates/index.html -->
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Text-to-Polygon Pipeline</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <link rel="stylesheet" href="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.css">
  <link rel="stylesheet" href="/static/app.css">
</head>
<body>
  <div id="sidebar">
    <h1>ki-geodaten</h1>
    <section id="prompt-panel">
      <label>Prompt
        <input id="prompt" type="text" placeholder="e.g. building">
      </label>
      <label>Tile preset
        <select id="preset" title="small=64m | medium=128m | large=192m max object diameter">
          <option value="small">small (solar, car, tree)</option>
          <option value="medium" selected>medium (building, school)</option>
          <option value="large">large (hall, stadium — slow)</option>
        </select>
      </label>
      <button id="submit" disabled>Draw BBox &amp; Start</button>
      <div id="status"></div>
    </section>
    <section id="job-history">
      <h2>Jobs</h2>
      <ul id="job-list"></ul>
    </section>
    <section id="review-panel" hidden>
      <h2>Review</h2>
      <p id="review-meta"></p>
      <button id="export-btn">Export GPKG</button>
      <a id="download-link" hidden>Download</a>
    </section>
  </div>
  <div id="map"></div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.js"></script>
  <script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create `app.css`**

```css
/* ki_geodaten/app/static/app.css
   NOTE: Polygon/NoData styling happens via Leaflet style options in app.js
   (color / fillColor / dashArray / fillOpacity). The canvas renderer ignores
   per-feature CSS classes, so we don't define .polygon-* classes here. */
html, body { margin: 0; height: 100%; font-family: system-ui, sans-serif; }
body { display: flex; }
#sidebar { width: 320px; padding: 12px; overflow-y: auto; border-right: 1px solid #ddd; }
#map { flex: 1; height: 100vh; }
label { display: block; margin: 8px 0; }
input, select, button { width: 100%; padding: 6px; box-sizing: border-box; }
#job-list { list-style: none; padding: 0; }
#job-list li { padding: 6px; border-bottom: 1px solid #eee; cursor: pointer; }
.badge { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 11px; color: #fff; }
.badge-PENDING, .badge-DOWNLOADING, .badge-INFERRING { background: #888; }
.badge-READY_FOR_REVIEW { background: #3a7; }
.badge-EXPORTED { background: #36c; }
.badge-FAILED { background: #c33; }
```

- [ ] **Step 3: Create `app.js`**

```javascript
// ki_geodaten/app/static/app.js
(() => {
  // Leaflet style constants — applied via native style() options, because
  // Canvas-Renderer ignores CSS classes per feature (Spec §9.1).
  const STYLE_ACCEPTED = { color: '#36c', fillColor: '#36c', fillOpacity: 0.2, weight: 1 };
  const STYLE_REJECTED = { color: '#c33', fillColor: '#c33', fillOpacity: 0.15, weight: 1, dashArray: '4,2' };
  const STYLE_NODATA   = { color: '#c93', fillColor: '#c93', fillOpacity: 0.25, weight: 1, dashArray: '6,3' };

  const map = L.map('map', { renderer: L.canvas() }).setView([48.13, 11.56], 13);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap',
  }).addTo(map);

  const drawnItems = new L.FeatureGroup().addTo(map);
  const drawControl = new L.Control.Draw({
    draw: {
      rectangle: { shapeOptions: { color: '#36c' } },
      polygon: false, polyline: false, circle: false, marker: false, circlemarker: false,
    },
    edit: { featureGroup: drawnItems, edit: false, remove: true },
  });
  map.addControl(drawControl);

  const submitBtn = document.getElementById('submit');
  const promptInput = document.getElementById('prompt');
  const presetSelect = document.getElementById('preset');
  const statusDiv = document.getElementById('status');
  const jobList = document.getElementById('job-list');
  const reviewPanel = document.getElementById('review-panel');
  const reviewMeta = document.getElementById('review-meta');
  const exportBtn = document.getElementById('export-btn');
  const downloadLink = document.getElementById('download-link');

  let currentBBox = null;
  let currentJobId = null;
  let currentLayer = null;
  let nodataLayer = null;
  let pollTimer = null;

  // ── Validation debouncing state (declared before openJob because openJob uses it) ──
  const pendingUpdates = new Map();
  let flushTimer = null;
  let isFlushing = false;
  const MAX_CLIENT_BUFFER_UPDATES = 100;

  const storageKey = id => `job:${id}:pending-validations`;
  const snapshotUpdates = () =>
    [...pendingUpdates].map(([pid, validation]) => ({pid, validation}));
  function persistPending(jobId) {
    if (!jobId) return;
    sessionStorage.setItem(storageKey(jobId), JSON.stringify(snapshotUpdates()));
  }
  const clearPendingFromStorage = id => sessionStorage.removeItem(storageKey(id));
  function hydratePending(id) {
    const raw = sessionStorage.getItem(storageKey(id));
    if (!raw) return;
    try {
      for (const {pid, validation} of JSON.parse(raw)) pendingUpdates.set(pid, validation);
    } catch (e) {}
  }

  map.on(L.Draw.Event.CREATED, (e) => {
    drawnItems.clearLayers();
    drawnItems.addLayer(e.layer);
    const b = e.layer.getBounds();
    currentBBox = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()];
    submitBtn.disabled = false;
  });

  map.on(L.Draw.Event.DELETED, () => { currentBBox = null; submitBtn.disabled = true; });

  async function submit() {
    const body = {
      prompt: promptInput.value.trim(),
      bbox_wgs84: currentBBox,
      tile_preset: presetSelect.value,
    };
    const r = await fetch('/jobs', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      statusDiv.textContent = 'Error: ' + (err.detail || r.statusText);
      return;
    }
    const { id } = await r.json();
    await refreshJobList();
    openJob(id);
  }
  submitBtn.onclick = submit;

  async function refreshJobList() {
    const r = await fetch('/jobs');
    const jobs = await r.json();
    jobList.innerHTML = '';
    for (const j of jobs) {
      const li = document.createElement('li');
      const processed = (j.tile_completed || 0) + (j.tile_failed || 0);
      li.innerHTML = `<span class="badge badge-${j.status}">${j.status}</span> ${j.prompt}
                     <small>${processed}/${j.tile_total||'?'}</small>`;
      li.onclick = () => openJob(j.id);
      jobList.appendChild(li);
    }
  }

  function stopPolling() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  }

  async function openJob(id) {
    stopPolling();
    // Spec §9.2: flush to the OLD job BEFORE we switch currentJobId, otherwise
    // leftover pending updates would be POSTed to the wrong job.
    if (currentJobId && currentJobId !== id) {
      await flushValidations();
    }
    pendingUpdates.clear();
    currentJobId = id;
    // Recover any batch that was stranded in sessionStorage (e.g. previous tab close).
    hydratePending(id);
    if (pendingUpdates.size) {
      void flushValidations();
    }

    const r = await fetch(`/jobs/${id}`);
    if (!r.ok) { statusDiv.textContent = 'Job not found'; return; }
    const job = await r.json();
    renderStatus(job);

    if (['PENDING', 'DOWNLOADING', 'INFERRING'].includes(job.status)) {
      pollTimer = setInterval(async () => {
        const rr = await fetch(`/jobs/${id}`);
        const jj = await rr.json();
        renderStatus(jj);
        if (!['PENDING', 'DOWNLOADING', 'INFERRING'].includes(jj.status)) {
          stopPolling();
          if (jj.status !== 'FAILED') loadPolygons(id);
        }
        refreshJobList();
      }, 3000);
    } else if (['READY_FOR_REVIEW', 'EXPORTED'].includes(job.status)) {
      loadPolygons(id);
    }
  }

  function renderStatus(job) {
    const processed = (job.tile_completed || 0) + (job.tile_failed || 0);
    statusDiv.innerHTML =
      `<b>${job.status}</b> &mdash; ${processed}/${job.tile_total||'?'}` +
      (job.error_reason ? ` <em>(${job.error_reason})</em>` : '');
    reviewPanel.hidden = !['READY_FOR_REVIEW', 'EXPORTED'].includes(job.status);
    if (!reviewPanel.hidden) {
      const stale = job.export_stale ? ' <strong>(export stale — re-export)</strong>' : '';
      reviewMeta.innerHTML =
        `validation rev ${job.validation_revision}, exported rev ${job.exported_revision ?? '—'}${stale}`;
      downloadLink.hidden = !['EXPORTED'].includes(job.status) || job.export_stale;
      downloadLink.href = `/jobs/${job.id}/export.gpkg`;
      downloadLink.textContent = 'Download GPKG';
    }
  }

  function featureStyle(f) {
    return f.properties.validation === 'ACCEPTED' ? STYLE_ACCEPTED : STYLE_REJECTED;
  }

  async function loadPolygons(id) {
    if (currentLayer) { map.removeLayer(currentLayer); currentLayer = null; }
    if (nodataLayer) { map.removeLayer(nodataLayer); nodataLayer = null; }
    const [polys, nods] = await Promise.all([
      fetch(`/jobs/${id}/polygons`).then(r => r.json()),
      fetch(`/jobs/${id}/nodata`).then(r => r.json()),
    ]);
    currentLayer = L.geoJSON(polys, {
      style: featureStyle,
      onEachFeature: (f, layer) => {
        layer.on('click', () => toggleValidation(f, layer));
      },
    }).addTo(map);
    nodataLayer = L.geoJSON(nods, {
      style: () => STYLE_NODATA,
    }).addTo(map);
    if (polys.features.length) {
      try { map.fitBounds(currentLayer.getBounds()); } catch (e) {}
    }
  }

  function toggleValidation(feature, layer) {
    feature.properties.validation =
      feature.properties.validation === 'ACCEPTED' ? 'REJECTED' : 'ACCEPTED';
    layer.setStyle(featureStyle(feature));
    queueValidation(feature.properties.id, feature.properties.validation);
  }

  function queueValidation(pid, validation) {
    if (!currentJobId) return;
    pendingUpdates.set(pid, validation);
    persistPending(currentJobId);
    if (isFlushing) return;
    if (pendingUpdates.size >= MAX_CLIENT_BUFFER_UPDATES) {
      void flushValidations();
      return;
    }
    if (flushTimer) clearTimeout(flushTimer);
    flushTimer = setTimeout(() => void flushValidations(), 3000);
  }

  async function flushValidations() {
    if (isFlushing || !pendingUpdates.size || !currentJobId) return;
    isFlushing = true;
    const jobId = currentJobId;
    // Snapshot BUT keep pending in memory until the POST succeeds — otherwise
    // a network error silently loses the user's validations (Spec §9.2).
    const snapshot = new Map(pendingUpdates);
    const updates = [...snapshot].map(([pid, validation]) => ({pid, validation}));
    let failed = false;
    try {
      const r = await fetch(`/jobs/${jobId}/polygons/validate_bulk`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({updates}),
      });
      if (!r.ok) throw new Error(`server ${r.status}`);
      // Remove only the entries we just sent; new clicks during the POST stay.
      for (const [pid, sentVal] of snapshot) {
        if (pendingUpdates.get(pid) === sentVal) {
          pendingUpdates.delete(pid);
        }
      }
      if (pendingUpdates.size === 0) {
        clearPendingFromStorage(jobId);
      } else {
        persistPending(jobId);
      }
    } catch (e) {
      // Network / server error: leave pending intact for next retry.
      // Re-persist so the current snapshot is durable across refresh.
      failed = true;
      persistPending(jobId);
    } finally {
      isFlushing = false;
      if (pendingUpdates.size && currentJobId === jobId) {
        if (flushTimer) clearTimeout(flushTimer);
        flushTimer = setTimeout(
          () => void flushValidations(),
          failed ? 3000 : 0,
        );
      }
    }
  }

  window.addEventListener('pagehide', () => {
    if (!pendingUpdates.size || !currentJobId) return;
    persistPending(currentJobId);   // Durable recovery path.
    fetch(`/jobs/${currentJobId}/polygons/validate_bulk`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        updates: snapshotUpdates().slice(0, MAX_CLIENT_BUFFER_UPDATES),
      }),
      keepalive: true,
    }).catch(() => {});
  });

  exportBtn.onclick = async () => {
    if (!currentJobId) return;
    await flushValidations();
    const r = await fetch(`/jobs/${currentJobId}/export`, {method: 'POST'});
    if (r.ok) openJob(currentJobId);
    else statusDiv.textContent = 'Export failed';
  };

  refreshJobList();
})();
```

- [ ] **Step 4: Smoke-test manually**

Run: `uvicorn ki_geodaten.app.main:app --reload --reload-dir ki_geodaten --reload-exclude 'data/*' --reload-exclude '*.db*'`
Open `http://127.0.0.1:8000/` in a browser. Verify: map loads, rectangle draw enables submit, invalid region (Berlin) yields an error banner.

- [ ] **Step 5: Commit**

```bash
git add ki_geodaten/app/templates/index.html ki_geodaten/app/static/app.js ki_geodaten/app/static/app.css
git commit -m "feat(ui): Leaflet map, draw, preset dropdown, validation debounce"
```

---

## Task 28: Run Scripts (`scripts/run-server.sh`, `scripts/run-worker.sh`)

**Files:**
- Create: `scripts/run-server.sh`
- Create: `scripts/run-worker.sh`
- Test: manual (no unit tests — bash wrappers)

**See Spec §12** — uvicorn needs `--reload-exclude` for SQLite WAL files. Worker supervisor loops until manual kill.

- [ ] **Step 1: Write `scripts/run-server.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
exec uvicorn ki_geodaten.app.main:app \
    --host 127.0.0.1 --port 8000 \
    --reload \
    --reload-dir ki_geodaten \
    --reload-exclude 'data/*' \
    --reload-exclude '*.db*'
```

Make executable: `chmod +x scripts/run-server.sh` (Windows users run `python -m uvicorn …` directly; note in README).

- [ ] **Step 2: Write `scripts/run-worker.sh`**

```bash
#!/usr/bin/env bash
# Supervisor loop — on clean exit (after MAX_JOBS_PER_WORKER) or crash, respawn.
# Spec §10: flushes VRAM fragmentation by killing and re-loading the python process.
set -u
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Make a single Ctrl-C stop the supervisor cleanly instead of respawning.
trap 'echo "[supervisor] stopping"; exit 0' INT TERM

while true; do
    python -m ki_geodaten.worker.loop
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "[supervisor] worker exited rc=$rc, restarting in 2s"
    else
        echo "[supervisor] worker exited cleanly, restarting in 2s"
    fi
    sleep 2
done
```

`chmod +x scripts/run-worker.sh`.

- [ ] **Step 3: Verify shell syntax**

Run: `bash -n scripts/run-server.sh && bash -n scripts/run-worker.sh`
Expected: no output (syntax OK).

- [ ] **Step 4: Commit**

```bash
git add scripts/
git commit -m "chore: add run-server.sh and run-worker.sh supervisor scripts"
```

---

## Task 29: End-to-End Integration Test

**Files:**
- Create: `tests/test_end_to_end.py`

**See Spec §11 End-to-End** — mini GeoTIFF fixture, mock WMS download, stub segmenter returning fixed mask; verify the pipeline runs POST→WORKER→EXPORT and produces a valid two-layer GPKG.

This test exercises the full worker + API together, using monkeypatches to replace the real segmenter and the `download_dop20` WMS download.

- [ ] **Step 1: Write test**

```python
# tests/test_end_to_end.py
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

# End-to-end needs osgeo.gdal.BuildVRT; skip cleanly when not installed.
pytest.importorskip("osgeo")
# fiona is optional for the assertion about layer listing.
fiona = pytest.importorskip("fiona")

import rasterio
from rasterio.transform import from_bounds
from fastapi.testclient import TestClient

from ki_geodaten.app.main import create_app
from ki_geodaten.jobs.store import get_job
from ki_geodaten.worker.loop import run_forever
from ki_geodaten.pipeline.segmenter import MaskResult

def _make_vrt(path: Path, bbox):
    tif = path.parent / "chunk_0_0.tif"
    w = round((bbox[2] - bbox[0]) / 0.2)
    h = round((bbox[3] - bbox[1]) / 0.2)
    with rasterio.open(
        tif, "w", driver="GTiff", width=w, height=h, count=3,
        dtype="uint8", crs="EPSG:25832", transform=from_bounds(*bbox, w, h),
        nodata=0,
    ) as dst:
        arr = np.full((3, h, w), 128, dtype="uint8")
        dst.write(arr)
    from osgeo import gdal
    gdal.BuildVRT(str(path), [str(tif)])

class _StubSegmenter:
    def predict(self, tile, prompt):
        mask = np.zeros((tile.size, tile.size), dtype=bool)
        mask[500:524, 500:524] = True  # 24×24 px = ~4.8×4.8 m
        return [MaskResult(mask=mask, score=0.95, box_pixel=(500, 500, 524, 524))]
    def encoder_token_count(self, s):
        return len(s.split())

def test_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # Monkeypatch download_dop20 to synthesise a VRT on disk instead of HTTP
    def fake_download(bbox_utm, *, out_dir, **kwargs):
        out_dir.mkdir(parents=True, exist_ok=True)
        vrt = out_dir / "out.vrt"
        _make_vrt(vrt, bbox_utm)
        return vrt
    monkeypatch.setattr(
        "ki_geodaten.worker.orchestrator.download_dop20", fake_download,
    )

    from concurrent.futures import ThreadPoolExecutor
    app = create_app(
        executor_factory=lambda: ThreadPoolExecutor(max_workers=2),
        token_counter=lambda s: len(s.split()),
    )
    with TestClient(app) as client:
        r = client.post(
            "/jobs",
            json={"prompt": "building", "bbox_wgs84": [11.55, 48.13, 11.56, 48.14]},
        )
        assert r.status_code == 200
        jid = r.json()["id"]

        # Run the worker inline (single job, exit quickly)
        run_forever(
            db_path=app.state.db_path, data_root=app.state.data_root,
            segmenter_factory=_StubSegmenter,
            wms_url="", layer="by_dop20c", max_pixels=6000,
            wms_version="1.1.1", fmt="image/png", crs="EPSG:25832",
            origin_x=0.0, origin_y=0.0,
            min_polygon_area_m2=0.01, safe_center_nodata_threshold=0.0,
            max_jobs=1, poll_interval=0.01, idle_exit_after=1,
        )

        job = get_job(app.state.db_path, jid)
        assert job["status"] == "READY_FOR_REVIEW"

        # Polygons endpoint returns at least one feature
        fc = client.get(f"/jobs/{jid}/polygons").json()
        assert fc["type"] == "FeatureCollection"
        assert len(fc["features"]) >= 1

        # Export produces a valid GPKG with both layers
        exp = client.post(f"/jobs/{jid}/export")
        assert exp.status_code == 200
        gpkg_path = Path(exp.json()["gpkg_path"])
        assert gpkg_path.exists()
        layers = fiona.listlayers(str(gpkg_path))
        assert {"detected_objects", "nodata_regions"}.issubset(layers)

        # Download works
        dl = client.get(f"/jobs/{jid}/export.gpkg")
        assert dl.status_code == 200
        assert len(dl.content) > 0
```

- [ ] **Step 2: Run**

Run: `pytest tests/test_end_to_end.py -v`
Expected: PASS.

If it fails due to GDAL availability of BuildVRT in the test environment, skip the test module at import time via `pytest.importorskip("osgeo")` and still commit.

- [ ] **Step 3: Commit**

```bash
git add tests/test_end_to_end.py
git commit -m "test: end-to-end integration across API and worker"
```

---

## Task 30: README

**Files:**
- Create: `README.md`

Goal: document local setup, how to install `sam3` and `gdal` (the two non-pip deps), how to run server and worker, and how to point at the LDBV OpenData WMS.

- [ ] **Step 1: Write `README.md`**

```markdown
# Text-to-Polygon Pipeline (Bayern DOP20 + SAM 3.1)

Local prototype that extracts georeferenced polygons from Bayerische DOP20
orthophotos using a free-text prompt and SAM 3.1 zero-shot segmentation.

See `docs/superpowers/specs/2026-04-22-text-to-polygon-design.md` for the
full design rationale.

## Requirements

- Python 3.12
- CUDA 12.6 + PyTorch 2.7 (RTX 4070 / 12 GB VRAM tested)
- `gdal` (install via conda, not pip — needed for `BuildVRT`)
- `sam3` from `facebookresearch/sam3` (install per upstream README)
- SQLite 3.35+ (bundled with Python)

## Install

```bash
conda create -n ki-geodaten python=3.12 gdal -c conda-forge
conda activate ki-geodaten
pip install -e .[dev]
# then follow https://github.com/facebookresearch/sam3 to install sam3
```

Download the SAM 3.1 checkpoint into `models/sam3.1_hiera_large.pt`.

## Configure

Copy `.env.example` → `.env`. The defaults already point at the LDBV OpenData WMS (no auth, CC BY 4.0):

```
WMS_URL=https://geoservices.bayern.de/od/wms/dop/v1/dop20
WMS_LAYER=by_dop20c
WMS_VERSION=1.1.1
WMS_FORMAT=image/png
WMS_CRS=EPSG:25832
WMS_MAX_PIXELS=6000
```

**Before first run:** complete Task 0 / `docs/wms-verification.md` and run the
adjacent-chunk edge test to verify the WMS samples its source raster
consistently across BBox boundaries.

## Run

Two processes in separate terminals:

```bash
./scripts/run-server.sh   # http://127.0.0.1:8000
./scripts/run-worker.sh   # GPU worker + supervisor
```

On Windows, run these via `bash` (Git Bash / WSL) or invoke the commands
directly:

```powershell
python -m uvicorn ki_geodaten.app.main:app --reload `
    --reload-dir ki_geodaten --reload-exclude "data/*" --reload-exclude "*.db*"
python -m ki_geodaten.worker.loop
```

## Tests

```bash
pytest
```

Tests do not require `sam3` or GPU; they stub the segmenter. The end-to-end
test needs `gdal.BuildVRT`, which is provided by the conda-installed GDAL.

## Known limits

- Only Bayern (LDBV OpenData WMS for DOP20).
- 1 km² max AOI (see Spec §9.1).
- `tile_preset` selects max object diameter: 64 m / 128 m / 192 m.
- Large preset is not interactive (~1 h per 1 km²).
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add setup, run, and limits overview"
```

---

## Self-Review Checklist (after implementation)

Before marking the plan done, verify:

**Spec invariants**
- [ ] Every Spec section §5.1–§12 has at least one implementing task.
- [ ] `validation_revision` is written ONLY by `validate_bulk`; never by export.
- [ ] `exported_revision` is written ONLY by the export endpoint.
- [ ] Task 0 WMS verification is complete (PNG RGBA confirmed + adjacent-chunk edge test passed) and `config.py` WMS defaults match `docs/wms-verification.md`.
- [ ] `nodata_regions.geometry_wkb` always stores the **safe-center** polygon (size-2·margin), never the full tile.
- [ ] `bbox_utm_snapped` is the unexpanded user AOI; the download bbox is only used at download time.
- [ ] GeoJSON responses and GPKG are both clipped to `bbox_utm_snapped` (Clip-Window-Semantik).
- [ ] AOI-clipped outputs are evaluated only against equivalently AOI-clipped reference data; they are not compared against full ALKIS objects.
- [ ] `masks_to_polygons` uses `connectivity=8` and excludes `value != 1`.
- [ ] Worker never holds a write transaction longer than one tile.
- [ ] Export route is wrapped by `app.state.export_lock`.
- [ ] Export route refreshes `finished_at` when setting `EXPORTED`, so retention starts at export time.
- [ ] `validate_bulk` uses `executemany`, never variadic `VALUES(...)`.
- [ ] GeoJSON precision ≤ 1e-6° (set via `shapely.set_precision`).
- [ ] GeoJSON geometry validation and polygon-only filtering run after `set_precision`, not only before it.
- [ ] Bayern fence AND UTM-area check are both enforced in `POST /jobs`.
- [ ] `run-server.sh` passes `--reload-exclude 'data/*' --reload-exclude '*.db*'`.
- [ ] Worker startup hook both marks DOWNLOADING/INFERRING jobs as FAILED **and** rmtree's orphan `data/dop/*` directories.
- [ ] Retention cleanup CASCADEs polygon/nodata deletion and removes `.gpkg` files.
- [ ] Retention cleanup deletes jobs in batches of ≤900 IDs, never one giant `IN (...)` clause.
- [ ] Empty GeoPackage layers are written with explicit Fiona schema and explicit CRS, not inferred from an empty GeoDataFrame.
- [ ] Re-export deletes `.gpkg`, `.gpkg-wal`, and `.gpkg-shm` before writing a fresh GeoPackage.
- [ ] NoData detection uses GDAL/raster dataset masks only (`src.dataset_mask(...) == 0`), never raw RGB-black heuristics.

**Bugs found during plan review — must stay fixed**
- [ ] `POST /jobs/{id}/export` is declared as `def` (not `async def`), so the blocking GDAL/Fiona call runs in FastAPI's threadpool and doesn't block the event loop.
- [ ] `validate_bulk` counts via `cursor.rowcount` AFTER `executemany` — relies on Python ≥3.12 cumulative semantics; `pyproject.toml` enforces this.
- [ ] Orchestrator catches `Exception` (NOT `BaseException`) around `segmenter.predict(...)` — `BaseException` would eat Ctrl-C / SystemExit.
- [ ] Orchestrator iterates tiles lazily (`for tile in iter_tiles(...)`) and obtains `tile_total` via a cheap `iter_grid(src, cfg)` count — never materializes all tiles in memory.
- [ ] Geojson route uses `functools.partial`, not a lambda, when submitting to `ProcessPoolExecutor` (lambdas are not picklable).
- [ ] Geojson ProcessPool functions return finished JSON strings; FastAPI routes return `Response(content=..., media_type="application/json")` and do not serialize large dicts on the event loop.
- [ ] WMS GetMap uses `SRS=` (1.1.1) not `CRS=` (1.3.0); BBOX axis order matches the negotiated WMS version (1.1.1 + EPSG:25832 → X,Y).
- [ ] WMS PNG response is decoded as RGBA and wrapped into a 4-band GeoTIFF with `colorinterp=(red,green,blue,alpha)` so `dataset_mask()` returns the alpha channel for NoData detection.
- [ ] Export route unpickles via module-level `update_status` import (no in-function `from ... import ... as _us` shadowing).
- [ ] `_job_view` returns `bbox_wgs84` as a parsed **array**, not a JSON string.
- [ ] Frontend styles polygons via native Leaflet `style` options (color / fillColor / dashArray), NOT CSS classes — `L.canvas()` ignores per-feature className.
- [ ] Frontend progress displays `(tile_completed + tile_failed) / tile_total`, not only successful tiles.
- [ ] Frontend `openJob(id)` flushes to the PREVIOUS `currentJobId` before switching, then clears `pendingUpdates`, then calls `hydratePending(id)` so session-persisted updates are recovered.
- [ ] Frontend `flushValidations()` only removes entries from `pendingUpdates` AFTER a successful POST; on network/server error the batch stays in memory and is re-persisted for retry.
- [ ] Frontend `pagehide` uses `fetch(..., keepalive: true)` for the final JSON POST; it does not use JSON `sendBeacon()`.
- [ ] Frontend validation flush has an `isFlushing` guard so buffer-threshold clicks cannot spawn parallel duplicate POSTs.
- [ ] `Sam3Segmenter.predict()` converts all SAM tensors to detached CPU NumPy/Python values before `local_mask_nms`; no GPU tensor reaches NumPy code.
- [ ] SAM mask conversion uses `.detach().cpu().numpy().copy()` so returned NumPy arrays do not keep PyTorch tensor storage alive.
- [ ] Per-tile SAM raw outputs and temporary tensor refs are deleted before returning from `predict()`, so the orchestrator's `torch.cuda.empty_cache()` can release VRAM.
- [ ] `local_mask_nms` runs a `box_pixel` overlap prefilter before any dense mask IoU/containment comparison.
- [ ] Orchestrator clears per-tile exception traceback/context before `torch.cuda.empty_cache()` after OOM/inference failures.

**Production hardening**
- [ ] `app.state.token_counter` uses the **real SAM text tokenizer only** in production; it never builds the SAM vision model or allocates CUDA memory. The whitespace-split fallback logs a WARNING (never silent).
- [ ] Task 12A verified the actual upstream SAM 3.1 prediction and tokenizer APIs.
- [ ] `create_app(executor_factory=..., token_counter=...)` accepts overrides so tests can inject a `ThreadPoolExecutor` and a deterministic token counter — production uses the defaults.
- [ ] `scripts/run-worker.sh` traps SIGINT/SIGTERM so a single Ctrl-C stops the supervisor cleanly.
- [ ] End-to-end test uses `pytest.importorskip("osgeo")` / `pytest.importorskip("fiona")` so the suite degrades gracefully when GDAL/Fiona aren't installed.

**Resolved spec decision**
- [ ] Margin-expansion semantics are fixed: download expands by `CENTER_MARGIN_PX × 0.2 m` = 32/64/96 m. Max object diameter remains `2 × CENTER_MARGIN_PX × 0.2 m` = 64/128/192 m.

If a checkbox fails, fix inline and re-run the relevant test.
