# Text-to-Polygon Pipeline — Design Spec

**Datum (Entwurf):** 2026-04-22
**Letzte Revision:** 2026-04-23
**Status:** Entwurf zur Implementierung
**Scope:** Prototyp (Exploration), Bayern, lokal auf RTX 4070 / 12 GB VRAM / 64 GB RAM

---

## 1. Zweck

Automatisierte Extraktion von georeferenzierten Vektor-Polygonen aus DOP20-Orthophotos in Bayern anhand frei formulierbarer Text-Prompts (Zero-Shot). Der Prototyp dient der methodischen Erkundung:

- Validierung von Zero-Shot-Segmentierung (SAM 3.1) auf amtlichen Luftbildern
- Evaluation von Precision / Recall / IoU gegen Referenzdaten (z. B. ALKIS)
- Basis für spätere Erweiterungen (weitere Bundesländer, Batch-Vergleich mehrerer Modelle)

Nicht-Ziele: kommerzielles SaaS, Echtzeit-Interaktion, Multi-User-Deployment, Skalierung über eine GPU hinaus.

## 2. Funktionale Anforderungen

- **FR-1** — Nutzer zeichnet Bounding Box in Leaflet-Karte, gibt freien Text-Prompt ein, startet Job.
- **FR-2** — System lädt DOP20 vom Bayerischen LDBV-WCS, führt Pipeline aus, präsentiert Ergebnis-Polygone auf der Karte.
- **FR-3** — Alle detektierten Polygone sind **initial als `ACCEPTED` vormarkiert**. Der Nutzer arbeitet im Subtraktions-Workflow: er klickt nur False Positives an, die auf `REJECTED` gesetzt werden. Das spart bei 3 000 Polygonen 2 990 redundante Klicks gegenüber einem Opt-in-Modell.
- **FR-4** — Export der akzeptierten Polygone als zweischichtiges GeoPackage (`detected_objects` + `nodata_regions`) in EPSG:25832.
- **FR-5** — Jobs laufen asynchron; Nutzer kann UI verlassen und später zurückkehren; Job-Historie sichtbar.
- **FR-6** — Ein fehlgeschlagener Job kann durch Neuanlage (gleiche BBox + Prompt) erneut versucht werden; es gibt bewusst keinen In-place-Retry, um Zustandsinkonsistenzen zu vermeiden.

## 3. Nicht-funktionale Anforderungen

- **NFR-1 (VRAM):** Nie mehr als ein SAM-3.1-Modell gleichzeitig im VRAM — Concurrency physikalisch auf 1 begrenzt.
- **NFR-2 (Methodische Validität):** Keine stillen False Negatives. Fehlgeschlagene Kacheln werden als `nodata_regions` dokumentiert.
- **NFR-3 (Nativ-Auflösung):** WCS-Requests werden grid-aligned zum LDBV-Pixelraster abgesetzt — keine Server-seitige Interpolation. Dies gilt für die Außen-BBox **und** für jeden paginierten Sub-Chunk.
- **NFR-4 (Kein Blocking):** Webserver blockiert nie auf GPU-Arbeit. Antwortzeit auf `POST /jobs` < 50 ms.
- **NFR-5 (Crash-Resilienz):** Ein fehlgeschlagener Job oder Worker-Absturz blockiert die Queue nicht länger als einen Worker-Neustart (~10 s).
- **NFR-6 (Disk-Hygiene):** Kein unbegrenztes Wachstum von Zwischendateien. DOP-Chunks werden nach erfolgreichem Export gelöscht; `nodata_regions` und `polygons` bleiben in SQLite **bis zur Retention-Grenze** (NFR-7).
- **NFR-7 (DB-Retention):** `FAILED`- und `EXPORTED`-Jobs werden nach `RETENTION_DAYS = 7` vollständig gelöscht (inkl. Polygone, NoData-Regionen, GeoPackage-Datei). Ein täglicher Cleanup-Task im Worker (Leerlaufzeit oder beim Worker-Restart) führt `DELETE FROM jobs WHERE status IN ('FAILED','EXPORTED') AND finished_at < ?` aus (Polygone und NoData-Regionen verschwinden via `ON DELETE CASCADE`), löscht `data/results/{job_id}.gpkg`, und schließt mit `VACUUM`. Verhindert, dass `jobs.db` bei 1 000 Explorations-Läufen auf 10 GB anschwillt.

## 4. Architektur-Überblick

### 4.1 Prozess-Topologie

Zwei voneinander isolierte Prozesse, die ausschließlich über eine SQLite-Datenbank kommunizieren:

```
┌─────────────────────────────┐         ┌──────────────────────────────┐
│  Webserver-Prozess          │         │  GPU-Worker-Prozess          │
│  (uvicorn, FastAPI)         │         │  (python -m ki_geodaten.worker)
│                             │         │                              │
│  - HTTP-Endpunkte           │         │  - Pollt SQLite alle 2 s     │
│  - Leaflet-UI ausliefern    │         │  - SAM 3.1 geladen (~6 GB)   │
│  - keine GPU-Zugriffe       │         │  - strikt sequenziell        │
└──────────────┬──────────────┘         └───────────────┬──────────────┘
               └────────┐         ┌─────────────────────┘
                        ▼         ▼
                   ┌────────────────────┐
                   │  SQLite (WAL-Mode) │
                   │  jobs.db           │
                   └────────────────────┘
```

- **Concurrency = 1** ist OS-garantiert (ein Worker-Prozess).
- **WAL-Modus** erlaubt parallele Reads + einen gleichzeitigen Writer ohne „database is locked".
- **Worker-Restart-Supervisor** (`scripts/run-worker.sh`): nach `MAX_JOBS_PER_WORKER = 50` terminiert der Worker sauber und wird via `while true`-Loop neu gestartet. Löst PyTorch-VRAM-Fragmentierung.

### 4.2 Schichten-Modell

```
ki_geodaten/
├── app/          # Schicht 1: Webserver (FastAPI + Leaflet)
├── worker/       # Schicht 2: GPU-Worker (Poll-Loop, Orchestrator)
├── pipeline/     # Schicht 3: Pure Bildverarbeitung (seitenwirkungsfrei)
├── jobs/         # Querschnitt: SQLite-Zugriff, Pydantic-Modelle, Geo-Utils
├── config.py
├── data/
└── tests/
```

Abhängigkeitsrichtung: `app/` → `jobs/`; `worker/` → `pipeline/` und `jobs/`. Weder `app/` noch `pipeline/` kennen sich.

### 4.3 Koordinaten-Referenzsysteme (Zuständigkeiten)

| Ort | CRS | Grund |
|---|---|---|
| Leaflet-UI (Draw, Display) | EPSG:4326 (WGS84) | Leaflet-Standard |
| HTTP-Bodies (BBox im POST /jobs) | EPSG:4326 | Leaflet liefert so |
| Interne Persistenz (SQLite WKB) | EPSG:25832 | LDBV-natives CRS, keine Transformationsverluste |
| WCS-Anfragen an LDBV | EPSG:25832 (grid-snapped) | NFR-3 |
| GeoPackage-Export | EPSG:25832 | FR-4 |
| GeoJSON-Antworten an UI (`/jobs/{id}/polygons`) | EPSG:4326 | Leaflet-Standard |

Die Transformation 25832 → 4326 passiert **nur** in `app/routes/jobs.py` unmittelbar vor dem GeoJSON-Serialize, über eine cached `pyproj.Transformer`-Instanz. `pipeline/` kennt nur EPSG:25832.

## 5. Pipeline-Stufen

Jede Stufe ist ein Modul in `pipeline/` mit klar typisierter Ein-/Ausgabe, isoliert testbar.

### 5.1 `wcs_client.py` — DOP20-Download

**Aufgabe:** Bounding Box in EPSG:25832 → lokales VRT (Virtual Raster) über heruntergeladene DOP20-Chunks.

**Kritische Aspekte:**

1. **Grid-Snapping (NFR-3):** BBox aus WGS84 (Leaflet) wird nach EPSG:25832 transformiert. **Naive 4-Ecken-Projektion ist falsch:** Ein achsenparalleles WGS84-Rechteck ist in UTM nicht mehr achsenparallel (die Meridiane konvergieren), eine Corner-Only-Transformation schneidet die gekrümmten Ränder ab. Korrekt ist `pyproj.Transformer.transform_bounds()`, das die Ränder mit mehreren Stützpunkten (Default 21 pro Kante) densifiziert und dann die umschließende UTM-Box berechnet:

    ```python
    transformer = pyproj.Transformer.from_crs(4326, 25832, always_xy=True)
    minx, miny, maxx, maxy = transformer.transform_bounds(
        lon_min, lat_min, lon_max, lat_max, densify_pts=21
    )
    ```

    Danach wird jede Koordinate auf das verifizierte DOP20-Raster geschnappt. **Wichtig:** Das Snap darf **nicht** implizit von einem Grid-Origin `(0, 0)` ausgehen. Wenn die Coverage einen abweichenden Origin `(origin_x, origin_y)` hat, muss dieser in die Formel eingehen; andernfalls entsteht trotz korrekter 0.2-m-Schrittweite ein systematischer Subpixel-Shift. Zusätzlich ist naives `math.floor(x / 0.2) * 0.2` wegen IEEE-754-Repräsentation von 0.2 nicht exakt — z. B. ergibt `math.floor(0.6 / 0.2) * 0.2 == 0.4` statt `0.6`. Wir verwenden deshalb `decimal.Decimal`-basiertes, origin-sensitives Snapping:

    ```python
    from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
    _STEP = Decimal("0.2")
    _ORIGIN_X = Decimal(str(origin_x))
    _ORIGIN_Y = Decimal(str(origin_y))

    def snap_floor(x: float, origin: Decimal) -> float:
        return float((((Decimal(str(x)) - origin) / _STEP).to_integral_value(ROUND_FLOOR) * _STEP) + origin)

    def snap_ceil(x: float, origin: Decimal) -> float:
        return float((((Decimal(str(x)) - origin) / _STEP).to_integral_value(ROUND_CEILING) * _STEP) + origin)

    # Anwendung:
    minx_s = snap_floor(minx, _ORIGIN_X)   # enlargen Richtung außen
    miny_s = snap_floor(miny, _ORIGIN_Y)
    maxx_s = snap_ceil(maxx, _ORIGIN_X)
    maxy_s = snap_ceil(maxy, _ORIGIN_Y)
    ```

    `origin_x` / `origin_y` stammen aus den verifizierten WCS-Capabilities der Coverage. Falls die Coverage tatsächlich auf `(0, 0)` liegt, degeneriert die Formel auf den einfacheren Spezialfall. Ohne origin-sensitives Snap führt der WCS-Server trotz korrekter Schrittweite Sub-Pixel-Interpolation durch.

2. **WCS-Pagination:** LDBV-Coverages haben ein hartes Pixel-Limit (`MAX_WCS_PIXELS = 4000` in `config.py`, exakter Wert beim ersten `GetCapabilities`-Call verifizieren). Die snap-gealignte Außen-BBox wird in ein Grid aus Sub-BBoxes zerlegt, von denen **jede ebenfalls auf das 0.2-m-Raster ausgerichtet ist**. Innenliegende Chunks nutzen die volle Kantenlänge `MAX_WCS_PIXELS * 0.2 m = 800 m`; Rand-Chunks dürfen kleiner sein, müssen aber weiterhin aus einer ganzzahligen Pixelzahl bestehen. Entscheidend ist nicht „immer 800 m“, sondern **nahtlose gemeinsame Chunk-Kanten auf demselben 0.2-m-Grid**. Jeder Chunk wird als `data/dop/{job_id}/chunk_{r}_{c}.tif` gespeichert.

3. **Harte Timeouts:** Jeder HTTP-Request mit `timeout=(10, 60)` (connect / read). Retries via `urllib3.Retry(total=3, backoff_factor=2, status_forcelist=[500, 502, 503, 504])` → exponentielle Backoff 2–8 s. Nach endgültigem Fehlschlag: `WCSError`-Exception, Job `FAILED` mit `error_reason='WCS_TIMEOUT'`.

4. **Margin-Expansion (Edge-Protection):** Nach dem Grid-Snapping wird die BBox in alle vier Richtungen um `CENTER_MARGIN × 0.2 m` erweitert (64 m für small, 128 m für medium, 192 m für large — passend zum `tile_preset` des Jobs). **Warum zwingend:** Der BBox-Center-Filter in 5.4 verwirft Polygone, deren Mittelpunkt im Rand-Streifen liegt. Zwischen Kacheln ist das korrekt — dort übernimmt der Nachbar. Am absoluten VRT-Rand gibt es aber keinen Nachbarn. Ohne diese Expansion wäre die äußere `CENTER_MARGIN`-Zone der vom Nutzer angeforderten Fläche systematisch leer — das verfälscht jede ALKIS-Evaluation. Mit der Expansion liegt die angeforderte Fläche vollständig in Safe-Zentren. Der Download wird etwas größer (für medium-Preset bei 2×2 km BBox: 2.256×2.256 km), bleibt aber unkritisch.

   **Wichtig für alle Downstream-Schritte:** Die expandierte Download-BBox ist ein **internes Hilfskonstrukt**. `jobs.bbox_utm_snapped` bleibt die **unexpandierte, vom Nutzer angeforderte AOI** nach Transform + Grid-Snap. Alle user-sichtbaren Ergebnisse (GeoJSON, Export, Metrik-Inputs) werden später wieder auf genau diese AOI zurückgeführt.

5. **Minimalgröße garantieren:** Nach Expansion prüft der Client, ob `(maxx - minx) < TILE_SIZE * 0.2 m = 204.8 m` oder analog in y. Falls ja, wird die BBox **symmetrisch weiter expandiert** auf exakt 204.8 m (ebenfalls grid-aligned). So ist garantiert, dass das resultierende VRT mindestens 1024 × 1024 px groß ist und `tiler.iter_tiles()` nicht auf Out-of-Bounds-Reads läuft.

6. **Pixel-Dimensionen für WCS-Request:** Die `WIDTH`/`HEIGHT`-Parameter des WCS-`GetCoverage`-Calls werden aus der BBox berechnet. **Achtung Fließkomma-Falle:** `int((maxx_s - minx_s) / 0.2)` kann wegen IEEE-754-Ungenauigkeiten 1499 statt 1500 ergeben (z. B. `300.0 / 0.2 == 1499.9999999999998`). Der Server würde dann 300 m in 1499 px interpolieren — das macht das Grid-Snapping zunichte. Lösung: **zwingend `round()` statt `int()`**, oder noch robuster via `Decimal`:
    ```python
    width_px = round((maxx_s - minx_s) / 0.2)
    # oder: int((Decimal(str(maxx_s)) - Decimal(str(minx_s))) / Decimal("0.2"))
    ```

7. **Chunk-Grenzen lückenlos, nicht versetzt:** Benachbarte Chunks müssen an derselben Rasterkante zusammentreffen: `next_minx = prev_maxx`, `next_miny = prev_maxy`. Ein künstlicher `+ 0.2 m`-Versatz würde zwischen zwei Chunks eine echte 1-Pixel-Lücke erzeugen; `gdal.BuildVRT` mosaikiert solche Lücken als NoData-Streifen. Die Robustheit kommt daher, dass alle Chunk-Bounds und `WIDTH`/`HEIGHT` aus demselben grid-gesnappten Koordinatensystem und ganzzahligen Pixelzahlen abgeleitet werden. Zu verifizieren beim ersten `GetCoverage`: zwei benachbarte Chunks mit identischer Grenzkante erzeugen im VRT **weder Lücke noch Doppelreihe**.

8. **VRT-Erzeugung:** Nach Download `gdal.BuildVRT(out.vrt, chunk_files)` — reines XML, kein Pixeltransfer. Downstream-Tiler liest on-demand, RAM-Footprint ≈ 0. VRT-Pfad wird in `jobs.dop_vrt_path` gespeichert.

**Signatur:**
```python
class WCSError(Exception): ...

def download_dop20(bbox_utm: BBox25832, center_margin_px: int, out_dir: Path) -> Path:
    """
    Lädt DOP20 paginiert, schreibt chunk_*.tif + out.vrt nach out_dir.
    bbox_utm MUSS bereits auf 0.2m-Grid geschnappt sein.
    center_margin_px steuert die Edge-Expansion (64/128/192 m je nach tile_preset).
    Returns: Path zur .vrt-Datei.
    Raises: WCSError bei endgültigem HTTP-Fehler.
    """
```

### 5.2 `tiler.py` — Lazy Tile-Iteration

**Aufgabe:** VRT + TileConfig → Iterator von `Tile`-Objekten für die Inferenz.

**TileConfig-Parameter (bei DOP20, 20 cm/px):** `TILE_SIZE` ist konstant 1024 px. `OVERLAP` und daraus abgeleitetes `CENTER_MARGIN = OVERLAP / 2` sind **pro Job konfigurierbar** über ein Preset-System, weil die maximale Objektgröße vom Zielprompt abhängt (siehe 5.4, 8). Default ist `medium`.

| Preset | `OVERLAP` | `CENTER_MARGIN` | Max-Objekt | Safe-Zentrum | Tile-Step | Typische Prompts | Tiles / 1×1 km² |
|---|---|---|---|---|---|---|---|
| `small` | 320 px | 160 px | **64 m** | 704 × 704 px | 704 px | solar panel, pool, car, tree | ~64 |
| `medium` (Default) | 640 px | 320 px | **128 m** | 384 × 384 px | 384 px | building, house, school, warehouse | ~196 |
| `large` | 960 px | 480 px | **192 m** | 64 × 64 px | 64 px | industrial hall, airport, stadium | ~6 200 (**sehr teuer**, ~1 h) |

**Invarianten aller Presets:** `TILE_STEP = SAFE_CENTER_SIZE` ⇒ Safe-Zentren kacheln die Ebene exklusiv. `MAX_OBJECT = 2 × CENTER_MARGIN` ⇒ Objekte, deren BBox-Center im Safe-Zentrum liegt, sind noch vollständig im Tile. Jede Änderung muss diese Invarianten erhalten.

**Edge-Handling am VRT-Rand:** Wenn `(VRT_width - TILE_SIZE) % (TILE_SIZE - OVERLAP) != 0`, würden Rand-Kacheln über das VRT hinausragen. Lösung: **letzte Kachel pro Zeile/Spalte wird nach innen verschoben**, sodass ihre rechte/untere Kante exakt mit dem VRT-Rand abschließt (daher evtl. größerer Overlap mit der vorletzten Kachel — unkritisch). Kein Padding, kein Rand-Skip. Wichtig: `tile_row` / `tile_col` sind dabei **logische Grid-Indizes**, die `iter_grid()` explizit mitliefert; sie dürfen **nicht** aus `row_off // step` rekonstruiert werden, weil die nach innen verschobene letzte Kachel sonst denselben Index wie ihre Vorgängerin bekommen kann.

**Tile-Datenklasse:**
```python
@dataclass(frozen=True)
class Tile:
    array: np.ndarray           # Shape (H=1024, W=1024, C=3), dtype uint8, RGB-Reihenfolge
    pixel_origin: tuple[int, int]   # (row, col) der Tile-Oberkante-links im VRT
    size: int                   # = TILE_SIZE (immer 1024)
    center_margin: int          # aus Preset
    affine: Affine              # rasterio-Affintransform für dieses Tile (EPSG:25832)
    tile_row: int               # Grid-Indizes für Traceability und NoData-Persistenz
    tile_col: int

@dataclass(frozen=True)
class TileConfig:
    size: int = 1024
    overlap: int = 640          # medium-Preset Default
    center_margin: int = 320    # = overlap / 2

    @classmethod
    def from_preset(cls, preset: Literal["small", "medium", "large"]) -> "TileConfig":
        params = {"small": (320, 160), "medium": (640, 320), "large": (960, 480)}
        overlap, margin = params[preset]
        return cls(size=1024, overlap=overlap, center_margin=margin)
```

**Datentyp-Begründung:** `uint8` statt `float32`, weil (a) DOP20 nativ 8-bit-RGB ist und (b) SAM-3-Preprocessing die Normalisierung intern vornimmt. Spart Faktor 4 RAM pro Tile (3 MB statt 12 MB).

**GDAL-Dateihandle-Kontrolle und Per-Tile-Affine:** Ein VRT aus dutzenden Chunks kann GDAL veranlassen, viele TIFFs gleichzeitig offen zu halten. Standard-Linux-`ulimit -n` ist oft 1024 — bei großen Jobs droht `Too many open files`. Zusätzlich muss die Affine-Transformation pro Kachel berechnet werden (nicht die globale VRT-Affine), damit `rasterio.features.shapes` später korrekte UTM-Koordinaten liefert:

```python
from rasterio.windows import Window

with rasterio.Env(GDAL_MAX_DATASET_POOL_SIZE=256):
    with rasterio.open(vrt_path) as src:
        for tile_row, tile_col, row_off, col_off in iter_grid(src, cfg):
            window = Window(col_off=col_off, row_off=row_off, width=cfg.size, height=cfg.size)
            tile_affine = src.window_transform(window)   # ← zwingend per-Tile, nicht src.transform!
            array_chw = src.read(indexes=[1, 2, 3], window=window)   # nur RGB; Shape (C, H, W)
            array_hwc = array_chw.transpose(1, 2, 0)                 # Shape (H, W, C)
            yield Tile(
                array=array_hwc,
                pixel_origin=(row_off, col_off),
                size=cfg.size,
                center_margin=cfg.center_margin,
                affine=tile_affine,
                tile_row=tile_row,
                tile_col=tile_col,
            )
```

Das zwingt GDAL, alte Handles aggressiv zu schließen und wiederzuverwenden. `src.window_transform(window)` hängt den Pixel-Offset der Kachel korrekt an die globale Affine, sodass lokale `row`/`col`-Koordinaten der Maske in 5.4 automatisch zu UTM-Koordinaten werden.

**NoData-Erkennung vor Inferenz:** DOP20-Coverages haben endliche Grenzen (Landesgrenzen, Lücken im Bestand). Wenn die Margin-Expansion aus 5.1 Punkt 4 über diese Grenzen hinaus geht oder ein Chunk-Download Lücken hat, liefert der Server NoData-Pixel. **Primärsignal dafür sind Raster-Masken / GDAL-NoData-Metadaten, nicht die RGB-Werte selbst.** SAM 3.1 halluziniert an harten Datenkanten Objekte, die die Evaluation verfälschen. Für NFR-2 ist aber **nicht die gesamte Kachel**, sondern das **Safe-Zentrum** entscheidend: Sobald dort NoData liegt, ist die exklusive Verantwortungszone der Kachel kompromittiert und ein stilles False Negative möglich.

Vor Übergabe an SAM prüft der Tiler deshalb pro Kachel zuerst die **verifizierte Raster-Maske** des WCS-Outputs:

```python
nodata_mask = src.read_masks(1, window=window) == 0
safe = nodata_mask[
    cfg.center_margin : cfg.size - cfg.center_margin,
    cfg.center_margin : cfg.size - cfg.center_margin,
]

if safe.mean() > SAFE_CENTER_NODATA_THRESHOLD:   # für NFR-2 praktisch 0.0
    # Exklusive Verantwortungszone kompromittiert → Safe-Zentrum als NoData markieren
    skip_tile(reason="NODATA_PIXELS")
    continue
```

NoData **außerhalb** des Safe-Zentrums ist tolerierbar: Der Center-Keep-Filter verwirft Halluzinationen aus diesen Randbereichen ohnehin. NoData **innerhalb** des Safe-Zentrums darf dagegen nie still weiterlaufen, weil sonst Objekte mit Mittelpunkt in der exklusiven Verantwortungszone unbemerkt verloren gehen könnten.

RGB-Schwarz `(0,0,0)` darf nur als **sekundärer Fallback** verwendet werden, wenn beim echten WCS-Output explizit verifiziert wurde, dass schwarze Pixel tatsächlich NoData kodieren und nicht legitime Bildinhalte sein können. Dieser Fallback ist im Spec **nicht Default**.

**Signatur:**
```python
def iter_tiles(vrt_path: Path, cfg: TileConfig) -> Iterator[Tile]:
    """Lazy via rasterio.windows.Window; liest pro Tile nur den benötigten Ausschnitt."""
```

### 5.3 `segmenter.py` — SAM 3.1 Inferenz

**Aufgabe:** `(Tile, Prompt) → list[MaskResult]`, wobei eine Liste **tile-lokal deduplizierter** Instanzen zurückgegeben wird.

**API-Semantik:** SAM 3.1 akzeptiert einen Text-Prompt und liefert über seinen internen DETR-Detektor **alle** Instanzen des Konzepts im Bild als separate Masken. Für den Prompt „solar panel" auf einer Kachel mit 12 Modulen → 12 `MaskResult`-Einträge.

**Maßnahmen:**
- Modell wird **einmalig** beim Worker-Start geladen (nicht pro Job), spart ~8 s pro Job.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` als Env-Variable reduziert Fragmentierung.
- **Prompt-Längenprüfung muss auf der finalen Encoder-Sequenz passieren:** Das API-Limit orientiert sich nicht am rohen User-String, sondern an der **tatsächlich an den Text-Encoder übergebenen Token-Sequenz** nach Anwendung des Modell-Templates und aller Spezialtokens. Der Webserver darf daher nicht nur „77 User-Tokens“ zählen, sondern muss denselben Encode-Pfad wie `Sam3Segmenter.predict()` bzw. dessen Text-Wrapper verwenden.
- Per-Tile try/except um `predict()`: bei `torch.cuda.OutOfMemoryError` wird das **Safe-Zentrum** der Kachel als `nodata_regions` (reason=`OOM`) markiert, der Loop fährt fort. Bei `Exception`: reason=`INFERENCE_ERROR` mit Traceback-Tail.
- **Tile-lokale Masken-Deduplikation ist zwingend:** Die BBox-Center-Rule aus 5.4 löst nur Cross-Tile-Dopplungen. Innerhalb **einer** Kachel kann SAM für dasselbe Objekt mehrere stark überlappende Masken liefern. Deshalb werden die Rohmasken vor der Polygonisierung score-absteigend dedupliziert: Ein Kandidat wird verworfen, wenn er zu einer bereits behaltenen Maske entweder `mask_iou >= LOCAL_MASK_NMS_IOU` erreicht **oder** eine starke Verschachtelung aufweist (`intersection_area / min(area_a, area_b) >= LOCAL_MASK_CONTAINMENT_RATIO`). Damit verschwinden tile-interne Mehrfachdetektionen (z. B. Dach vs. Dach+Schatten) bevor sie als Polygone persistiert werden.
- **Tensor-Scope-Cleanup pro Iteration:** `torch.cuda.empty_cache()` gibt nur unreferenzierten VRAM frei — Python-Referenzen blockieren die Freigabe. Die alte Kachel bleibt bis zur nächsten Schleifeniteration im Scope und überlagert die neue Allokation. Auf einer 12-GB-Karte kann das am Kachelwechsel zum Crash führen. Regel am Ende **jedes** Schleifendurchlaufs (normal **und** im except-Block):
   ```python
   # innerhalb der for-tile-Schleife nach dem Call
   masks_cpu = [MaskResult(m.mask.cpu().numpy(), m.score, m.box_pixel) for m in gpu_masks]
   del gpu_masks, tile   # Referenzen freigeben
   torch.cuda.empty_cache()
   yield masks_cpu
   ```
   Alles, was nach dem Yield gebraucht wird (Masken, Scores, Boxes), wird vorher auf die CPU kopiert.

**Datenklasse & Signatur:**
```python
@dataclass(frozen=True)
class MaskResult:
    mask: np.ndarray    # Shape (H, W), dtype bool, True = Objekt-Pixel
    score: float        # SAM-3.1 Confidence, [0, 1]
    box_pixel: tuple[int, int, int, int]   # (row_min, col_min, row_max, col_max) exklusiv

class Sam3Segmenter:
    def __init__(self, checkpoint: Path, device: str = "cuda"): ...

    def predict(self, tile: Tile, prompt: str) -> list[MaskResult]:
        """
        Liefert eine lokal deduplizierte MaskResult-Instanz pro detektiertem Konzept-Vorkommen.
        Leere Liste, wenn nichts erkannt wurde.
        Raises: torch.cuda.OutOfMemoryError (vom Caller abgefangen).
        """
```

### 5.4 `merger.py` — Center-Keep + Polygonisierung

**Aufgabe:** `list[MaskResult] → GeoDataFrame[EPSG:25832]` pro Tile. Die Per-Tile-Ergebnisse werden im Orchestrator concat-ed.

**Drei Schritte, in dieser Reihenfolge:**

1. **Center-Keep-Filter (NFR-2) — BBox-Center-basiert:**
    - Sicheres Zentrum: Pixelbereich `[margin, size-margin) × [margin, size-margin)` (halboffen, damit Zentren nicht überlappen).
    - **Regel:** Polygon wird behalten, wenn der **Mittelpunkt seiner `box_pixel`** (aus SAM-Output, Section 5.3) im sicheren Zentrum liegt:
      ```python
      r_min, c_min, r_max, c_max = mask.box_pixel
      cr = (r_min + r_max) / 2
      cc = (c_min + c_max) / 2
      keep = margin <= cr < size - margin and margin <= cc < size - margin
      ```
    - **Warum BBox-Center statt echter Centroid:** Ein geometrischer Centroid einer komplex geformten Maske (z. B. C- oder L-förmiges Gebäude) kann außerhalb des Polygons liegen (im Innenhof) — und damit nahe einer Safe-Center-Grenze in Float-instabile Bereiche driften. Der BBox-Mittelpunkt ist dagegen (a) aus SAMs integer-Box direkt ableitbar, also FP-frei, (b) billiger zu berechnen, (c) unabhängig von der Polygon-Topologie.
    - **Eindeutige Zuordnung:** Weil `TILE_STEP = TILE_SIZE - OVERLAP = SAFE_CENTER_SIZE`, kacheln die Safe-Zentren die Ebene lückenlos und ohne Überlappung (halboffenes Intervall). Der BBox-Mittelpunkt jedes Objekts liegt in **genau einem** Safe-Zentrum ⇒ jedes Objekt wird von genau einem Tile behalten (keine Dopplungen, keine Verluste an Safe-Center-Grenzen).
    - **Max. Objektgröße (Preset-abhängig):** Damit ein Objekt, dessen BBox-Mittelpunkt am Safe-Center-Rand liegt, noch vollständig innerhalb der Tile-Bounds (1024 px) liegt und SAM es ungekürzt sieht, gilt: `max_object_diameter ≤ 2 × CENTER_MARGIN`. Das ergibt 64 m (small), 128 m (medium), 192 m (large). Objekte größer als die Preset-Obergrenze werden zerschnitten, die IoU gegen ALKIS kollabiert — der Nutzer muss das passende Preset für seinen Prompt wählen (Section 8). Siehe Section 14.
2. **Raster → Geometrie-Dicts:** `rasterio.features.shapes(...)` liefert **keine** Shapely-Geometrien, sondern Tupel `(geom_dict, pixel_value)`. Zusätzlich muss der Hintergrund explizit ausgeschlossen werden, sonst entsteht u. a. das invertierte Full-Tile-Hintergrundpolygon. Für SAM-Masken ist dabei **`connectivity=8` zwingend**: Die Default-4-Nachbarschaft trennt diagonal berührende Pixelketten künstlich auf und kann schräg stehende Masken unnötig zerfasern. Korrekt ist daher:

    ```python
    from rasterio.features import shapes
    from shapely.geometry import shape as shapely_shape

    for geom_dict, value in shapes(
        mask_result.mask.astype("uint8"),
        mask=mask_result.mask,
        transform=tile.affine,
        connectivity=8,
    ):
        if value != 1:
            continue
        geom = shapely_shape(geom_dict)   # GeoJSON-like dict -> Shapely-Geometrie
    ```

3. **Topologische Bereinigung:** Erst **nach** der Konvertierung in eine Shapely-Geometrie wird `shapely.make_valid(geom)` aufgerufen. Das Resultat kann ein `Polygon`, ein `MultiPolygon`, eine `GeometryCollection` (mit gemischten Typen) oder Artefakte wie `LineString`/`Point` (bei nur diagonal-benachbarten Pixeln) sein. Der Bereinigungs-Filter iteriert und behält **ausschließlich Polygon-artige Komponenten**:

    ```python
    from shapely.geometry import Polygon, MultiPolygon, GeometryCollection

    def extract_polygons(geom) -> list[Polygon]:
        if isinstance(geom, Polygon):
            return [geom] if not geom.is_empty else []
        if isinstance(geom, MultiPolygon):
            return [p for p in geom.geoms if not p.is_empty]
        if isinstance(geom, GeometryCollection):
            out = []
            for sub in geom.geoms:
                out.extend(extract_polygons(sub))   # rekursiv, ignoriert LineString/Point
            return out
        return []   # LineString, Point etc. werden verworfen
    ```

    Danach werden Polygone mit `area < MIN_POLYGON_AREA_M2` oder `is_empty` verworfen. Ohne diese Polygon-Only-Filterung scheitert `exporter.py` beim Schreiben der Polygon-Layer in GeoPackage mit `FionaValueError`.

Ein globales, tile-übergreifendes IoU-NMS ist **nach** der tile-lokalen Masken-Deduplikation nicht mehr nötig: Die BBox-Center-Rule ist über Kachelgrenzen exklusiv und vermeidet Cross-Tile-Duplikate. Die NMS-Pflicht besteht also **lokal pro Tile**, nicht global über den gesamten Job.

**Signaturen:**
```python
def keep_center_only(masks: list[MaskResult], tile: Tile) -> list[MaskResult]:
    """Filtert nach BBox-Mittelpunkt im Safe-Zentrum (halboffenes Intervall)."""

def masks_to_polygons(masks: list[MaskResult], tile: Tile) -> gpd.GeoDataFrame:
    """
    Wendet Raster→Vektor, make_valid und MIN_POLYGON_AREA_M2-Filter an.
    Spalten: geometry (EPSG:25832), score, source_tile_row, source_tile_col.
    """
```

### 5.5 `exporter.py` — Zweischichtiges GeoPackage

**Aufgabe:** `(detected_gdf, nodata_gdf, requested_bbox) → .gpkg`. Wird beim ersten Export aus `READY_FOR_REVIEW` erzeugt und bei Folge-Exporten aus `EXPORTED` überschrieben. Export ist **wiederholbar**; `EXPORTED` bedeutet „mindestens einmal exportiert“, nicht „Review eingefroren“.

Zwei Layer im Output, beide in EPSG:25832:

- **`detected_objects`**: akzeptierte Polygone + Spalten `score`, `source_tile_row`, `source_tile_col`.
- **`nodata_regions`**: **Safe-Zentrum-Footprints** (nicht Tile-Footprints!) der fehlgeschlagenen Tiles + Spalte `reason`.

**Kritisch:** Die NoData-Geometrie ist das **Safe-Zentrum** (z. B. 384 × 384 px im medium-Preset), nicht der volle Tile-Footprint (1024 × 1024 px). Grund: Tiles überlappen um 640 px (medium); würde man den Full-Footprint als NoData markieren, exkludierte man damit Safe-Zentren der 8 Nachbar-Tiles, die dort erfolgreich segmentiert haben — ALKIS-Features in diesen Bereichen würden fälschlich aus der Evaluation fliegen und die Metriken künstlich verschlechtern. Das Safe-Zentrum ist der einzige Raumbereich, für den diese Kachel exklusiv verantwortlich war.

Für die methodisch saubere P/R-Evaluation in QGIS gilt dieselbe AOI-Semantik wie für die Vorhersagen: Referenz-Features (z. B. ALKIS-Gebäude) werden zuerst auf dieselbe Nutzer-AOI geclippt und danach werden diejenigen Referenzteile exkludiert, die `nodata_regions` schneiden. So wird nicht ein ungeclipptes Referenzobjekt mit einem geclippten Vorhersageobjekt verglichen.

**AOI-Clip vor user-sichtbarer Ausgabe ist zwingend:** Weil 5.1 die Download-BBox absichtlich über die Nutzer-AOI hinaus expandiert, dürfen Export und UI diese Hilfszone **nicht** unverändert zurückgeben. Die AOI-Semantik des Produkts lautet dabei explizit: **Die Nutzer-BBox ist ein räumliches Clip-Window, kein Center-Ownership-Filter.** Ein Objekt wird also im Output berücksichtigt, wenn seine Geometrie die AOI schneidet; im Export und in GeoJSON erscheint dann nur der innerhalb der AOI liegende Geometrieteil. Vor GeoPackage-Export und vor den GeoJSON-Antworten werden `detected_gdf` und `nodata_gdf` deshalb räumlich auf `jobs.bbox_utm_snapped` reduziert. Für Polygone an der AOI-Kante wird die Geometrie mit der AOI geschnitten (`intersection(requested_bbox_polygon)`), leere Resultate werden verworfen. **Wichtig:** Der Clip kann aus einem einzelnen Polygon wieder ein `MultiPolygon` oder eine `GeometryCollection` machen. Deshalb läuft **nach dem AOI-Clip erneut** derselbe Polygon-Only-Normalisierungspfad wie in 5.4 (`make_valid` + `extract_polygons` + `explode`), bevor die Geometrien in Polygon-Layer oder GeoJSON serialisiert werden. Gleiches gilt für `nodata_regions`, damit weder Detections noch NoData außerhalb des angefragten Gebiets im Output erscheinen.

**Re-Export nach Re-Validation:** Ein Nutzer kann nach dem ersten Export False Positives entdecken, sie auf REJECTED setzen und erneut exportieren. Fiona/GeoPandas-Verhalten bei existierenden `.gpkg`-Dateien ist uneinheitlich (Append in Duplikate **oder** `ValueError: Layer already exists`). Deshalb **zwingend** vor dem Schreiben:

```python
out_path.unlink(missing_ok=True)   # alten Export löschen, GeoPandas schreibt frisch
```

**Leere Ergebnis-Layer sind ein First-Class-Fall:** Wenn ein Prompt keine Treffer liefert oder der Nutzer alles auf `REJECTED` setzt, ist `detected_gdf` leer. Das Export-Verhalten darf dann **nicht** vom impliziten Schema-Inferenzpfad von GeoPandas/Fiona abhängen. Der Exporter muss beide Layer mit **explizitem Schema** anlegen, auch bei `0` Features:

```python
DETECTED_SCHEMA = {
    "geometry": "Polygon",
    "properties": {
        "score": "float",
        "source_tile_row": "int",
        "source_tile_col": "int",
    },
}
NODATA_SCHEMA = {
    "geometry": "Polygon",
    "properties": {"reason": "str:32"},
}
```

Für nicht-leere GeoDataFrames kann GeoPandas weiter genutzt werden; für leere Layer muss der Exportpfad das Schema explizit setzen und trotzdem eine gültige `.gpkg`-Tabelle erzeugen. Der Contract lautet: **`POST /jobs/{id}/export` erzeugt immer ein lesbares GeoPackage mit beiden Layern**, auch wenn `detected_objects` leer ist.

**CRS bei leeren GeoDataFrames explizit erzwingen:** Das explizite Schema allein reicht nicht; der leere Layer muss bereits als GeoDataFrame mit gesetztem CRS erzeugt werden, bevor er an Fiona/GeoPandas übergeben wird, z. B.:

```python
empty_detected = gpd.GeoDataFrame(
    {"score": [], "source_tile_row": [], "source_tile_col": []},
    geometry=[],
    crs="EPSG:25832",
)
```

Ohne diesen Schritt kann ein formal valides GeoPackage entstehen, dessen leerer Layer in QGIS als `Unknown CRS` auftaucht.

**Signatur:**
```python
def export_two_layer_gpkg(
    detected_gdf: gpd.GeoDataFrame,
    nodata_gdf: gpd.GeoDataFrame,
    requested_bbox: BaseGeometry,
    out_path: Path,
) -> None:
    """
    Überschreibt out_path unconditional. Beide Layer müssen CRS EPSG:25832 haben.
    detected_gdf wird auf validation=='ACCEPTED' vorgefiltert vom Caller.
    Beide Layer werden vor dem Schreiben auf requested_bbox geclippt.
    """
```

## 6. Job-Zustandsautomat

```
    ┌─ Worker-Startup-Hook räumt ab ─┐
    │                                │
[POST /jobs]                         │
    │                                │
    ▼                                │
┌─────────┐    ┌─────────────┐    ┌──────────────────────┐
│ PENDING │───▶│ DOWNLOADING │───▶│     INFERRING        │
└─────────┘    └─────────────┘    │  (Tiling + SAM 3.1)  │
                                  └──────────┬───────────┘
                                             │
                                             ▼
                                   ┌────────────────────┐
                                   │ READY_FOR_REVIEW   │
                                   └──────────┬─────────┘
                                              │ first [POST /jobs/{id}/export]
                                              ▼
                                      ┌──────────────┐
                                      │  EXPORTED    │
                                      └──────┬───────┘
                                             │ repeat export / re-validation allowed
                                             └───────────────────────┐

                 Jeder Zustand kann bei Fehler → FAILED (+ error_reason, error_message).
                 FAILED ist Endzustand. `EXPORTED` bleibt reviewbar; ein Retry erfolgt nur durch Erzeugung eines neuen Jobs.
```

- **Nur der Worker** verändert GPU-relevante Zustände. Der Webserver darf drei review-seitige Zustandsänderungen vornehmen: `PENDING` beim `POST /jobs`, `READY_FOR_REVIEW → EXPORTED` beim ersten erfolgreichen `POST /jobs/{id}/export`, und idempotentes `EXPORTED → EXPORTED` bei Folge-Exporten.
- **Kein separater `TILING`-Zustand:** Tiling und Inferenz laufen im selben Iterator-Loop (`for tile in iter_tiles(): segmenter.predict(tile, prompt)`).
- Fortschritts-Feedback in der UI kommt aus `jobs.tile_completed / jobs.tile_total`.
- **Crash-Recovery:** Worker markiert beim Start alle Jobs in `DOWNLOADING` oder `INFERRING` als `FAILED` mit `error_reason='WORKER_RESTARTED'`.
- **Kein In-place-Retry:** Fehlerhafte Jobs bleiben `FAILED`. Nutzer muss neuen Job einreichen (FR-6). Review und Re-Export eines erfolgreichen Jobs bleiben dagegen erlaubt.

## 7. SQLite-Schema

Bei jeder neuen Connection:
```sql
PRAGMA journal_mode = WAL;
PRAGMA busy_timeout = 5000;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
```

Drei Tabellen:

```sql
CREATE TABLE jobs (
    id                TEXT PRIMARY KEY,                 -- UUID v4
    prompt            TEXT NOT NULL,
    bbox_wgs84        TEXT NOT NULL,                    -- JSON [minx,miny,maxx,maxy]
    bbox_utm_snapped  TEXT NOT NULL,                    -- JSON, grid-aligned 0.2 m
    tile_preset       TEXT NOT NULL CHECK (tile_preset IN ('small', 'medium', 'large')),
    status            TEXT NOT NULL CHECK (status IN (
                          'PENDING', 'DOWNLOADING', 'INFERRING',
                          'READY_FOR_REVIEW', 'EXPORTED', 'FAILED')),
    error_reason      TEXT CHECK (error_reason IS NULL OR error_reason IN (
                          'WCS_TIMEOUT', 'WCS_HTTP_ERROR', 'OOM',
                          'INFERENCE_ERROR', 'WORKER_RESTARTED',
                          'EXPORT_ERROR', 'INVALID_GEOMETRY')),
    error_message     TEXT,                             -- Stacktrace-Tail (letzte ~20 Zeilen)
    dop_vrt_path      TEXT,                             -- nach Download gesetzt
    gpkg_path         TEXT,                             -- nach Export gesetzt
    tile_total        INTEGER,
    tile_completed    INTEGER NOT NULL DEFAULT 0,
    tile_failed       INTEGER NOT NULL DEFAULT 0,
    validation_revision INTEGER NOT NULL DEFAULT 0,     -- erhöht sich bei jeder erfolgreichen validate_bulk-Tx
    exported_revision  INTEGER,                         -- Validation-Revision, die in gpkg_path materialisiert ist; NULL bis zum ersten Export
    created_at        TEXT NOT NULL,                    -- ISO-8601 UTC
    started_at        TEXT,
    finished_at       TEXT
);
CREATE INDEX idx_jobs_status_created ON jobs(status, created_at);

CREATE TABLE polygons (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id           TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    geometry_wkb     BLOB NOT NULL,                     -- EPSG:25832, shapely.wkb.dumps
    score            REAL NOT NULL,
    source_tile_row  INTEGER NOT NULL,
    source_tile_col  INTEGER NOT NULL,
    validation       TEXT NOT NULL DEFAULT 'ACCEPTED' CHECK (validation IN ('ACCEPTED', 'REJECTED'))
);
CREATE INDEX idx_polygons_job ON polygons(job_id);

CREATE TABLE nodata_regions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id           TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    geometry_wkb     BLOB NOT NULL,                     -- Safe-Zentrum-Footprint (EPSG:25832), NICHT Tile-Footprint
    tile_row         INTEGER NOT NULL,
    tile_col         INTEGER NOT NULL,
    reason           TEXT NOT NULL CHECK (reason IN (
                          'OOM', 'INFERENCE_ERROR', 'INVALID_GEOMETRY', 'NODATA_PIXELS'))
);
CREATE INDEX idx_nodata_job ON nodata_regions(job_id);
```

**Begründung WKB statt WKT:** `rasterio.features.shapes` erzeugt Pixel-Staircase-Polygone mit tausenden Vertices. WKB ist ~3× kompakter und ~10× schneller zu serialisieren als WKT.

**Disk-Hygiene (NFR-6):** Beim Status-Übergang `READY_FOR_REVIEW → EXPORTED` werden `data/dop/{job_id}/` (Chunks + VRT) rekursiv gelöscht. Polygone sind bereits in SQLite persistiert, das GeoPackage unter `data/results/{job_id}.gpkg`. Auch beim Übergang `→ FAILED` werden die Download-Artefakte aufgeräumt.

## 8. HTTP-Endpunkte (FastAPI)

Alle Request/Response-Bodies sind JSON. Koordinaten-BBoxes einheitlich als `[minx, miny, maxx, maxy]`-Array.

| Methode | Pfad | Body (Request) | Response | Zweck |
|---|---|---|---|---|
| `GET` | `/` | — | HTML | Einzige Webseite (Leaflet + Job-Liste) |
| `POST` | `/jobs` | `{prompt: str, bbox_wgs84: [float, float, float, float], tile_preset?: "small" \| "medium" \| "large"}` | `{id: uuid, status: "PENDING"}` | Neuen Job anlegen (NFR-4). `tile_preset` Default `medium` |
| `GET` | `/jobs` | — | `[{id, prompt, status, created_at, tile_completed, tile_total, validation_revision, exported_revision, export_stale, ...}]` | Job-Historie |
| `GET` | `/jobs/{id}` | — | `{id, prompt, status, error_reason, error_message, tile_completed, tile_total, validation_revision, exported_revision, export_stale, ...}` | Status-Polling + Review-UI-Metadaten |
| `GET` | `/jobs/{id}/polygons` | — | GeoJSON `FeatureCollection` (EPSG:4326, auf Nutzer-AOI geclippt); Feature-Properties: `id`, `score`, `validation` | Für Leaflet-Darstellung |
| `GET` | `/jobs/{id}/nodata` | — | GeoJSON `FeatureCollection` (EPSG:4326, auf Nutzer-AOI geclippt); Feature-Properties: `reason` | Für Leaflet-Darstellung (transparent-gestreift o. ä.) |
| `POST` | `/jobs/{id}/polygons/validate_bulk` | `{updates: [{pid: int, validation: "ACCEPTED" \| "REJECTED"}, ...]}` | `{ok: true, updated: int}` | Bulk-Validierung (FR-3); siehe 9.2 warum kein Per-Polygon-Endpoint |
| `POST` | `/jobs/{id}/export` | — | `{gpkg_path: str}` | Erzeugt oder überschreibt GeoPackage aus allen `ACCEPTED`-Polygonen; erlaubt wenn `status ∈ {"READY_FOR_REVIEW", "EXPORTED"}` |
| `GET` | `/jobs/{id}/export.gpkg` | — | Binär (`application/geopackage+sqlite3`) | Download |

**Validierungsregeln Server-seitig:**
- `POST /jobs`: `bbox_wgs84` muss valide Bounds haben (`minx < maxx`, `miny < maxy`), und die Flächenprüfung für `MAX_BBOX_AREA_KM2 = 1.0` erfolgt **nicht in WGS84-Grad**, sondern auf der **unexpandierten User-BBox nach Transformation nach EPSG:25832** (bzw. äquivalent geodätisch). Nur so entspricht „1 km²“ tatsächlich dem Browser-/Runtime-Limit. Prompt non-empty nach `strip()`, Länge ≤ `MAX_PROMPT_CHARS`, **und** die vom echten Modell-Encode-Pfad erzeugte finale Textsequenz muss in `MAX_ENCODER_CONTEXT_TOKENS` passen. Char-Limits sind nur ein grober UX-Guard; maßgeblich ist die **vollständig templatisierte** Encoder-Sequenz inklusive Spezialtokens. Bei Überschreitung → HTTP 422 mit klarer Fehlermeldung („Prompt zu lang für Modellkontext; bitte kurze Nomenphrase verwenden“). `tile_preset` muss in `{"small", "medium", "large"}` liegen, fehlend ⇒ `medium`.
- **Geographic Fence (Bayern):** Die gesamte BBox muss innerhalb WGS84-Grenzen `lon ∈ [8.9, 13.9]`, `lat ∈ [47.2, 50.6]` liegen (`BAYERN_BBOX_WGS84` in config). Verletzung ⇒ HTTP 422 `Unprocessable Entity`. Ohne diesen Check würde eine BBox über Berlin oder dem Atlantik als PENDING akzeptiert, der Worker produzierte FAILED-Jobs mit nichtssagenden WCS-Fehlern.
- `POST /jobs/{id}/export`: 409 Conflict, wenn Status weder `READY_FOR_REVIEW` noch `EXPORTED` ist. 404, wenn Job nicht existiert. **Der Export-Aufruf selbst ist durch ein modul-weites `threading.Lock` serialisiert** — GDAL/Fiona sind nicht thread-safe, zwei parallele GeoPackage-Writes (z. B. bei Doppelklick) würden Uvicorn per Segfault reißen. Der Lock wrappt den `export_two_layer_gpkg`-Aufruf; wartende Requests antworten nach ihrem Abschluss regulär, es gibt kein Queue-Backlog weil Export-Zeit im Sekundenbereich liegt. Nach erfolgreichem Schreiben gilt: `status='EXPORTED'` und `exported_revision = validation_revision`. Export und GeoJSON-Endpunkte geben ausschließlich Geometrien innerhalb von `bbox_utm_snapped` zurück.
- `POST /jobs/{id}/polygons/validate_bulk`: 409 Conflict für den ganzen Bulk-Call, wenn Job-Status nicht reviewbar ist (`PENDING`, `DOWNLOADING`, `INFERRING`, `FAILED`). `READY_FOR_REVIEW` **und** `EXPORTED` sind gültige Zielzustände. Implementierung **zwingend mit `cursor.executemany()`**, nicht mit manuellem `VALUES (...)`-Query-Building: letzteres sprengt bei hunderten Updates das `SQLITE_MAX_VARIABLE_NUMBER`-Limit (999 bzw. 32 766 je nach Kompilierung) und wirft `OperationalError`. `executemany` stückelt intern sicher und behandelt alles in einer Transaktion. Unbekannte PIDs werden ignoriert, die Response zählt nur erfolgreich aktualisierte (via `cursor.rowcount` nach Commit). In derselben Transaktion wird `jobs.validation_revision = validation_revision + 1` erhöht, damit der GeoJSON-Cache deterministisch invalidiert wird; `exported_revision` bleibt unverändert und markiert einen evtl. stale Export.

    ```python
    # korrekt:
    with conn:
        cursor.executemany(
            "UPDATE polygons SET validation = ? WHERE id = ? AND job_id = ?",
            [(u.validation, u.pid, job_id) for u in updates]
        )
        updated = cursor.rowcount
        cursor.execute(
            "UPDATE jobs SET validation_revision = validation_revision + 1 WHERE id = ?",
            (job_id,),
        )
    ```

**CPU-Lastige Endpunkte (Event-Loop- und Threadpool-Schutz):** `/jobs/{id}/polygons` und `/jobs/{id}/nodata` deserialisieren bis zu 10 000 WKB-Blobs, transformieren sie nach EPSG:4326 und serialisieren nach GeoJSON — mehrere Sekunden CPU-Zeit. Weder blockierendes `async def` **noch** Auslagerung in den globalen FastAPI/AnyIO-Threadpool sind dafür die richtige Zielarchitektur: Ersteres friert den Event-Loop ein, Letzteres kann unter Burst-Last den gemeinsamen Threadpool erschöpfen. Der Spec verlangt daher einen **dedizierten Serialisierungs-Executor** (`ProcessPoolExecutor`, klein begrenzt, z. B. 2 Worker) oder äquivalent `anyio.to_process.run_sync(...)`. Die HTTP-Route selbst bleibt leichtgewichtig; die schwere GeoJSON-Erzeugung läuft außerhalb von Event-Loop **und** Web-Threadpool.

**GeoJSON-Cache:** Da dieselben Polygone häufig mehrfach geladen werden (Job erneut öffnen, Browser-Refresh), wird die serialisierte `FeatureCollection` pro `(job_id, validation_revision, target='polygons'|'nodata')` gecacht. Jede erfolgreiche `validate_bulk`-Transaktion erhöht `validation_revision` und invalidiert nur den Polygon-Cache des Jobs.

**Payload-Reduktion via Coordinate-Precision:** Nach der EPSG:25832 → EPSG:4326-Transformation liefert pyproj Doubles mit 15 Nachkommastellen (Nanometer-Genauigkeit — völlig absurd bei 20-cm-Pixeln). Bei 3 000 Gebäuden × durchschnittlich ~50 Vertices ergibt das 20 MB+ GeoJSON. **Zwingend** vor der Serialisierung:

```python
from shapely import set_precision
geom_4326 = set_precision(geom_4326, grid_size=1e-6)   # ~11 cm am Äquator
```

Reduziert die Payload um 60–70 %, ohne Genauigkeit zu verlieren, die am Display jemals sichtbar wäre (1e-6° ≈ 11 cm bei Bayern-Breite).

## 9. Frontend (Leaflet + Vanilla JS)

Eine einzige HTML-Seite mit drei UI-Bereichen:

1. **Hauptkarte** (Leaflet) — Basemap: OSM, optional DOP20-WMS-Overlay zur visuellen Kontrolle. `leaflet-draw` für BBox-Selektion.
2. **Prompt-Panel** — Textfeld für **kurze Nomenphrasen** + Tile-Preset-Dropdown (Default `medium`, mit Tooltip-Hinweis auf Max-Objektgröße 64/128/192 m) + Start-Button. Zeigt nach `POST /jobs` nur „Job {id} läuft — Status: {status}".
3. **Job-Historie** — Liste vergangener Jobs mit Status-Badge. Klick lädt Polygone auf die Karte.

Beim Laden eines `READY_FOR_REVIEW`- oder `EXPORTED`-Jobs werden Polygone als Leaflet-Layer angezeigt; Klick auf ein Polygon toggelt lokal den Validation-Zustand (✓/✗-Styling), die Änderungen werden **lokal gepuffert** und gebündelt an `POST /jobs/{id}/polygons/validate_bulk` gesendet (Details 9.2). Der Webserver liefert dafür `validation_revision`, `exported_revision` und das abgeleitete Bool `export_stale = exported_revision IS NULL OR exported_revision < validation_revision`; bei `export_stale=true` zeigt die UI zusätzlich „Export veraltet — erneut exportieren“. NoData-Regionen werden als schraffierter Layer überlagert, damit der Nutzer erkennt, wo keine Auswertung stattfand.

Polling: Wenn Job im aktiven Zustand (`DOWNLOADING` / `INFERRING`), alle 3 s `GET /jobs/{id}` abfragen, bis Status terminal.

### 9.1 Rendering-Performance und BBox-Limit

Beim Prompt „building" in urbanem Bayern fallen pro km² typisch 2 000–3 000 Gebäude an. Das sind auch die Grenzen dessen, was Leaflet mit L.geoJSON + Canvas interaktiv (inkl. Hover/Click-Hit-Testing) flüssig verarbeitet. Canvas löst die DOM-Explosion von SVG, aber `L.geoJSON()` hält weiterhin ein JS-Objekt pro Feature mit Event-Listenern — bei ~10 k Polygonen blockieren Hit-Tests bei Mausbewegungen den Main-Thread. Deshalb **zwei verbindliche Maßnahmen**:

1. **`MAX_BBOX_AREA_KM2 = 1.0`** (siehe Section 12) — begrenzt die Flächengröße serverseitig auf 1×1 km. Das hält die Polygonzahl auch bei dichten Prompts unter 3 k und bleibt für den Browser handhabbar. Mehrere Gebiete werden als einzelne Jobs verarbeitet. Iterations-Zeit auf der RTX 4070 (grobe Richtwerte): small-Preset ~2 min, medium ~5–10 min, large ~30+ min.
2. **Canvas-Renderer erzwungen:** `L.map('map', {renderer: L.canvas()})`. Für spätere Skalierung wäre `geojson-vt` (Client-seitiges Vector-Tiling) der nächste Schritt — aber für den Prototyp nicht nötig, solange das 1-km²-Limit hält.

### 9.2 Client-seitiges Validation-Debouncing

Der Endpoint heißt bewusst `validate_bulk`, nicht per-Polygon. Grund: Schnelle Klicks (User validiert 20 Polygone in 2 s) würden 20 parallele `POST /...validate/{pid}` auslösen. Da SQLite im WAL-Modus nur einen Writer gleichzeitig erlaubt und parallel dazu der Worker committet, können mehrere dieser UPDATEs in den 5-s-`busy_timeout` laufen und mit `database is locked` fehlschlagen — der Nutzer verliert Validierungsdaten stillschweigend.

**Pattern im Frontend (Vanilla JS):**

```javascript
const pendingUpdates = new Map();   // pid → validation
let flushTimer = null;
const MAX_CLIENT_BUFFER_UPDATES = 100;
const STORAGE_KEY = `job:${jobId}:pending-validations`;

function snapshotUpdates() {
    return [...pendingUpdates].map(([pid, validation]) => ({pid, validation}));
}

function persistPendingToStorage() {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(snapshotUpdates()));
}

function clearPendingFromStorage() {
    sessionStorage.removeItem(STORAGE_KEY);
}

function hydratePendingFromStorage() {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    for (const {pid, validation} of JSON.parse(raw)) {
        pendingUpdates.set(pid, validation);
    }
}

function queueValidation(pid, validation) {
    pendingUpdates.set(pid, validation);
    persistPendingToStorage();
    if (pendingUpdates.size >= MAX_CLIENT_BUFFER_UPDATES) {
        void flushValidations();
        return;
    }
    if (flushTimer) clearTimeout(flushTimer);
    flushTimer = setTimeout(() => void flushValidations(), 3000);  // 3s Debounce
}

async function flushValidations() {
    if (!pendingUpdates.size) return;
    const updates = snapshotUpdates();
    pendingUpdates.clear();
    await fetch(`/jobs/${jobId}/polygons/validate_bulk`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({updates}),
    });
    clearPendingFromStorage();
}

hydratePendingFromStorage();
if (pendingUpdates.size) {
    void flushValidations();
}

window.addEventListener('pagehide', () => {
    if (!pendingUpdates.size) return;
    persistPendingToStorage();  // Recovery-Pfad, falls der Tab sofort weg ist
    const blob = new Blob(
        [JSON.stringify({updates: snapshotUpdates().slice(0, MAX_CLIENT_BUFFER_UPDATES)})],
        {type: 'application/json'}
    );
    navigator.sendBeacon(`/jobs/${jobId}/polygons/validate_bulk`, blob);
});
```

`pagehide`/`sendBeacon()` dient hier nur als Best-Effort für den **kleinen Restpuffer**. Der eigentliche Zuverlässigkeitsmechanismus ist: häufiges Debounce-Flush, harte Obergrenze `MAX_CLIENT_BUFFER_UPDATES`, und `sessionStorage`-Recovery beim nächsten Laden des Jobs. Große ungesendete Batches dürfen **nicht** erst beim Tab-Close entstehen. Der oft genannte „CORS/Preflight-Fallstrick“ greift in dieser Architektur **nicht**, weil UI und API same-origin vom selben FastAPI-Prozess ausgeliefert werden. Falls die UI später auf eine andere Origin ausgelagert wird, darf der Beacon-Pfad nicht unverändert übernommen werden; dann ist entweder ein same-origin Proxy oder ein anderer Close-Path nötig.

## 10. Fehlerbehandlung

**Drei Schutzebenen:**

1. **Pro Tile** — `Sam3Segmenter.predict()` in try/except. Bei `torch.cuda.OutOfMemoryError`: **Safe-Zentrum** der Kachel (nicht Tile-Footprint) in `nodata_regions` mit `reason='OOM'`, `empty_cache()`, Loop fährt fort. Bei anderen Python-Exceptions: gleiche Logik mit `reason='INFERENCE_ERROR'`. Safe-Zentrum statt Tile-Footprint, damit ALKIS-Features in den überlappenden Safe-Zentren der Nachbarn nicht fälschlich ausgeschlossen werden (siehe 5.5).
2. **Pro Job** — `worker/orchestrator.py` wrappt alle Pipeline-Aufrufe in globalem try/except. Jeder ungefangene Fehler → `status=FAILED`, `error_reason` aus bekanntem Enum setzen, `error_message` mit letzten Traceback-Zeilen befüllen.
3. **Pro Worker** — Startup-Hook in `worker/loop.py`, in dieser Reihenfolge:
   1. `UPDATE jobs SET status='FAILED', error_reason='WORKER_RESTARTED' WHERE status IN ('DOWNLOADING', 'INFERRING')`.
   2. **Disk-Zombie-Cleanup:** scanne `data/dop/` rekursiv. Für jedes Verzeichnis `data/dop/{job_id}/`: wenn der zugehörige Job in SQLite **nicht mehr** im Status `DOWNLOADING` oder `INFERRING` steht (inkl. nicht-existent), via `shutil.rmtree` rigoros löschen. Das fängt Fälle ab, in denen der Worker via SIGKILL (OOM-Killer, Stromausfall) beendet wurde und der Status-Übergang-basierte Cleanup aus Section 7 nicht lief.
   3. Danach: normales Polling.

   Fatale OS-Signale (SIGSEGV/SIGKILL) werden nicht abgefangen — der Supervisor startet den Worker neu, der Startup-Hook räumt DB **und** Disk auf.

**Write-Transaktions-Scope (SQLite-Lock-Vermeidung):** Der Worker darf **niemals** eine Write-Transaktion über die gesamte Job-Dauer halten. Ein medium-Preset-Job läuft 15–25 min — würde die Transaktion so lange offen bleiben, liefe jeder `POST /jobs`-Insert des Webservers in den 5-s-`busy_timeout` und schlüge mit `database is locked` fehl. Regel: **pro Kachel** öffnet der Orchestrator eine Connection, persistiert Polygone/NoData-Region dieser Kachel, committet sofort, schließt die Connection. Fortschritts-Updates (`tile_completed++`) sind ebenfalls separate Transaktionen. Long-Read-Connections des Webservers (die GeoJSON-Endpunkte) sind im WAL-Modus unkritisch, solange keine parallele Write-Tx offen ist.

**Restart-Strategie:** Worker terminiert nach `MAX_JOBS_PER_WORKER = 50` bzw. nach 24 h Laufzeit freiwillig mit Exit-Code 0. Supervisor (`scripts/run-worker.sh`) startet neu. Damit wird VRAM-Fragmentierung physisch geflusht.

## 11. Test-Strategie

Fokus auf `pipeline/`-Module (reine Funktionen, gut isolierbar). Keine Tests für SAM 3.1 selbst — dem Modell wird vertraut.

| Modul | Test-Typ | Was wird geprüft |
|---|---|---|
| `wcs_client.py` | Unit mit `responses`-Mock | Grid-Snapping (inkl. Chunk-Alignment), `transform_bounds`-Verwendung, Margin-Expansion (jedes Preset expandiert um korrekten CENTER_MARGIN × 0.2 m in alle 4 Richtungen), Mikro-BBox-Expansion (<204.8 m → genau 204.8 m), **origin-sensitives Snapping** (verifiziert mit abweichendem Origin, z. B. `origin_x=0.1`), WCS-Pixel-Dimension via `round()` (verifiziert mit `300.0/0.2`-Edge-Case → 1500, nicht 1499), **nahtlose Chunk-Kanten (`chunk_{n+1}.minx == chunk_n.maxx`) ohne NoData-Lücke und ohne Doppelreihe im fertigen VRT**, Pagination bei >4000 px, Timeout-Handling, Retry-Logik, `WCSError` bei Dauer-5xx |
| `tiler.py` | Unit mit synthetischem GeoTIFF (Fixture) | Korrekte Tile-Grenzen für alle drei Presets, `src.window_transform(window)`-basierte Per-Tile-Affine liefert korrekte UTM-Koordinaten (verifiziert via bekanntem Punkt), Overlap korrekt, Edge-Tile-Verschiebung am VRT-Rand, **letzte verschobene Rand-Kachel behält eindeutige logische `tile_row`/`tile_col`**, `TileConfig.from_preset()`, NoData-Tile-Erkennung basiert auf **Raster-Masken / NoData-Tags im Safe-Zentrum** (NoData im Safe-Zentrum → `reason='NODATA_PIXELS'`, NoData nur am Tile-Rand → Tile bleibt zulässig), **RGB-schwarze Pixel ohne NoData-Maske markieren kein Tile als NoData**, **Fixture mit 4-Band-Alpha-GeoTIFF → Tiler liest nur Band 1–3, Array-Shape korrekt `(H,W,3)` und NoData-Maske Shape `(H,W)`** |
| `segmenter.py` | Integration mit Stub/Snapshot-Outputs | Tile-lokale Deduplikation: score-sortierte Rohmasken werden bei hoher IoU oder starker Verschachtelung unterdrückt; unterschiedliche, räumlich getrennte Objekte bleiben erhalten |
| `merger.py` | Unit mit synthetischen Masken | BBox-Center-Keep (inkl. Grenzfall Mittelpunkt genau auf Safe-Center-Kante, halboffenes Intervall), C-/L-förmige Masken mit BBox-Center außerhalb der Maske, `rasterio.features.shapes`-Output `(geom_dict, value)` wird korrekt via `shapely.geometry.shape()` konvertiert, **`connectivity=8` erhält diagonal verbundene Maskenteile**, Hintergrund wird nicht als Polygon persistiert, `make_valid` repariert Bowtie, `MultiPolygon`-Zerlegung, **Diagonal-Berühr-Maske erzeugt GeometryCollection → LineString/Point werden via `extract_polygons` verworfen**, Flächenfilter |
| `jobs/store.py` (Transaktionen) | Integration mit File-SQLite | Paralleler Writer+Reader über separate Connections bleibt lock-frei (WAL), Pro-Kachel-Commit verursacht keinen Deadlock |
| `worker/loop.py` (Startup) | Integration, Fixture-Verzeichnisse | Zombie-Dirs in `data/dop/` werden beim Start gelöscht, wenn der zugehörige Job nicht mehr aktiv ist; aktive Jobs bleiben unangetastet |
| `app/routes/jobs.py` (Fence) | Unit mit TestClient | BBox außerhalb `BAYERN_BBOX_WGS84` → HTTP 422; BBox exakt am Grenzwert → akzeptiert; BBox-Fläche wird in EPSG:25832 statt naiv in WGS84-Grad geprüft; Prompt, dessen **final encodierte** Textsequenz `MAX_ENCODER_CONTEXT_TOKENS` überschreitet, → HTTP 422; `validate_bulk` nutzt `executemany` (verifiziert durch 1 500-Updates-Test ohne `OperationalError`), ignoriert unbekannte PIDs, ist für `READY_FOR_REVIEW` und `EXPORTED` erlaubt, erhöht `validation_revision`; Export-Endpoint ist per `threading.Lock` serialisiert (verifiziert durch paralleles Double-POST im TestClient), erlaubt Re-Export aus `EXPORTED` und setzt `exported_revision = validation_revision`; `GET /jobs` und `GET /jobs/{id}` liefern `validation_revision`, `exported_revision`, `export_stale`; **GeoJSON- und Export-Geometrien sind als Clip-Window-Semantik auf `bbox_utm_snapped` geclippt**; **GeoJSON-Response hat Precision ≤ 1e-6° (verifiziert: keine Koordinate mit >6 Nachkommastellen)**; CPU-schwere GeoJSON-Erzeugung wird in dediziertem Executor statt Web-Threadpool ausgeführt |
| Retention-Cleanup | Integration mit File-SQLite | `RETENTION_DAYS`-Cutoff löscht alte FAILED/EXPORTED-Jobs inkl. Geometrien (CASCADE) und `.gpkg`-Dateien, aktive Jobs bleiben erhalten, `VACUUM` wird ausgeführt |
| `pipeline/geo_utils.py` (Snapping) | Unit mit Edge-Case-Koordinaten | `snap_floor(0.6)==0.6`, `snap_floor(1.3)==1.2`, UTM-Range (600000.15 → 600000.0 / 600000.2) — verifiziert FP-Robustheit |
| `exporter.py` | Unit, Temp-Verzeichnis | Zwei Layer vorhanden, CRS = EPSG:25832, GeoPackage lesbar mit `fiona`, `nodata_regions`-Geometrie entspricht Safe-Zentrum (nicht Tile-Footprint), **AOI-Clip auf `bbox_utm_snapped` folgt explizit der Clip-Window-Semantik** (schneidende Objekte bleiben als geclipptes Teilobjekt erhalten), **Re-Export überschreibt existierende .gpkg-Datei ohne Duplikate / ValueError**, leeres `detected_gdf` erzeugt trotzdem gültigen `detected_objects`-Layer mit Polygon-Schema **und explizitem CRS EPSG:25832** |
| `jobs/store.py` | Unit mit temp File-SQLite | CHECK-Constraint-Violations erkannt, Status-Übergänge, WKB-Roundtrip, PRAGMA-Initialisierung; WAL wird nur auf dateibasierter SQLite verifiziert, nicht auf `:memory:` |
| End-to-End | Integration, Stub-SAM | Mini-GeoTIFF (256×256) als Fixture, Mock-WCS-Response, Stub-Segmenter gibt feste Masken zurück, vollständiger Pipeline-Lauf bis `EXPORTED`, Prüfung der zwei Layer im resultierenden .gpkg |

## 12. Konfiguration

`config.py` als zentrale Quelle. Überschreibbar über `.env` (pydantic-settings).

```python
# WCS — beim ersten GetCapabilities-Call verifizieren
WCS_URL: str              = "PLACEHOLDER_UNTIL_VERIFIED"   # z.B. https://geoservices.bayern.de/wcs/v2/dop20
WCS_COVERAGE_ID: str      = "PLACEHOLDER_UNTIL_VERIFIED"
WCS_MAX_PIXELS: int       = 4000
WCS_GRID_ORIGIN_X: float  = 0.0   # nach GetCapabilities verifizieren
WCS_GRID_ORIGIN_Y: float  = 0.0   # nach GetCapabilities verifizieren

# SAM 3.1
SAM3_CHECKPOINT: Path     = Path("models/sam3.1_hiera_large.pt")

# Tiling — Presets werden per Request gewählt (siehe 5.2, 8)
TILE_SIZE: int            = 1024
DEFAULT_TILE_PRESET: str  = "medium"    # small | medium | large

# Filtering
MIN_POLYGON_AREA_M2: float = 1.0
LOCAL_MASK_NMS_IOU: float  = 0.6
LOCAL_MASK_CONTAINMENT_RATIO: float = 0.9
SAFE_CENTER_NODATA_THRESHOLD: float = 0.0

# Worker
MAX_JOBS_PER_WORKER: int   = 50
WORKER_POLL_INTERVAL_SEC: float = 2.0

# API limits
MAX_BBOX_AREA_KM2: float     = 1.0   # = 1 km × 1 km; Leaflet+L.geoJSON schafft nur ~2–3k Polygone interaktiv (9.1)
MAX_PROMPT_CHARS: int        = 240   # grober UX-Guard; harte Grenze ist MAX_ENCODER_CONTEXT_TOKENS
MAX_ENCODER_CONTEXT_TOKENS: int = 77 # Prüfung auf finaler, templatisierter Encoder-Sequenz inkl. Spezialtokens
MAX_CLIENT_BUFFER_UPDATES: int = 100

# Geographic Fence — WGS84-Bounds von Bayern (grob)
BAYERN_BBOX_WGS84: tuple   = (8.9, 47.2, 13.9, 50.6)   # (lon_min, lat_min, lon_max, lat_max)

# Retention (Disk/DB-Hygiene)
RETENTION_DAYS: int          = 7     # FAILED/EXPORTED Jobs inkl. Geometrien nach N Tagen löschen
```

**Uvicorn Dev-Setup:** Uvicorn wird lokal mit `--reload` genutzt, um Code-Änderungen sofort zu sehen. SQLite im WAL-Modus erzeugt jedoch kontinuierlich `jobs.db-wal`/`jobs.db-shm`-Dateien — der File-Watcher würde den Server in einer Endlosschleife neu starten. Zwingend:

```bash
uvicorn ki_geodaten.app.main:app --reload \
    --reload-dir ki_geodaten \
    --reload-exclude 'data/*' \
    --reload-exclude '*.db*'
```

Die `scripts/run-server.sh` muss diese Flags setzen.

## 13. Abhängigkeiten

Python 3.12+, PyTorch 2.7+, CUDA 12.6+ (Voraussetzungen für SAM 3.1).

```
# ML
torch>=2.7
sam3                    # facebookresearch/sam3
# Geo
rasterio>=1.3
geopandas>=1.0
shapely>=2.0
pyproj>=3.6
gdal                    # via conda empfohlen, für BuildVRT
# Web
fastapi
uvicorn[standard]
jinja2
# Data
pydantic>=2
pydantic-settings
# HTTP
requests
urllib3>=2              # für Retry
# Testing
pytest
pytest-mock
responses               # für wcs_client-Mocks
```

## 14. Bekannte Einschränkungen (bewusst akzeptiert)

- **Nur Bayern:** `wcs_client` ist auf LDBV-WCS zugeschnitten. Erweiterung auf andere Bundesländer: neue `WcsClient`-Implementierung + Strategiewahl anhand BBox.
- **Ein Nutzer gleichzeitig:** Die UI hat kein Auth-Konzept, Jobs sind für alle sichtbar. Für lokale Einzelnutzung genügt das.
- **Keine Job-Stornierung:** Ein laufender Job kann nicht abgebrochen werden. Bei Bedarf: Worker-Prozess töten; der Startup-Hook markiert laufende Jobs als `FAILED`.
- **Maximale Zielobjekt-Kantenlänge hängt vom gewählten `tile_preset` ab** (Section 5.2, 8): 64 m (small), 128 m (medium — Default), 192 m (large). Objekte, die größer als das gewählte Maximum sind, werden an Safe-Center-Grenzen zersägt — IoU gegen ALKIS kollabiert. Der Nutzer muss das Preset passend zum Prompt wählen. `large` erzeugt bei 1 km² ~6 200 Kacheln (~1 h Laufzeit) — ein Job in diesem Preset ist kein Interaktiv-Workflow.
- **Kein In-place-Retry:** FAILED-Jobs werden nicht automatisch neu versucht. Nutzer muss neuen Job einreichen.
- **SAM License:** SAM 3 unterliegt der SAM-Lizenz von Meta. Für den Prototyp unkritisch, vor produktiver Nutzung prüfen.
- **UTM-Zonen-Verzerrung in Ostbayern:** Die Pipeline operiert durchgängig in EPSG:25832 (UTM-Zone 32N), weil das LDBV Bayern auch in der Zone-33N-Region (östlich von Passau / Bayerischer Wald) seine Daten zwangsweise in 32N ausliefert. Am östlichen Rand tritt dadurch projektionsbedingte Streckung auf, die Flächenberechnungen in `merger.py` und IoU-Auswertungen leicht verzerrt. Konsequenz: Metriken aus Würzburg (zentral in 32N) sind nicht 1:1 auf Metriken aus Passau übertragbar. Für den Prototyp akzeptiert; korrekt wäre eine coverage-spezifische EPSG-Wahl pro BBox.
- **Geodätische Transformation (4326 ↔ 25832) ohne BETA2007-Gitter:** pyproj verwendet per Default die analytische Helmert-Transformation, was in Bayern einen systematischen Lagefehler von 20–80 cm verursachen kann. Der amtliche NTv2-Gitterdatensatz BETA2007 wäre für Produktion-Grade-Arbeit die korrekte Wahl. **Für den Prototyp bewusst akzeptiert**, weil der Fehler ausschließlich an zwei Rändern auftritt: (a) bei der Umwandlung der User-gezeichneten BBox von WGS84 nach UTM (betrifft nur den Ausschnitt der empfangenen Kachel um ~50 cm) und (b) beim Serialisieren der Polygone nach GeoJSON für Leaflet (Display-relevant, nicht Metrik-relevant). **Die IoU-Evaluation gegen ALKIS ist nicht betroffen**, weil sowohl DOP20-extrahierte Polygone als auch ALKIS-Referenz in nativem EPSG:25832 vorliegen — zwischen ihnen findet keine Transformation statt. Falls später geodätische Strenge gefordert ist: `pyproj` mit explizit via `PROJ_NETWORK=ON` geladenem `de_adv_BETA2007.tif`/`.gsb` initialisieren.

## 15. Offene Punkte für die Implementierungsphase

- **WCS-Verifikation (höchste Priorität):** Folgende Punkte müssen beim ersten `GetCapabilities`-Call verifiziert werden, bevor irgendein Code produktiv läuft:
    1. Service-URL und Coverage-ID (Platzhalter in Config)
    2. Exakter Wert von `MAX_WCS_PIXELS` (nicht geraten)
    3. Gitter-Origin der DOP20-Coverage (Annahme: UTM-Zone-32N-Origin, aber bestätigen)
    4. **OGC-WCS-2.0-Axis-Order:** EPSG:25832 ist in manchen CRS-Registries als `(Northing, Easting)` definiert, in anderen als `(Easting, Northing)`. Wenn der `subset`-Parameter in falscher Reihenfolge gesendet wird (z. B. `subset=E(x1,x2)&subset=N(y1,y2)` vs `subset=N(y1,y2)&subset=E(x1,x2)`), lehnt der Server mit HTTP 400 "Invalid Axis" ab. Die korrekte Achsenbezeichnung (`E`/`N`, `x`/`y`, `Long`/`Lat`) und Reihenfolge muss manuell aus dem GetCapabilities-XML abgelesen und hart im Client codiert werden.
    5. Verhalten benachbarter Chunk-Kanten bei identischen Bounds — es muss verifiziert werden, dass `next_minx = prev_maxx` im fertigen VRT weder eine Lücke noch eine Doppelreihe erzeugt.
- **MIN_POLYGON_AREA_M2 kalibrieren:** Nach ersten Tests — Default 1 m² als Ausgangspunkt, Rausch-Pixel-Artefakte beobachten.
- **Performance-Messung:** Tatsächliche Tile-Rate auf der RTX 4070 mit SAM 3.1 — beeinflusst die UI-Fortschrittsanzeige (absolute Zeit vs. % der Tiles) und den Wert von `MAX_JOBS_PER_WORKER`.
- **SAM-3.1-Model-Größe:** Hiera-Large ist Default. Falls VRAM bei großen Prompts knapp wird, Fallback auf Hiera-Base prüfen.
