# LDBV DOP20 WMS Verification

Verified on: 2026-04-25

- WMS_URL:           https://geoservices.bayern.de/od/wms/dop/v1/dop20
- WMS_VERSION:       1.1.1
- WMS_LAYER:         by_dop20c
- WMS_CRS:           EPSG:25832
- WMS_FORMAT:        image/png
- WMS_MAX_PIXELS:    6000
- PNG band count:    4 (RGBA, mode='RGBA', dtype=uint8) — Alpha is NoData mask
- Pixel resolution:  0.2 m native
- Coverage bbox (EPSG:25832, layer `by_dop20c`): minx=497000 miny=5234000 maxx=857000 maxy=5604000

## Evidence

- GetCapabilities URL: https://geoservices.bayern.de/od/wms/dop/v1/dop20?SERVICE=WMS&REQUEST=GetCapabilities
- Root element confirms version: `<WMT_MS_Capabilities version="1.1.1">`
- Supported formats (verbatim from `<Format>` tags inside `<GetMap>`):

  ```xml
  <Format>image/jpeg</Format>
  <Format>image/png</Format>
  <Format>image/tiff</Format>
  <Format>image/vnd.jpeg-png</Format>
  ```

- Supported CRS (verbatim from `<SRS>` tags on layer `by_dop20c`):

  ```xml
  <SRS>EPSG:25832</SRS>
  <SRS>EPSG:25833</SRS>
  <SRS>EPSG:31468</SRS>
  <SRS>EPSG:4258</SRS>
  <SRS>EPSG:4326</SRS>
  <SRS>EPSG:3857</SRS>
  <SRS>EPSG:5678</SRS>
  ```

- Max pixels clause (verbatim from service `<Abstract>`):

  > "Der WMS ist auf eine maximale Ausdehnung von 6.000 x 6.000 Pixel begrenzt."

- Layer bbox (verbatim from `<BoundingBox>` of `by_dop20c`):

  ```xml
  <BoundingBox SRS="EPSG:25832"
              minx="497000" miny="5.234e+06" maxx="857000" maxy="5.604e+06" />
  ```

- Access constraints (verbatim from `<AccessConstraints>`):

  > "CC BY 4.0 vgl. https://creativecommons.org/licenses/by/4.0/deed.de"

- Tested GetMap URL (Munich-area, 500×500 px):
  `https://geoservices.bayern.de/od/wms/dop/v1/dop20?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&LAYERS=by_dop20c&STYLES=&SRS=EPSG:25832&BBOX=690000,5334000,690100,5334100&WIDTH=500&HEIGHT=500&FORMAT=image/png&TRANSPARENT=TRUE`
  - HTTP 200, Content-Type: `image/png`, 695 723 bytes
  - PIL: `mode='RGBA'`, `size=(500, 500)`, `getbands()=('R', 'G', 'B', 'A')`, `dtype=uint8`
- PIL-verified band count of response: **4 (RGBA)**

### Adjacent-chunk edge test

- URL A (`BBOX=690000,5334000,690100,5334100`, 500×500):
  `https://geoservices.bayern.de/od/wms/dop/v1/dop20?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&LAYERS=by_dop20c&STYLES=&SRS=EPSG:25832&BBOX=690000,5334000,690100,5334100&WIDTH=500&HEIGHT=500&FORMAT=image/png&TRANSPARENT=TRUE`
- URL B (`BBOX=690100,5334000,690200,5334100`, 500×500):
  `https://geoservices.bayern.de/od/wms/dop/v1/dop20?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&LAYERS=by_dop20c&STYLES=&SRS=EPSG:25832&BBOX=690100,5334000,690200,5334100&WIDTH=500&HEIGHT=500&FORMAT=image/png&TRANSPARENT=TRUE`

**Test 1 — adjacent-strip diff (A col 499 vs B col 0):**
A's rightmost column samples the strip x∈[690099.8, 690100.0] m, B's leftmost column samples x∈[690100.0, 690100.2] m. They are physically adjacent (not overlapping), so a small diff is expected. Result over 500 rows:

| Channel | Mean abs diff | Max abs diff |
|--------:|:-------------:|:------------:|
| R       | 6.05 / 255    | 68           |
| G       | 6.02 / 255    | 65           |
| B       | 5.89 / 255    | 73           |
| A       | 0.00          | 0            |

This is normal variation between two real adjacent 0.2 m pixels in an orthophoto.

**Test 2 — bit-exact stitch test (definitive):** Request a single combined 1000×500 chunk covering A∪B (`BBOX=690000,5334000,690200,5334100`, `WIDTH=1000&HEIGHT=500`) and compare with the stitched `concat(A, B)` along axis=1. If the WMS aligns adjacent-bbox requests to its source raster, these must be byte-identical.

- Combined-vs-Stitched mean abs diff (RGB): **0.0**
- Combined-vs-Stitched max abs diff (RGB):  **0**
- All RGB diff percentiles (50%, 95%, 99%): **0**
- Seam columns (combined col 499 ≡ A col 499; combined col 500 ≡ B col 0): mean diff **0.000** on both sides.

**Verdict — PASS.** When BBoxes are snapped to the 0.2 m meter grid with `next_minx == prev_maxx` and WIDTH/HEIGHT chosen so px_size = 0.2 m exactly, the WMS returns byte-identical pixels at the seam. Mosaicking is gap-free, duplicate-free, and pixel-accurate. No structural drift; no resampling jitter. The naive `concat(A, B)` strategy is safe.

## Notes

- WMS version is 1.1.1, so BBOX axis order for EPSG:25832 is `minx,miny,maxx,maxy` (longitude/easting first), and the request parameter is `SRS=` (not `CRS=`). This must be honoured in the client.
- Native PNG response from `LAYERS=by_dop20c` is RGBA. The alpha channel is the NoData mask (255 inside coverage, 0 outside). Inside the verified Munich AOI alpha was uniformly 255.
- Service abstract enforces a hard 6000×6000 px GetMap cap. Tile chunking must respect this, e.g. ≤ 6000 px per side per request.
- Layer also exposes `image/jpeg`, `image/tiff`, `image/vnd.jpeg-png`. PNG is chosen for lossless RGBA. JPEG would not preserve the alpha channel.
- `LatLonBoundingBox` for `by_dop20c` is `minx=8.957621 miny=47.162543 maxx=14.032767 maxy=50.587905` (WGS84) — useful for client-side AOI prevalidation.
