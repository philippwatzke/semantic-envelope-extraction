# LDBV DOP20 WMS Verification

Verified on: TBD

- WMS_URL:           https://geoservices.bayern.de/od/wms/dop/v1/dop20
- WMS_VERSION:       1.1.1
- WMS_LAYER:         by_dop20c
- WMS_CRS:           EPSG:25832
- WMS_FORMAT:        image/png
- WMS_MAX_PIXELS:    6000
- PNG band count:    TBD (expected: 4 = RGBA, alpha = NoData mask)
- Pixel resolution:  0.2 m native
- Coverage bbox (EPSG:25832): minx=497000 miny=5234000 maxx=857000 maxy=5604000

## Evidence

- GetCapabilities URL: https://geoservices.bayern.de/od/wms/dop/v1/dop20?SERVICE=WMS&REQUEST=GetCapabilities
- Supported formats (verbatim from <Format> tags): TBD
- Supported CRS (verbatim from <SRS> tags): TBD
- Max pixels clause (from <Abstract>): TBD
- Tested GetMap URL (Munich-area, 500×500 px): TBD
- PIL-verified band count of response: TBD
- Adjacent-chunk edge test — URL A: TBD
- Adjacent-chunk edge test — URL B: TBD
- Edge test result: pixel at column 0 of chunk B vs pixel at column (width-1) of chunk A — TBD (numerical diff)

## Notes

Complete this file before implementing the DOP client (Tasks 7/8). The DOP
client must use the verified format, layer name, and edge-alignment behaviour
from this document, not guessed defaults.
