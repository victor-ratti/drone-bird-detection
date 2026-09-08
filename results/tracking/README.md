# Tracking runs

Stats and per-frame track records written by `src/track.py`. Annotated videos
are not versioned (regenerate them with the script); `evidence/` holds
downscaled frames cited in `../05_tracking.md`.

| Prefix | Video | Source, license | Note |
|---|---|---|---|
| `hawk_c25`, `hawk_c10` | Hawk attacks drone, 1920x1080, 24 fps, 855 frames | Wikimedia Commons, CC BY 3.0 | Drone point of view; detector confidence 0.25 vs 0.10 |
| `ystad_c25`, `ystad_c10` | Hovering drone, Ystad 2026, 2696x2160, 30 fps, 604 frames | Wikimedia Commons, CC BY 4.0 | Top-down close-up; same two thresholds |
| `173rd` | 173rd Airborne Brigade drone operator training, 1920x1080, 1233 frames | Wikimedia Commons, public domain | Edited documentary, unusable as a benchmark |
| `blm`, `blm_shot` | BLM drone training, 1920x1280, 932 frames | Wikimedia Commons, public domain | Edited documentary; `blm_shot` is its one real 17-frame flight shot |

The `173rd` and `blm` runs predate the tentative-track fix and the migration to
`trackers.ByteTrackTracker`; their numbers are kept as the record of what an
edited montage produces, not as tracker measurements.
