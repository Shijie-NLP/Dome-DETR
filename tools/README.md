# tools

Scripts that are not part of training or evaluation. Run them from the repository root with the
torch venv. Files that upstream Dome-DETR ships under `tools/` (benchmark, deployment, inference,
dataset conversion) are not on this branch yet; they come back one at a time once reviewed.

## analysis/cdn_negative_overlap.py

Do the contrastive-denoising (CDN) negatives of one ground-truth box land on *another* GT box?
CDN queries are sampled exactly as training does (`src/zoo/dome/denoising.py`) and each one is
compared with its source GT and with the other GT it overlaps most. Ground truth comes from the
Hub through the repo's dataset classes (`visdrone`, `aitod`, `voc`), filtered like a training
target: ignore regions and degenerate boxes out. Images with fewer than two boxes are skipped.

```
python tools/analysis/cdn_negative_overlap.py visdrone --image 0000059_01886_d_0000114 --plot out.png   # one image, by object size
python tools/analysis/cdn_negative_overlap.py aitod --split train --num-images 500                       # a sample, by GT count
```

Columns:

| column | meaning |
| --- | --- |
| `GT-GT>=thr` | GT boxes whose best IoU with another GT is at least `thr`; how crowded the image is, no CDN involved |
| `other>=thr` | queries whose best IoU with a GT other than their source is at least `thr` |
| `other>own` | queries closer to some other GT than to their own source |
| `beats pos` | negatives only: take the other GT the negative overlaps most; the negative's IoU with it beats that GT's own positive in the same CDN group |

### VisDrone2019-DET train, full split

6465 of 6471 images (6 have fewer than two boxes), 2 draws each, `box_noise_scale=1.0`,
`thr=0.3`, seed 0:

| GT per image | images | median size | GT-GT>=0.3 | pos other>=0.3 | pos other>own | neg other>=0.3 | neg other>own | neg beats pos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-50 | 3841 | 34.2px | 8.8% | 6.8% | 3.7% | 5.6% | 25.5% | 4.7% |
| 51-200 | 2544 | 24.4px | 15.2% | 12.6% | 7.0% | 9.6% | 38.8% | 7.8% |
| 201-500 | 78 | 20.3px | 19.9% | 16.2% | 9.3% | 12.6% | 47.1% | 10.3% |
| >500 | 2 | 15.9px | 24.2% | 15.0% | 8.8% | 12.4% | 42.8% | 10.4% |
| all | 6465 | 28.5px | 11.4% | 9.3% | 5.2% | 7.3% | 31.2% | 6.1% |

Over the whole split 6.1% of CDN negatives are a better box for some other GT than that GT's
own positive, yet are trained as background; the share grows with density (4.7% in sparse images,
10.3% at 201-500 GT; the `>500` row is two images). A third of the negatives are closer to
another GT than to their source, but most of those overlap neither (only 7.3% reach IoU 0.3).
Positives are ambiguous too: 5.2% sit closer to another GT than to the one they regress to.

### AI-TOD-v2 train, full split

7819 of 11214 images (10 are empty and 3385 hold a single box), 2 draws each,
`box_noise_scale=1.0`, `thr=0.3`, seed 0:

| GT per image | images | median size | GT-GT>=0.3 | pos other>=0.3 | pos other>own | neg other>=0.3 | neg other>own | neg beats pos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-50 | 6191 | 11.8px | 0.9% | 1.3% | 0.7% | 1.9% | 9.3% | 1.7% |
| 51-200 | 1394 | 11.8px | 1.8% | 3.5% | 2.0% | 6.5% | 28.1% | 5.4% |
| 201-500 | 205 | 11.3px | 1.7% | 5.2% | 3.0% | 10.9% | 40.7% | 8.8% |
| >500 | 29 | 10.0px | 4.2% | 14.5% | 7.9% | 22.3% | 59.8% | 16.6% |
| all | 7819 | 11.8px | 1.1% | 2.3% | 1.3% | 4.0% | 16.3% | 3.3% |

AI-TOD is tinier (median 12px against 28px) but far less crowded: only 1.1% of its boxes overlap
another box at IoU 0.3, against 11.4% on VisDrone, and 3.3% of its negatives beat another GT's
positive, about half the VisDrone rate. The dense tail behaves like VisDrone and worse: at
201-500 boxes the two datasets match (8.8% vs 10.3%), and the 29 images above 500 boxes reach
16.6%, with six negatives in ten closer to another GT than to their own source. The effect is a
function of crowding, not of object size.

## analysis/run_report.py

One report per training run, written into the run's own output directory: `RESULTS.md` and
`curves.png`. It reads what `train.py` leaves there (`config.yml`, the per-epoch `log.txt`,
`console.log`, the checkpoints and the COCOeval dumps under `eval/`) and answers the questions
one asks of a finished or half-finished run: how it was set up, whether it finished, the best
and last AP with every evaluator entry, which epoch each `.pth` holds, the stage-2 reloads,
AP/loss/learning-rate/epoch-time curves, per-class AP over the periodic `eval/` snapshots, and
the warnings the console caught.

```
python tools/analysis/run_report.py outputs/dfine_s_visdrone/2026-09-09_17-13-58
python tools/analysis/run_report.py outputs/*/*      # every run, one report each
```

`--every N` sets the epoch spacing of the progress table (default 10); `--no-figure` and
`--no-per-class` skip the plot and the `eval/` reading (the only part that needs torch).
