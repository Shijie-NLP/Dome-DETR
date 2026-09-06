# tools

Scripts that are not part of training or evaluation. Run them from the repository root with the
torch venv. Files that upstream Dome-DETR ships under `tools/` (benchmark, deployment, inference,
dataset conversion) are not on this branch yet; they come back one at a time once reviewed.

## analysis/cdn_negative_overlap.py

Do the contrastive-denoising (CDN) negatives of one ground-truth box land on *another* GT box?
CDN queries are sampled exactly as training does (`src/zoo/dome/denoising.py`) and each one is
compared with its source GT and with the other GT it overlaps most.

```
python tools/analysis/cdn_negative_overlap.py --image 0000059_01886_d_0000114 --plot out.png   # one image, by object size
python tools/analysis/cdn_negative_overlap.py --num-images 500                                 # a sample, by GT count
```

Reads VisDrone from `--root` (default `~/Data/datasets/visdrone`, official layout); ignored
regions and `others` are dropped. Columns:

| column | meaning |
| --- | --- |
| `GT-GT>=thr` | GT boxes whose best IoU with another GT is at least `thr`; how crowded the image is, no CDN involved |
| `other>=thr` | queries whose best IoU with a GT other than their source is at least `thr` |
| `other>own` | queries closer to some other GT than to their own source |
| `beats pos` | negatives only: take the other GT the negative overlaps most; the negative's IoU with it beats that GT's own positive in the same CDN group |

### VisDrone2019-DET train, full split

All 6471 images, 2 draws each, `box_noise_scale=1.0`, `thr=0.3`, seed 0 (commit d9dd387):

| GT per image | images | median size | GT-GT>=0.3 | pos other>=0.3 | pos other>own | neg other>=0.3 | neg other>own | neg beats pos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-50 | 3847 | 34.2px | 8.7% | 6.9% | 3.8% | 5.6% | 25.5% | 4.8% |
| 51-200 | 2544 | 24.4px | 15.2% | 12.5% | 7.0% | 9.5% | 38.7% | 7.8% |
| 201-500 | 78 | 20.3px | 19.9% | 15.9% | 9.1% | 13.1% | 47.6% | 10.5% |
| >500 | 2 | 15.9px | 24.2% | 13.6% | 7.8% | 11.7% | 40.7% | 8.8% |
| all | 6471 | 28.5px | 11.4% | 9.3% | 5.2% | 7.3% | 31.2% | 6.1% |

Over the whole split 6.1% of CDN negatives are a better box for some other GT than that GT's
own positive, yet are trained as background; the share grows with density (4.8% in sparse images,
10.5% at 201-500 GT; the `>500` row is two images). A third of the negatives are closer to
another GT than to their source, but most of those overlap neither (only 7.3% reach IoU 0.3).
Positives are ambiguous too: 5.2% sit closer to another GT than to the one they regress to.
