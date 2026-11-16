# saugios-rankos

Comparing pose estimators (SMPLest-X, Sapiens, MediaPipe) as input to one shared isolated ASL sign classifier: 44 signs, 1,348 videos, participant-independent train/val/test split.

## Workflow

```
videos ──► pose estimator ──► <model>_to_keypoints ──► keypoints/kp_<model>/ ──► sign_classifier ──► results/<run>/
```

Each pose estimator gets its own notebooks; all of them end in the same keypoint format (one `.npz` per video, the same 52 joints), and every model is trained and tested by the same `sign_classifier.ipynb`.

## Files

| File | What it does |
|---|---|
| `sign_classifier.ipynb` | **Shared classifier.** Set `KP_DIR`, `RUN_NAME` (and `COORDS`, `MIN_CONF`) in section 2, Run All. Section 8 compares all runs. Its appendix converts raw SMPLest-X output. |
| `sapiens_pipeline.ipynb` | Sapiens-1B on all videos, with the same person detector as SMPLest-X (YOLOv8x). Writes raw output to `sapiens_results/`, then runs `sapiens_to_keypoints.ipynb`. |
| `sapiens_to_keypoints.ipynb` | Sapiens output (133 COCO-WholeBody keypoints) → `keypoints/kp_sapiens/` (52 joints). |
| `mediapipe_classifier.ipynb` | MediaPipe under lower video quality (resolution, fps, compression, codec), with its own classifier (separate from `sign_classifier.ipynb`). |
| `mediapipe_to_keypoints.ipynb` | MediaPipe keypoints from `mediapipe_classifier.ipynb`'s cache → `keypoints/kp_mediapipe_<scenario>/`, then prints or runs the classifier per scenario. |

## Folders

| Folder | Content | In git |
|---|---|---|
| `dataset/` | videos and `video_info/all_videos.txt` (labels + split) | yes |
| `keypoints/kp_<model>/` | shared keypoint format, one `.npz` per video | yes |
| `results/<run>/` | classifier results per run (`results.json`, `per_sign_test.csv`, `confusion_test.npy`) | yes |
| `results/figures/` | comparison figures | yes |
| `sapiens/` | Sapiens code (cloned by `sapiens_pipeline.ipynb`) | no |
| `sapiens_results/` | raw Sapiens output, one `.json` + `.jpg` per frame | no |
| `data/`, `cache/` | data and cache of `mediapipe_classifier.ipynb` | no |

`sign_classifier.py` and `smplestx_to_keypoints.py` are written by `sign_classifier.ipynb` when it runs; they are not edited by hand and not committed.

## Results so far (test top-1, 3 seeds)

| Run | Test top-1 |
|---|---|
| SMPLest-X 2D | 0.629 ± 0.017 |
| SMPLest-X 3D | 0.637 ± 0.022 |
| Sapiens 2D (`MIN_CONF = 0`, like-for-like) | 0.670 ± 0.004 |

`MIN_CONF = 0` is the like-for-like setting: SMPLest-X confidence is only 0 or 1, so the 0.3 filter removes nothing there but about half of the Sapiens keypoints. See the Sapiens appendix of `sign_classifier.ipynb`.
