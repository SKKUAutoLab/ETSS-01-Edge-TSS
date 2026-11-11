# MLKit — Traffic Surveillance System (TSS)

### Pipeline & Project Structure Guide

> Onboarding reference for the MLKit workspace and its main application, the multi-camera vehicle
> tracking pipeline in `projects/tss`. Based on a review of the code, configs, environment and data in
> this workspace (October 2026; updated November 2026). Paths are relative to the repository root `MLKit/`
> unless noted.

**Contents**

1. [Project Overview](#1-project-overview)
2. [Tech Stack & Dependencies](#2-tech-stack--dependencies)
3. [Directory Tree](#3-directory-tree)
4. [Architecture](#4-architecture)
5. [Pipeline & Data Flow](#5-pipeline--data-flow)
6. [Core Components & Modules](#6-core-components--modules)
7. [Configuration](#7-configuration)
8. [Data Layout](#8-data-layout)
9. [Setup & Execution](#9-setup--execution)
10. [Conventions](#10-conventions)
11. [Where to Look](#11-where-to-look)
12. [Known Issues & Gotchas](#12-known-issues--gotchas)

---

## 1. Project Overview

MLKit is a research workspace for traffic-video analytics. Its main application, **TSS**
(`projects/tss`), is an **offline multi-target multi-camera (MTMC) vehicle tracking pipeline**. From
time-aligned videos of several roadside cameras, it:

1. detects vehicles in every frame (YOLOX),
2. describes each detection with an appearance (re-identification, *re-ID*) feature,
3. tracks vehicles within each camera (single-camera tracking, *SCT*), and
4. links the tracks of the same vehicle across cameras (inter-camera tracking, *ICT*), so every vehicle
   gets one **global ID**.

The pipeline runs in six stages, driven either by a **PyQt6 desktop GUI** (one button per stage, live
previews) or by **shell scripts** that wrap a single Python entry point. It is built on an
in-house PyTorch utility library (`factory/`, `vision/`, `file/`, … under `projects/tss/src/`, §6.9), and descends from AI City Challenge (AIC21 /
AIC22) MTMC code.

The workspace is currently set up for the bundled dataset `tsd_suwon_19_46_20`:

| Item | Value |
|---|---|
| Cameras (road order) | `19_SUWON` → `46_SUWON` → `20_SUWON` |
| Scene ("subset") | `clearsky_1764390903` (suffix = Unix time of the recording, 2025-11-29) |
| Videos | 15 fps, ~9,028 frames (~10 min) each; 1280×720 (19, 46), 1024×768 (20) |
| Classes | `bus`, `small truck`, `truck` |
| Detector | YOLOX-X, one fine-tuned model per camera |
| Re-ID model | ResNet101-IBN-a, 384×384 input, 2048-d features |
| Last stored score | IDF1 **93.80** (`projects/tss/data/tsd_suwon_19_46_20/outputs/track_mtmc.txt`) |

---

## 2. Tech Stack & Dependencies

| Area | Technology | Version in the working `etss-vehicereidentification` env | Role |
|---|---|---|---|
| Language | Python | 3.10 | All application code |
| Scripting | Bash | — | Stage runner scripts |
| Deep learning | PyTorch / torchvision | 2.7.1 / 0.22.1 (CUDA 11.8 builds) | Models, NMS |
| Detection | YOLOX (Megvii), vendored copy | 0.3.0 | Vehicle detection |
| Re-identification | AIC re-ID baseline (vendored), `yacs` | yacs 0.1.8 (`timm` is not installed; only the unused ConvNeXt and transformer-block backbones import it) | Appearance features |
| Tracking | FairMOT-style `JDETracker` (vendored), `lap`, `filterpy` | lap 0.5.12, filterpy 1.4.5 | Single-camera tracking |
| Matching | scikit-learn `AgglomerativeClustering`, k-reciprocal re-ranking | scikit-learn 1.7.2 | Cross-camera IDs |
| Vision / numerics | OpenCV, NumPy, SciPy, Shapely, Pillow | 4.12.0, 2.2.6, 1.15.3, 2.1.2, 12.2.0 | I/O, geometry |
| GUI | PyQt6 (Qt Designer `.ui` loaded at runtime) | 6.9.1 | Desktop front end |
| Config | PyYAML, Munch, easydict | 6.0.3, 4.0.0, 1.13 | YAML → attribute-style dicts |
| Logging | loguru, rich, tqdm | 0.7.3, 13.9.4, 4.67.3 | Logs, progress bars |
| Evaluation | In-house MOT/MTMC evaluator; TrackEval (vendored) | — | IDF1, MOTA, HOTA |
| Annotation | X-AnyLabeling | JSON format 3.3.3 | ROI and zone polygons |
| Environment | conda + pip | — | `setup.sh` |
| Shared library | In-house toolkit under `projects/tss/src/` (author Long H. Pham) | — | Registries, video I/O, utilities |

The C++ source in the tree (`MinCostPerfMatch`) belongs to a legacy vendored component and is not built or
used.

**Hardware.** In practice you need an NVIDIA GPU. YOLOX-X takes about 5 s per frame on CPU, and the
cross-camera re-ranking step always calls `.cuda()`. Inference needs about 2.4 GB of weights (§9.3).

> **YOLOX source.** `poetry install` in `projects/tss` (§9.2) registers `projects/tss/third_party/YOLOX`
> in the `etss-vehicereidentification` env, so `import yolox` loads the vendored copy. The env also holds
> `site-packages/traffic flow estimation.pth` from the sibling ETSS-02-VehicleDetTrack checkout, which
> points at an identical YOLOX copy; this project's `.pth` file sorts first, so it wins.

---

## 3. Directory Tree

`★` marks the current pipeline path. *legacy* means kept for reference but unused by current runs.
*vendored* means third-party code copied into the repo.

```
MLKit/
├── projects/
│   └── tss/                      ★ Traffic Surveillance System (pyproject v0.0.2); the working directory (§9)
│       ├── pyproject.toml        ★ Poetry metadata; `poetry install` makes `src` and `yolox` importable (§9.2)
│       ├── setup.sh              ★ pip install list (torch 2.7.1+cu118 and dependencies), then poetry install
│       ├── run_gui_vehicle_tracking.sh ★ Launcher: activates conda env `etss-vehicereidentification`, opens the GUI
│       ├── log.txt                 FairMOT log, rewritten in the working directory at import time
│       ├── runs/
│       │   ├── main_mtmc.py      ★ CLI entry point: one camera config + stage flags
│       │   ├── run_track_mtmc_button_*.sh ★ One script per stage (called by the GUI)
│       │   ├── run_track_mtmc_cert.sh     Runs all stages from the shell
│       │   ├── run_track_mtmc_gui.sh      GUI launcher without conda activation
│       │   └── other run_track_*.sh       legacy: dated/AIC22 variants, most call missing files
│       ├── src/
│       │   ├── builder.py        ★ Registries: CAMERAS, DETECTORS, REIDENTIFIER, MOTIONS, TRACKERS
│       │   ├── configuration.py  ★ Path constants (data_dir, models_zoo_dir, checkpoints_dir, …), load_config(), copy_config_file()
│       │   ├── camera/           ★ mtmc_camera.py: MTMCTrackingCamera, the stage orchestrator
│       │   │                       (base.py: BaseCamera interface)
│       │   ├── detectors/
│       │   │   ├── detect.py     ★ CLI YOLOX detection for one camera (start with python -m)
│       │   │   ├── yolox_adaptor.py ★ YOLOX_Adapter: the same detection for the GUI, frame by frame
│       │   │   └── basedetector.py BaseDetector interface (YOLOX_Adapter implements it)
│       │   ├── reidentifiers/    ★ ReidFeature (features)
│       │   │   ├── baseline/       vendored re-ID backbones + make_model
│       │   │   └── configs/        aictest.yml, the base re-ID model config
│       │   ├── matchers/         ★ ReidMatching (cross-camera)
│       │   │   └── utils/          zone_intra.py, filter.py, visual_rr.py, rerank.py, ...
│       │   ├── trackers/         ★ FairMOT wrapper + fairmot/ (vendored JDETracker)
│       │   │                       sort.py, motion/: SORT + Kalman bbox motion (unused)
│       │   ├── objects/            Instance, BaseObject data classes (SORT path)
│       │   ├── io/               ★ mtmc_drawer.py: MTMCDrawer (videos)
│       │   ├── views/            ★ GUI: multicamera_multiobject_tracking_view.py + mtmc.ui
│       │   │                       (display_*.jpg = detection previews)
│       │   ├── factory/          ★ Registry + Factory: the plug-in mechanism
│       │   ├── data/               ClassLabels, dataset and augmentation helpers
│       │   ├── file/               load/dump (YAML, JSON, pickle), path predicates
│       │   ├── vision/             FrameLoader / FrameWriter, bbox ops, color, geometry, filters
│       │   ├── enhance/            legacy: unused copy of vision/enhance/normalize.py (imports a missing tss.*)
│       │   ├── math/ type/ utils/  Math helpers, type aliases, device selection, rich console
│       │   └── constants.py
│       ├── data/                   Datasets: videos, masks, configs, outputs (see §8)
│       │   └── tsd_suwon_19_46_20/
│       ├── tests/                  Evaluation tooling (no unit tests)
│       │   ├── mot_evaluation/   ★ evaluate_mtmc.py: IDF1 of MTMC results (nested .git)
│       │   ├── TrackEval/          vendored official TrackEval (HOTA / CLEAR / Identity)
│       │   └── performance_test/   legacy AIC21 vehicle-counting evaluation
│       ├── third_party/YOLOX/    ★ Vendored YOLOX (yolox/, exps/, tools/): the `yolox` package that every
│       │                           `from yolox… import` loads, registered by `poetry install` (§9.2)
│       └── models_zoo/             Weights, ~78 GB in total
│           ├── yolox/yolox_x_<cam>_suwon/ ★ best_ckpt.pth (+ per-epoch checkpoints, train_log.txt)
│           └── pretrained/         Re-ID backbones (resnet/resnet101_ibn_a_2.pth ★, ...),
│                                   legacy YOLOv5 / ScaledYOLOv4 weights (their code is removed)
├── ultilities/                     Helper scripts (the folder name is spelled this way)
│   ├── convert_xanylabeling_to_zone_mtmc.py ★ polygons → zone mask + ROI mask
│   ├── create_video_demo.py, create_video_sync_mtmc.py   demo and multi-camera grid videos
│   ├── find_vehicle_via_cameras.py  lists track IDs seen by at least N cameras in an MTMC text file
│   ├── mot_evaluation/             Copy of the MOT evaluator
│   └── playground.py, mtmc_det_gth_to_pkl.py (empty), sample images (gt.jpg, image.jpg, lenna.png)
├── docs/                           pipeline_framework.readme.md (copy of this guide), prompt notes
├── log.txt                         Stale (empty): runs now write projects/tss/log.txt
├── .claude/, .agents/              AI-assistant skills (style guides, SOLID, docstring format)
└── .vscode/, .idea/                Editor settings (projects/tss/data excluded from search)
```

---

## 4. Architecture

### 4.1 Component view

```
┌────────────────────────────────────────────────────────────────────────┐
│ PyQt6 GUI  ·  projects/tss/src/views/multicamera_multiobject_tracking_view.py │
│ stage buttons · Run All · 3 live display panes · message log           │
└──────────┬───────────────────────────────────────────┬─────────────────┘
           │ QProcess (stages 1, 3, 4, 5)              │ QThread (stage 2)
           ▼                                           ▼
 runs/run_track_mtmc_button_<stage>.sh       detectors/yolox_adaptor.py
           │  loops over the 3 cameras                 │  YOLOX-X, one model per camera
           ▼                                           │
 runs/main_mtmc.py --<stage flags> --config <cam>.yaml │
           │  load_config() → Munch                    │
           ▼                                           │
 MTMCTrackingCamera (src/camera/mtmc_camera.py)     │
   ├─ ReidFeature        REIDENTIFIER "re_identifier_feature"
   ├─ FairMOT            TRACKERS     "fairmot_track_mtmc"
   ├─ ReidMatching       REIDENTIFIER "re_identifier_matching"
   │     └─ ZoneIntra · visual_rerank · AgglomerativeClustering
   ├─ MTMCDrawer         src/io/mtmc_drawer.py
   └─ src/ toolkit       Factory · FrameLoader · FrameWriter · ClassLabels
           │                                           │
           ▼                                           ▼
   projects/tss/data/<dataset>/temps/   ◄──────────────┘
   pickles · txt · debug videos  ──►  tests/mot_evaluation/evaluate_mtmc.py (IDF1)
```

How the pieces interact:

- **The GUI is a thin front end.** Apart from YOLOX detection, it doesn't import the pipeline. It launches
  the stage scripts as child processes and watches their stdout.
- **One process per camera per stage.** Each stage script calls `main_mtmc.py` once per camera (or once
  per scene). `main_mtmc.py` loads the YAML config, adds the stage flags, and constructs one
  `MTMCTrackingCamera`.
- **Files are the interface.** The camera builds only the components the requested stages need, runs
  them, and writes results under `temps/`. The next stage, usually in a new process, reads them back.
  Stages share no in-memory state.

### 4.2 Plug-in registry

Components are created from config by name. `projects/tss/src/builder.py` defines five
`src.factory.Factory` registries. Classes register with a decorator, and the camera builds them from
the YAML `name` field:

```python
@TRACKERS.register(name="fairmot_track_mtmc")
class FairMOT(JDETracker): ...

self.tracker = TRACKERS.build(**tracker_cfg)      # tracker_cfg["name"] == "fairmot_track_mtmc"
```

| Registry | Registered name | Class | File | Status |
|---|---|---|---|---|
| `CAMERAS` | `aic_mtmc_camera` | `MTMCTrackingCamera` | `src/camera/mtmc_camera.py` | current (instantiated directly by `main_mtmc.py`) |
| `REIDENTIFIER` | `re_identifier_feature` | `ReidFeature` | `src/reidentifiers/reid_feature.py` | current |
| `REIDENTIFIER` | `re_identifier_matching` | `ReidMatching` | `src/matchers/reid_matching.py` | current |
| `TRACKERS` | `fairmot_track_mtmc` | `FairMOT` | `src/trackers/fairmot_track_mtmc.py` | current |
| `TRACKERS` | `sort` | `SORT` | `src/trackers/sort.py` | unused |
| `MOTIONS` | `kf_bbox_motion` | `KFBBoxMotion` | `src/trackers/motion/kf_bbox_motion.py` | SORT only |
| `DETECTORS` | — | — | — | empty: the legacy YOLOv5 wrappers were removed |

YOLOX is **not** a registered detector. It runs as a standalone script (§6.2).

### 4.3 Entry points

| Entry point | Purpose |
|---|---|
| `projects/tss/run_gui_vehicle_tracking.sh` | Normal way to run: activates `etss-vehicereidentification`, starts the GUI |
| `projects/tss/src/views/multicamera_multiobject_tracking_view.py` | GUI module (`MTMCMainWindow`, `main()`) |
| `projects/tss/runs/main_mtmc.py` | CLI: run selected stages for one camera (or scene) config |
| `projects/tss/runs/run_track_mtmc_button_<stage>.sh` | Stage scripts covering all cameras; also called by the GUI |
| `projects/tss/runs/run_track_mtmc_cert.sh` | All stages in sequence from the shell |
| `projects/tss/src/detectors/detect.py` | Standalone YOLOX detection for one camera |
| `projects/tss/tests/mot_evaluation/evaluate_mtmc.py` | IDF1 against the ground truth |
| `ultilities/convert_xanylabeling_to_zone_mtmc.py` | Builds zone and ROI masks from annotations |

---

## 5. Pipeline & Data Flow

### 5.1 Stages

Stages 1, 2, 3a and 4 run **once per camera**. Stages 3b and 5 run **once per scene**, with
`--config default.yaml --config_feat_merge reid_merge.yaml`. All artifacts go to
`projects/tss/data/<dataset>/temps/`, written `temps/` below.

| # | Stage (GUI button) | Script → command | Main outputs in `temps/` |
|---|---|---|---|
| 1 | Image extraction | `run_track_mtmc_button_image_extraction.sh` → `main_mtmc.py --images_extraction` | `images/<cam>/000000.jpg …` |
| 2 | Detection | GUI: in-process `YOLOX_Adapter.detect_source` (`detectors/yolox_adaptor.py`). Shell: `run_track_mtmc_button_detection.sh` → `detectors/detect.py` | `dets_crop/yolox/<cam>/*.jpg`, `dets_label/yolox/<cam>/*.txt`, `dets_crop_pkl/yolox/<cam>/<cam>_dets_crop.pkl`, `dets_debug/yolox/<cam>.mp4` |
| 3a | Re-identification | `run_track_mtmc_button_reidentification.sh` → `--feature_extraction --config reid/<cam>_resnet101_ibn_a_2.yaml` | `dets_feat/resnet101_ibn_a_2/<cam>/<cam>_dets_feat.pkl` |
| 3b | ↳ feature merge | same script → `--feature_merge` | `dets_feat_merge/<cam>/<cam>_dets_feat.pkl` |
| 4a | Single-camera tracking | `run_track_mtmc_button_single_camera_tracking.sh` → `--tracking` | `mots_feat/fairmot/<cam>/<cam>_mot_feat_raw.pkl`, `mots/fairmot/<cam>/<cam>_mot.txt` |
| 4b | ↳ zone matching | same script → `--matching_zone --use_mots_feat_raw --config_feat_merge reid_merge.yaml` | `mots_feat/fairmot/<cam>/<cam>_mot_feat_zone.pkl`, `…/<cam>_mot_feat_merge.pkl` |
| 4c | ↳ debug videos | same script → `--save_tracks_img`, `--save_tracks_zone` | `mots_debug/fairmot/<cam>_mots_feat_{raw,zone}.mp4` |
| 5a | Inter-camera tracking | `run_track_mtmc_button_inter_camera_tracking.sh` → `--matching_scene` | `mtmc_result.pkl` |
| 5b | ↳ write result | same script → `--write_result` | `mtmc_result.txt` |
| 5c | ↳ draw result | same script → `--save_mtmc_result` | `mtmc_debug/<cam>.mp4` |
| 5d | ↳ evaluate | same script → `tests/mot_evaluation/evaluate_mtmc.py` | IDF1 on stdout and in `outputs/track_mtmc.txt` |
| 6 | Show result (GUI only) | plays `temps/mtmc_debug/<cam>.mp4` in the three panes | — |

Optional steps that no stage script uses:

- `--detection_filter`: rule-based filtering of merged detections into `<cam>_dets_feat_filter.pkl`,
  consumed by `--tracking --use_dets_feat_filter`.
- `--tracking_postprocess`: post-association plus track NMS, writing `<cam>_mot_feat.pkl` and
  `mots/fairmot/<cam>/res/`.
- `--save_dets_img_filter`: debug video of the filtered detections.

### 5.2 Data-flow diagram

Paths are relative to `temps/`, except the inputs marked `◄`.

```
 <subset>/<cam>/<cam>.mp4                                         ◄ input video
        │ [1] image extraction                                       per camera
        ▼
 images/<cam>/NNNNNN.jpg ──┐
 <subset>/<cam>/<cam>_roi.jpg ◄ ROI mask
                           │ [2] YOLOX detection (ROI-masked)        per camera
                           ▼
 dets_crop/yolox/<cam>/*.jpg  +  dets_crop_pkl/yolox/<cam>/<cam>_dets_crop.pkl
        │ [3a] re-ID features, 2048-d per crop                      per camera, per model
        ▼
 dets_feat/<model>/<cam>/<cam>_dets_feat.pkl
        │ [3b] merge models (L2-normalize, then mean)               scene
        ▼
 dets_feat_merge/<cam>/<cam>_dets_feat.pkl
        │ [4a] FairMOT tracking                                     per camera
        ▼
 mots_feat/fairmot/<cam>/<cam>_mot_feat_raw.pkl   (+ mots/.../<cam>_mot.txt)
        │ [4b] zone lookup + tracklet filter   ◄ zone/<cam>_zones.png
        ▼
 <cam>_mot_feat_zone.pkl  →  <cam>_mot_feat_merge.pkl   (one record per tracklet)
        │ [5a] cross-camera clustering         ◄ reid_merge.yaml (camera order)    scene
        ▼
 mtmc_result.pkl     {(cam, track_id): global_id}
        │ [5b] write result
        ▼
 mtmc_result.txt ──[5c]──► mtmc_debug/<cam>.mp4 ──[6]──► GUI "Show Result"
        └───────[5d]──► evaluate_mtmc.py vs groundtruths/gt.txt ──► IDF1
```

### 5.3 Artifact formats

| Artifact | Format |
|---|---|
| Detection pickle `<cam>_dets_crop.pkl` | `dict[crop_name → {bbox: (x1, y1, x2, y2), frame: "00000123", id: int, imgname: "<crop_name>.jpg", class: int, conf: float}]`, where `crop_name = <frame:08d>_<det:06d>`. Stage 3 adds `feat` (2048-d `float32`). |
| Track pickle `<cam>_mot_feat_raw.pkl` | `dict["<cam>_<track:08d>_<frame:04d>.png" → {bbox, frame: "000123", id, imgname, class, score, feat}]`. Zone matching adds `zone` (0–4). |
| Tracklet pickle `<cam>_mot_feat_merge.pkl` | `dict[track_id → {cam, tid, mean_feat, zone_list, frame_list, tracklet, io_time}]` |
| Cluster pickle `mtmc_result.pkl` | `{"cluster": {(cam, track_id): global_id}}` |
| YOLO labels `dets_label/.../<frame:08d>.txt` | `cls cx cy w h conf`, normalized to [0, 1] |
| MOT text `<cam>_mot.txt` | `frame,id,x,y,w,h,score,-1,-1,-1` |
| MTMC result and ground truth (`mtmc_result.txt`, `groundtruths/gt.txt`) | `<cam> <global_id> <frame> <x> <y> <w> <h> -1 -1`, space-separated, frames 1-based |

Frame indices are 0-based in every pickle. `writing_result` adds 1 for the text output.

---

## 6. Core Components & Modules

### 6.1 Orchestrator — `MTMCTrackingCamera`

`projects/tss/src/camera/mtmc_camera.py`

- **Construction.** `main_mtmc.py` builds it from the camera YAML, a `process` dict of stage flags, and the
  merge config (`featuremerger`, loaded from `--config_feat_merge`).
- **`__init__`.**
  - Resolves directories in `init_dirs`: `root_dir = data/<dataset>`, `video_dir = <root>/<subset>`,
    `outputs_dir = <root>/temps`.
  - Loads the class labels and creates every output folder (`init_data_writer`).
  - **Opens the camera video to read its frame count**, so the video must exist even for scene-level
    stages.
- **Lazy components.** It builds only what the requested stages need (re-ID model, tracker, matcher),
  through the registries.
- **`run()`.** Executes the enabled stages in a fixed order: extract → detect → features → merge →
  filter → track → post-process → draw tracks → zone matching → draw zones → scene matching → write
  result → draw MTMC. It frees GPU memory after detection and after feature extraction.
- **Isolation.** Every stage reads its inputs from files, so any stage can be rerun alone.

### 6.2 Detection — YOLOX

`projects/tss/src/detectors/detect.py`, `yolox_adaptor.py`; YOLOX 0.3.0 itself is vendored in `projects/tss/third_party/YOLOX/`

- **Two front ends, one logic.** `detect.py` (CLI) and `YOLOX_Adapter` (for the GUI) do the same work.
  In `detect.py`, `create_predictor(opt)` loads an experiment file and its checkpoint, and
  `detect(predictor, opt)` iterates over the extracted frames. `YOLOX_Adapter` is built from the camera
  YAML's `detector` section (`variant` is the experiment file), and its generator `detect_source(...)`
  iterates over the frames and yields each annotated one. It runs FP16 (the camera YAMLs set `half: true`) without
  Conv+BN fusion, and keeps YOLOX's box math in FP16, as detect_yield did; fusion plus float32 NMS moved 369
  of 654 label lines by up to 1 px.
- **Experiment files.** `third_party/YOLOX/exps/example/custom/yolox_x_<cam>_suwon.py`: YOLOX-X (depth 1.33, width 1.25),
  3 classes, 1280×1280 input. Checkpoints are `projects/tss/models_zoo/yolox/yolox_x_<cam>_suwon/best_ckpt.pth`.
- **Masking.** The ROI mask (`<subset>/<cam>/<cam>_roi.jpg`) is applied with `cv2.bitwise_and` before
  inference, so masked areas produce no detections.
- **Settings.** Class-agnostic NMS at IoU 0.45, confidence threshold 0.1.
- **Outputs.** Crops, YOLO label files, the detection pickle, a debug video, and a preview image,
  `projects/tss/src/views/display_<last character of the camera name>.jpg`. For the Suwon cameras that is
  always `display_N.jpg`; the GUI uses the in-memory frame signal instead.

### 6.3 Re-identification features — `ReidFeature`

`projects/tss/src/reidentifiers/reid_feature.py` (registered as `re_identifier_feature`)

- **Model.** Built with `baseline/model/make_model.py` from `configs/aictest.yml`, overriding `MODEL.NAME`
  (`reid_backbone`) and `INPUT.SIZE_TEST` (`reid_size_test`). Weights load from `projects/tss/models_zoo/<reid_model>`.
- **Inference.** Resize to 384×384, then ImageNet normalization. Features are flip-averaged
  (`TEST.FLIP_FEATS: 'yes'`). The output is one 2048-d vector per crop.
- **Backbones** in `baseline/model/backbones/`: ResNet-IBN-a, ResNeXt-IBN, SE-ResNet-IBN, DenseNet-IBN,
  ResNeSt, ViT (transformer), ConvNeXt. Weights for several are in `projects/tss/models_zoo/pretrained/`.
- **Feature merge** (`run_feature_merge`). For each camera, it L2-normalizes and averages the features of
  every model listed under `reidentifier:` in `reid_merge.yaml`. That list currently holds one model,
  `resnet101_ibn_a_2`, but it supports ensembles.

### 6.4 Single-camera tracking — `FairMOT`

`projects/tss/src/trackers/fairmot_track_mtmc.py` (registered as `fairmot_track_mtmc`) wraps
`fairmot/fm_tracker/multitracker.py::JDETracker`.

- **Input.** Per frame, the merged detections and their features, pre-filtered by torchvision NMS
  (`nms_max_overlap`, 0.99, so effectively off) and `tracker.min_confidence`.
- **Association cascade** in `JDETracker.update`:
  1. appearance (embedding) distance fused with Kalman-filter motion gating, linear assignment at 0.4;
  2. remaining tracks against remaining detections by pixel distance, up to 150 px;
  3. unconfirmed tracks by IoU distance, 0.7.
- **Track state.** Features are an exponential moving average (α = 0.9). Lost tracks are kept for
  `frame_rate` frames, i.e. 1 s.
- **Outputs.** The raw track pickle and MOT text. Optional `post_process` runs `post_association.associate`
  twice, then `track_nms(0.65)`.
- `sort.py` with `motion/kf_bbox_motion.py` is a classic SORT tracker on `MovingObject`s. No current
  config uses it, and it can no longer be built: `MovingObject` was removed (§12, item 15).

### 6.5 Zones and tracklet filtering — `ZoneIntra`

`projects/tss/src/matchers/utils/zone_intra.py`, called by `ReidMatching.match_zone`.

A box's zone is the color of `data/<dataset>/zone/<cam>_zones.png` at the box center:

| Zone | Mask color | Meaning |
|---|---|---|
| 0 | black | Interior: no zone |
| 1 | white | Side branch (non-highway) |
| 2 | red | Side branch (non-highway) |
| 3 | green | Road to/from the **next** camera in the chain |
| 4 | blue | Road to/from the **previous** camera in the chain |

`filter_mot` keeps a tracklet only when `is_ignore` returns 0. Simplified, it drops:

- tracklets that start and end in the same zone (parked vehicles, false positives), except green/blue
  tracklets that pass through the interior;
- tracklets that only move between the side branches (1 ↔ 2);
- at the first camera, tracklets between a side branch and blue; at the last camera, tracklets between a
  side branch and green. Those roads lead to no monitored camera.

`merge_trajectory_zone` then reduces each surviving tracklet to one record:

- mean feature, preferring boxes larger than 2000 px²;
- zone list and frame list;
- entry/exit time (`io_time`), using the camera offset from `timestamp/<subset>.txt`.

### 6.6 Cross-camera matching — `ReidMatching`

`projects/tss/src/matchers/reid_matching.py` (registered as `re_identifier_matching`)

1. **Pairing.** `sub_cluster` loads every camera's tracklet pickle and groups tracklets into **adjacent
   camera pairs**, using their entry/exit zones and the camera order in `reid_merge.yaml`
   (`utils/filter.py::subcam_list`, `subcam_list2`). A tracklet leaving camera *i* through green is
   compared only with tracklets entering camera *i+1* through blue, and the same in reverse.
2. **Similarity.** Per pair: L2-normalized features → k-reciprocal visual re-ranking
   (`utils/visual_rr.py`, on GPU) → multiplied by an intra-camera mask, so there are no matches within one
   camera.
3. **Clustering.** `AgglomerativeClustering` with complete linkage on the precomputed distance. A second
   round re-clusters, using features averaged over the first-round clusters.
4. **Global IDs.** A cluster is kept only if it spans at least two cameras and repeats none. Each kept
   cluster gets a global ID, saved to `mtmc_result.pkl`.
5. **Output.** `writing_result` writes the zone-filtered boxes of matched tracklets to `mtmc_result.txt`,
   clipping boxes to the image size (read from the ROI mask).

A spatio-temporal filter (`st_filter`) exists but is commented out. The clustering thresholds are
hard-coded (§7.3).

### 6.7 Visualization — `MTMCDrawer`

`projects/tss/src/io/mtmc_drawer.py` renders the detection, track, zone-track and final MTMC videos. The
final MTMC drawing runs one thread per camera, with a stable color per global ID and a frame-number panel.

### 6.8 GUI — `MTMCMainWindow`

`projects/tss/src/views/multicamera_multiobject_tracking_view.py`

- **Layout.** Loads `mtmc.ui` at runtime (`uic.loadUi`): six stage buttons, **Run All**, three display
  panes, and a message log.
- **Stages 1, 3, 4, 5** run `bash runs/run_track_mtmc_button_<stage>.sh` (from `projects/tss`) through
  `QProcess`. `PROCESS …` markers on stdout highlight the active stage.
- **Stage 2** runs in-process. `DetectionWorker` (`src/views/display_workers.py`), on a `QThread`, builds
  one `YOLOX_Adapter` per camera, round-robins their `detect_source` generators and emits each annotated
  frame to the display panes.
- **Display.** `DisplayThread` refreshes the panes at 15 fps. **Show Result** plays
  `temps/mtmc_debug/<cam>.mp4`.
- **Run All** chains the stages in `process_finished`. It doesn't call `run_track_mtmc_cert.sh`.
- **Hard-coded setup.** Cameras and paths are fixed at the top of the module (`detector_configs`,
  `mtmc_viz`); each camera's detector settings come from the `detector` section of its YAML.

### 6.9 Shared toolkit (`projects/tss/src/`)

These packages were `torchkit/core` until 2026-11-03, when they moved into `projects/tss/src/` unchanged
(imports changed from `torchkit.core.X` to `projects.tss.src.X`). The same day `torchkit/utils.py` was
merged into `projects/tss/utils.py` and the `torchkit/` folder was deleted. On 2026-11-04 the
`projects.tss.` prefix was dropped, and `utils.py` was renamed to `configuration.py` and moved into `src/`:
imports are now `src.X`, provided by the editable install of `projects/tss/pyproject.toml` (§9.2). Pieces TSS relies on:

| Symbol | Location | Role |
|---|---|---|
| `Factory`, `Registry` | `projects/tss/src/factory/` | Name → class registries (`register`, `build`, `build_from_dict`) |
| `FrameLoader`, `FrameWriter` | `projects/tss/src/vision/io/frame.py` | Batched reading of videos and image folders; writing `.jpg` frames and/or `.mp4` |
| `ClassLabels` | `projects/tss/src/data/` | Loads `class_labels.json` |
| `load`, `is_json_file`, `is_basename`, `is_stem` | `projects/tss/src/file/` | File I/O and path predicates |
| `console`, `select_device` | `projects/tss/src/utils/` | rich console logging, CUDA device selection |
| `bbox_xyxy_to_cxcywh_norm`, `batch_bbox_iou`, … | `projects/tss/src/vision/` | Box conversions and IoU |

### 6.10 Legacy and vendored code (not on the current path)

| Code | Origin / purpose |
|---|---|
| `objects/`, `trackers/sort.py`, `data/<dataset>/rmois/` | AIC21 vehicle counting (regions/movements of interest) |
| `enhance/` | Stray copy of `vision/enhance/normalize.py`; nothing imports it |
| `matchers/utils/zone.py` | Older zone logic with AIC camera numbering; `zone_intra.py` replaces it |
| `trackers/fairmot/post_processing/MinCostPerfMatch/` (C++), `post_processing/cache/` | Post-processing assets from AIC data |
| `tests/TrackEval/` | Third-party reference |
| `runs/*_2024.sh`, `runs/*_20260703.sh`, `runs/*_20220325.sh` | Dated snapshots of current files |

---

## 7. Configuration

### 7.1 Files

Configs live in `projects/tss/data/<dataset>/configs/`. `main_mtmc.py` resolves `--config` and
`--config_feat_merge` relative to that folder.

| File | Used by | Purpose |
|---|---|---|
| `<cam>.yaml` (`19_SUWON.yaml`, `46_SUWON.yaml`, `20_SUWON.yaml`) | per-camera stages | Full camera config |
| `reid/<cam>_<model>.yaml` | `--feature_extraction` | Same schema; selects one re-ID model and its output folder |
| `reid_merge.yaml` | feature merge, zone and scene matching, writing results | Camera **order** (road order, which drives the pairing in §6.6), re-ID models to merge, folder names |
| `default.yaml` | scene-level stages | Template camera config; also the default `--config_feat_merge` |

Related inputs: `class_labels.json`, `zone/`, `timestamp/`, and `<subset>/<cam>/<cam>_roi.jpg` (§8).

### 7.2 Camera YAML reference

| Key | Meaning | Notes |
|---|---|---|
| `dataset`, `subset`, `name` / `id_` | Dataset folder, scene folder, camera name | `--dataset` overrides `dataset` |
| `data.shape`, `data.frame_rate` | Frame size `[H, W, C]`, fps | Used for tracking and output videos |
| `data_loader.data`, `data_loader.ignore_region` | Video and ROI mask, relative to `<subset>/` | |
| `class_labels.file` | Class list JSON | Relative to the dataset folder |
| `detector.folder_out` | Subfolder name for detection outputs (`yolox`) | YOLOX takes its other settings from CLI args or the GUI dict |
| `reidentifier.reid_backbone`, `reid_model`, `reid_size_test` | Backbone, weights (relative to `projects/tss/models_zoo/`), input size | |
| `reidentifier.batch_size`, `folder_out` | Crops per batch; output subfolder of `dets_feat/` | |
| `tracker.name`, `min_confidence`, `nms_max_overlap`, `frame_rate`, `folder_out` | Tracker choice and pre-filters | `min_confidence` is 0.7 for cameras 19 and 20, 0.1 for 46 |
| `matching.dataset`, `zone`, `scene_name`, `timestamp_dir` | `ReidMatching` constructor arguments | |
| `data_writer.*` | Names of the output subfolders in `temps/` | |

Keys that are **currently ignored**:

- `reidentifier.use_ff`, `use_rerank`, `score_threshold`;
- `tracker.max_age`, `min_hit`, `iou_threshold`, `motion_model` (SORT-only);
- `tracker.min_box_area` (read, but its filter is commented out);
- the other `detector.*` keys;
- `data.file`, `data.stream`, `result_writer`.

### 7.3 Settings hard-coded in Python

| Setting | Location |
|---|---|
| Clustering thresholds (`dis_thrs` per camera pair; second-round distance 0.9) | `ReidMatching.get_labels` |
| Zone keep/drop rules; zone color thresholds | `ZoneIntra.is_ignore`, `ZoneIntra.get_zone` |
| `video_fps = 10.` for tracklet times (the videos are 15 fps) | `ReidMatching.merge_trajectory_zone` |
| 2000 px² minimum box area for tracklet mean features | `ReidMatching.merge_trajectory_zone` |
| Association thresholds 0.4 / 150 px / 0.7; feature EMA α 0.9 | `JDETracker.update`, `STrack` |
| Camera list, paths, display fps | Top of the GUI module (weights, conf and IoU come from each camera YAML's `detector` section) |
| Detection arguments (image size 1280, conf 0.1, GPU id) | `run_track_mtmc_button_detection.sh` |
| Class names | `class_labels.json` of the dataset, loaded by the GUI module. The CLI `detectors/detect.py` reads `opt.class_names`, which its parser never defines (known bug, out of scope) |

---

## 8. Data Layout

### 8.1 Dataset folder

```
projects/tss/data/tsd_suwon_19_46_20/
├── class_labels.json                 Classes: bus, small truck, truck (id, train_id, color)
├── configs/                          YAML configs (§7)
│   ├── 19_SUWON.yaml, 46_SUWON.yaml, 20_SUWON.yaml, default.yaml
│   ├── reid/<cam>_resnet101_ibn_a_2.yaml
│   └── reid_merge.yaml
├── clearsky_1764390903/              Scene ("subset"), one folder per camera
│   └── <cam>/
│       ├── <cam>.mp4                 Input video
│       ├── <cam>_roi.jpg             ROI mask: white = analyze, black = ignore
│       └── gt/                       Per-camera MOT ground truth (gt.txt, det.txt, seqinfo.ini)
├── zone/                             Zone annotations and masks (§8.2)
│   ├── <cam>.jpg, <cam>.json         Annotated frame + X-AnyLabeling polygons
│   ├── <cam>_zones.png               Color-coded zone mask, read by ZoneIntra
│   └── <cam>_roi.jpg                 ROI mask produced by the converter
├── timestamp/clearsky_1764390903.txt "<cam> <offset_seconds>" per camera (all 0 here)
├── groundtruths/gt.txt               MTMC ground truth (gt_ORI.txt = original version)
├── rmois/                            legacy ROI/MOI polygons for vehicle counting
├── temps/                          ★ Working directory: every artifact of the current pipeline
└── outputs/                          Artifacts of an earlier run (~12 GB) + evaluation output
```

Videos, images and pickles are git-ignored (`*.mp4`, `*.jpg`, `*.png`, `*.pkl` in `.gitignore`), so
datasets must be copied separately.

### 8.2 Adding a camera or a dataset

1. Put the video at `<subset>/<cam>/<cam>.mp4`.
2. In X-AnyLabeling, open a frame saved as `zone/<cam>.jpg`. Draw one polygon labelled `roi` and zone
   polygons labelled `1`–`4` (meanings in §6.5), then save `zone/<cam>.json`.
3. Set `FOLDER_ROOT` and `VIDEO_NAME` in the `__main__` block of
   `ultilities/convert_xanylabeling_to_zone_mtmc.py`, then run it. It opens an OpenCV window; press any key
   to continue. It writes `zone/<cam>.png` and `zone/<cam>_roi.jpg`.
4. Rename `zone/<cam>.png` to `zone/<cam>_zones.png`, the name `ZoneIntra` looks for. Copy
   `zone/<cam>_roi.jpg` to `<subset>/<cam>/<cam>_roi.jpg`; the two copies are identical in the bundled
   data.
5. Add a `<cam> <offset_seconds>` line to `timestamp/<subset>.txt`.
6. Copy and edit `configs/<cam>.yaml` and `configs/reid/<cam>_<model>.yaml`. Insert the camera into
   `reid_merge.yaml → camera` **in road order**.
7. Train or choose a YOLOX model and add its experiment file (§9.7).
8. Update the hard-coded camera lists: every `run_track_mtmc_button_*.sh` script, plus `detector_configs`
   and `mtmc_viz` in the GUI module. The GUI has three display panes.

---

## 9. Setup & Execution

Run every command from `MLKit/projects/tss`. All paths in the scripts, the configs and this section are
relative to it.

### 9.1 Prerequisites

- Linux with an NVIDIA GPU and a driver that supports CUDA 11.8 builds of PyTorch.
- Miniconda or Anaconda, with `conda` on `PATH` (the GUI launcher calls `conda activate etss-vehicereidentification`).
- A desktop session (X11 or Wayland) for the GUI and the zone converter.
- Model weights (§9.3) and a dataset folder (§8).

### 9.2 Environment

The working environment is the conda env `etss-vehicereidentification` (Python 3.10) plus the pip packages in `setup.sh`:

```bash
cd MLKit/projects/tss
conda create -n etss-vehicereidentification python=3.10 -y
conda activate etss-vehicereidentification
bash setup.sh                  # ends with `poetry install`, which installs this project (editable)
pip install scikit-learn scipy pandas matplotlib pyyaml tqdm timm   # imported by the code, not listed in setup.sh
```

- `poetry install` reads `pyproject.toml` and adds a `.pth` file to the env that puts this folder and
  `third_party/YOLOX` on `sys.path`. After that, `src` and `yolox` import from any working
  directory and from scripts in any sub-folder; no `PYTHONPATH` is needed.
- To reinstall only the project, rerun the Poetry lines of `setup.sh` here:
  `rm -rf poetry.lock; poetry install --extras "dev"; rm -rf poetry.lock`.
- The project name contains spaces, so pip can't find the install by name (`pip show` reports it as
  not found). To remove it, delete its `.pth` file and `.dist-info` folder from the env's site-packages.

Quick check:

```bash
python -c "import yolox, torch; from src import MTMCTrackingCamera; print(yolox.__file__, torch.cuda.is_available())"
```

`yolox.__file__` must be `…/projects/tss/third_party/YOLOX/yolox/__init__.py`.

### 9.3 Model weights

`models_zoo/` holds about 78 GB, mostly per-epoch YOLOX checkpoints. The current pipeline needs only:

| File | Used by |
|---|---|
| `models_zoo/yolox/yolox_x_19_suwon/best_ckpt.pth` | Detection, camera 19 |
| `models_zoo/yolox/yolox_x_46_suwon/best_ckpt.pth` | Detection, camera 46 |
| `models_zoo/yolox/yolox_x_20_suwon/best_ckpt.pth` | Detection, camera 20 |
| `models_zoo/pretrained/resnet/resnet101_ibn_a_2.pth` | Re-ID (`configs/reid/*_resnet101_ibn_a_2.yaml`) |

Each YOLOX checkpoint is 757 MB and the re-ID model is 178 MB.

### 9.4 Run the GUI (recommended)

```bash
cd MLKit/projects/tss
bash run_gui_vehicle_tracking.sh
```

Click the stage buttons in order (Image Extraction → … → Show Result), or click **Run All**. Detailed
logs and progress bars go to the terminal; the message pane shows the stage state.
`runs/run_track_mtmc_gui.sh` is an alternative that skips the conda activation and sets
`CUDA_LAUNCH_BLOCKING=1`.

### 9.5 Run from the command line

One stage at a time, for all cameras:

```bash
bash runs/run_track_mtmc_button_image_extraction.sh
bash runs/run_track_mtmc_button_detection.sh            # fix its paths first (§12, item 2)
bash runs/run_track_mtmc_button_reidentification.sh
bash runs/run_track_mtmc_button_single_camera_tracking.sh
bash runs/run_track_mtmc_button_inter_camera_tracking.sh
```

Or run everything with `bash runs/run_track_mtmc_cert.sh`, which has the same detection-path
caveat.

One stage for one camera:

```bash
python runs/main_mtmc.py --dataset tsd_suwon_19_46_20 --config 19_SUWON.yaml --tracking
```

`main_mtmc.py` flags:

| Flag | Stage | Scope |
|---|---|---|
| `--dataset <name>` | Dataset folder under `data/`. Defaults to the absent `aic22_mtmc`, so always pass it. | — |
| `--config <file>` | Camera config, relative to `configs/` | — |
| `--config_feat_merge <file>` | Merge/scene config; pass `reid_merge.yaml` | — |
| `--images_extraction` | 1 | camera |
| `--detection` | 2 via the registry; not usable with YOLOX (§12) | camera |
| `--feature_extraction` / `--feature_merge` | 3a / 3b | camera / scene |
| `--detection_filter`, `--use_dets_feat_filter` | Optional detection filter / track on filtered detections | camera |
| `--tracking`, `--tracking_postprocess` | 4a / optional post-processing | camera |
| `--matching_zone`, `--use_mots_feat_raw` | 4b; use raw rather than post-processed tracks | camera |
| `--matching_scene`, `--write_result` | 5a / 5b | scene |
| `--save_tracks_img`, `--save_tracks_zone`, `--save_mtmc_result`, `--save_dets_img`, `--save_dets_img_filter` | Debug and result videos | camera / scene |
| `--verbose` | Adds `verbose` to the config | — |

Standalone detection for one camera:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.detectors.detect \
  --name 19_SUWON \
  --source data/tsd_suwon_19_46_20/temps/images/19_SUWON \
  --output data/tsd_suwon_19_46_20/temps/ \
  --weights models_zoo/yolox/yolox_x_19_suwon/best_ckpt.pth \
  --exp_file third_party/YOLOX/exps/example/custom/yolox_x_19_suwon.py \
  --roi_region data/tsd_suwon_19_46_20/clearsky_1764390903/19_SUWON/19_SUWON_roi.jpg \
  --conf-thres 0.1 --img-size 1280 --save-txt --save-conf
```

Class-agnostic NMS is on by default (`--agnostic-nms`).

### 9.6 Evaluate

```bash
python tests/mot_evaluation/evaluate_mtmc.py \
  --gt data/tsd_suwon_19_46_20/groundtruths/gt.txt \
  --re data/tsd_suwon_19_46_20/temps/mtmc_result.txt \
  --ou data/tsd_suwon_19_46_20/outputs/track_mtmc.txt
```

This prints IDF1; the last stored value is 93.80. The evaluator also computes IDP, IDR, MOTA, MOTP and
MOTAL; switch to `print_metrics_ext` in `evaluate_tracking` to print them. For HOTA and the official
CLEAR/Identity metrics, use `tests/TrackEval`.

### 9.7 (Re)train a detector

```bash
python third_party/YOLOX/tools/train.py \
  -f third_party/YOLOX/exps/example/custom/yolox_x_19_suwon.py \
  -d 1 -b 8 -c <initial_weights.pth>
```

- **Fix `data_dir` first.** In the experiment file it points to a COCO-format folder on another machine
  (`/media/vsw-ws-05/...`). In `yolox_x_20_suwon.py` it points to the **19_SUWON** dataset (§12).
- **Install the weights.** Copy the resulting `best_ckpt.pth` to `models_zoo/yolox/<exp_name>/`.
- **Reading the logs.** `train_ann` and `val_ann` are the same file, so the "best AP" in `train_log.txt`
  is AP50:95 on the training data, not a held-out score. The stored values are 91.12 (19), 97.54 (20) and
  97.22 (46).

---

## 10. Conventions

- **Working directory:** always `projects/tss`. Scripts and configs use paths relative to it, and code
  imports `src.*` and `yolox` through the editable install (§9.2), never `projects.tss.*`.
- **Style:** first-party code uses **tab** indentation, `# MARK: -` section headers, and `NOTE:`,
  `DEBUG:`, `FIXME:` and `SUGAR:` comment tags. Vendored code keeps its upstream style. Some comments are
  in Vietnamese.
- **Docstrings:** Google style (`Args:` / `Returns:`). The team's style guides (PEP 8 / Google Python
  style, SOLID, docstring format) are in `.claude/skills/`.
- **Extensibility:** new components register in `src/builder.py` registries and are selected by `name`
  in YAML.
- **Stage contracts:** stages exchange data only through files in `temps/`. Keep the pickle schemas in
  §5.3 stable, or update every consumer.
- **Versioning:** by dated copies (`runs/*_2024.sh`, `runs/*_20260703.sh`); the undated file is the live one.
  MLKit itself is not a git repository, although it has a Python `.gitignore`; only
  `projects/tss/tests/mot_evaluation/` has its own `.git`.
- **Testing:** there are no unit tests. Check the install with the quick check in §9.2 and
  `python runs/main_mtmc.py --help`; check pipeline results with `evaluate_mtmc.py` and the debug videos.

---

## 11. Where to Look

| I want to… | Look at |
|---|---|
| Change stage order or add a stage | `MTMCTrackingCamera.run()` (`src/camera/mtmc_camera.py`), flags in `runs/main_mtmc.py`, the matching `run_track_mtmc_button_*.sh` |
| Tune detection | `run_track_mtmc_button_detection.sh`, the camera YAML's `detector` section (GUI), `src/detectors/yolox_adaptor.py` |
| Swap or ensemble re-ID models | `configs/reid/*.yaml`, `reid_merge.yaml → reidentifier`, `src/reidentifiers/baseline/model/backbones/` |
| Tune single-camera tracking | `JDETracker.update` in `src/trackers/fairmot/fm_tracker/multitracker.py`; `tracker.*` in the camera YAML |
| Change zone rules | `ZoneIntra.is_ignore` in `src/matchers/utils/zone_intra.py` |
| Tune cross-camera matching | `ReidMatching.get_labels` / `get_sim_matrix`, `src/matchers/utils/filter.py`, `utils/visual_rr.py` |
| Change the output videos | `src/io/mtmc_drawer.py` |
| Change the GUI | `src/views/mtmc.ui` (Qt Designer), `src/views/multicamera_multiobject_tracking_view.py` |
| Add a camera or dataset | §8.2 |
| Evaluate results | `tests/mot_evaluation/evaluate_mtmc.py`, `tests/TrackEval/` |
| Shared utilities (I/O, registries, bbox ops) | `src/factory/`, `src/file/`, `src/vision/`, … (§6.9) |

All `src/`, `runs/` and `tests/` paths above are under `projects/tss/`.

---

## 12. Known Issues & Gotchas

| # | Issue | Effect | Workaround |
|---|---|---|---|
| 1 | Fixed on 2026-11-04. `yolox` used to come from the external ETSS-02-VehicleDetTrack checkout; `poetry install` (§9.2) now resolves it to `projects/tss/third_party/YOLOX` | — | — |
| 2 | **`temps/` vs `outputs/`.** The camera code and the GUI use `temps/`, but `run_track_mtmc_button_detection.sh` reads images from and writes detections to `outputs/` | `run_track_mtmc_cert.sh` fails at stage 3 (no `temps/dets_crop_pkl/...`) | In the detection script, change `outputs/` to `temps/` in `--source` and `--output`. `outputs/` holds an earlier run. |
| 3 | No detector named `yolox` is registered, so `main_mtmc.py --detection` cannot build one | Crash on `None.detect` | Use `detectors/detect.py` or the GUI |
| 4 | Fixed on 2026-11-04. `setup.sh` and `pyproject.toml` live in `projects/tss/`, and the final `poetry install` succeeds there | — | — |
| 5 | `--dataset` defaults to `aic22_mtmc`, which isn't present | File-not-found errors | Always pass `--dataset` |
| 6 | Every `main_mtmc.py` call opens the camera's video and creates all output folders | The video must exist even for scene stages | — |
| 7 | The zone converter writes `<cam>.png`, but `ZoneIntra` reads `<cam>_zones.png` | `KeyError` in zone matching | Rename after converting (§8.2) |
| 8 | `yolox_x_20_suwon.py` sets `data_dir` to the 19_SUWON dataset; all experiment files use absolute paths from another machine | Retraining camera 20 would use camera 19's data | Fix before retraining |
| 9 | `FairMOT.get_merge_feature` tests `'class' in feat_dic` instead of `value` | Every track gets class 2. Matching is class-agnostic, so IDs are unaffected | Fix the condition if per-class output matters |
| 10 | Hard-coded settings (§7.3), including `video_fps = 10` against 15 fps videos | `io_time` is off by 1.5×; harmless while `st_filter` stays disabled | Move to config when needed |
| 11 | Legacy runner scripts call the missing `main_aic21_track_mtmc.py` and absent datasets (`aic22_mtmc`, `aLong`, `synthehicle_town05`) | They fail if run | Use the undated `run_track_mtmc_button_*.sh` scripts |
| 12 | Training and legacy modules import packages missing from the env (`motmetrics`, `natsort`, `pytorch_lightning`, `rfconv`, `torch2trt`, bare `utils.*`) | Import errors only when those modules are used | Install as needed |
| 13 | Modules under `src/` (`camera/*`, `reidentifiers/*`) and `runs/main_mtmc.py` import `src.configuration`, whose `load_config()` and `copy_config_file()` need `src.file` | A top-level `from src… import` in `src/configuration.py` risks an import cycle through `src/__init__.py` | Keep those imports inside the functions |
| 14 | The editable install puts the whole `projects/tss/` folder on `sys.path` for every program in `etss-vehicereidentification`, and Python puts a script's own folder in front of it | tss's top-level names (`src`, `tests`, `runs`, `data`) can shadow, or be shadowed by, same-named modules of other projects in the env or of a script's own folder | Start tss code from `projects/tss` through `runs/`, `src/views/` or `python -m` |
| 15 | Removing `src/nn/` and `MovingObject` left dangling imports: `src.nn.Phase` in `src/data/dataset/datamodule.py`, `src.objects.MovingObject` in `src/trackers/base.py` (`BaseTracker.__init__`) and `src/trackers/sort.py` | Those modules fail when imported or constructed; the FairMOT pipeline doesn't use them | Delete them, or restore the symbols if SORT is needed |
