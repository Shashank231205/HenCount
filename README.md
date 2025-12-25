HenCount — Poultry CCTV Bird Counting & Weight Estimation

Author: Shashank

Project Name: HenCount

Role Targeted: Machine Learning / Computer Vision Engineer

1. Project Overview

HenCount is a lightweight poultry CCTV analytics system that automatically processes a fixed-camera farm video and produces:

Bird count over time

Stable bird tracking using centroid matching

Visual bird weight estimation (size-based proxy)

Annotated output video

JSON time-series data

This system is designed to be:

Fast

Low-compute

Easy to run on standard laptops

It is ideal for practical poultry farm monitoring.

2. What This Repository Contains

This GitHub repository contains:

Full source code

Video inference pipeline

Tracking & counting logic

Setup and run instructions

Because GitHub does not allow large ML files, the model, videos, and outputs are hosted on Google Drive.

3. Required Files (Google Drive)

Download these from Google Drive and place them in the project as shown:

Content	Drive Link
```
YOLO poultry model (best.pt)	https://drive.google.com/drive/folders/1qLACmAt0lIt9hX3gF5CT_LSPLUZTsABJ?usp=sharing

Input CCTV video (sample.mp4)	https://drive.google.com/drive/folders/1zpj63HZl0yg82n_NY7qCE77bm4x3fdDQ?usp=sharing

Output video + JSON	https://drive.google.com/drive/folders/101k3KqgSr37qyAwg8IERm0Fw3teCUXI8?usp=sharing
```
Place them inside:

HenCount/models/
HenCount/data/videos/
HenCount/outputs/

4. Folder Structure
```
HenCount/
├── app/
├── models/            ← YOLO model (from Drive)
│   └── best.pt
├── data/
│   └── videos/
│       └── sample.mp4
├── outputs/
│   ├── videos/
│   │   └── hen_output.mp4
│   └── json/
│       └── hen_counts.json
├── lite_pipeline.py   ← Main inference pipeline
├── requirements.txt
└── README.md
```
6. How Detection Works

This system uses a poultry-trained YOLOv8 model (best.pt).

For each frame:

YOLO detects all birds

Bounding boxes are returned

This gives:

(x1, y1, x2, y2) for each bird

6. How Tracking Works (Centroid Matching)

Instead of heavy tracking models, HenCount uses centroid-based tracking.

For each detection:

The center of the box is computed

It is compared to previous frame centroids

If distance < threshold → same bird

Else → new bird ID

This allows:

Stable tracking

No double counting

Extremely fast execution

7. How Bird Count Is Computed

For every processed frame:

Bird Count = number of active centroids being tracked


This creates a time-series:

time → bird count

8. How Weight Is Estimated

True bird weight is not available in video.

So a visual proxy is used:

Weight Index = bounding box area / frame area


Larger birds → larger boxes → higher weight index

The system computes:

Average Weight Index per frame

9. Frame Skipping for Speed

To make this pipeline fast:

FRAME_SKIP = 5


Only every 5th frame is analyzed, which:

Reduces compute by 80%

Keeps temporal trends accurate

10. How to Run HenCount
Step 1 — Install dependencies
```
pip install -r requirements.txt
```
Step 2 — Run the pipeline
```
python lite_pipeline.py
```

This will generate:
```
outputs/videos/hen_output.mp4
outputs/json/hen_counts.json
```
11. Output Files
🎥 Annotated Video
```
outputs/videos/hen_output.mp4
```
Contains:

Green boxes on birds

Count overlay

Weight index overlay

📄 JSON Output
```
outputs/json/hen_counts.json
```
Example:
```
[
  {
    "time_sec": 0.2,
    "count": 4,
    "avg_weight_index": 0.0523
  },
  {
    "time_sec": 0.4,
    "count": 5,
    "avg_weight_index": 0.0611
  }
]
```
12. What the Evaluator Should Check

To verify this project:

Open hen_output.mp4

Open hen_counts.json

Observe stable counting and size-based weight trends

These confirm:

Detection

Tracking

Counting

Weight estimation

13. Final Summary

HenCount is a fast, practical, production-style poultry CCTV system that provides:

Poultry-specific YOLO detection

Lightweight centroid-based tracking

Accurate bird counting

Visual weight estimation

Clean time-series analytics

It is designed for real-world poultry farm deployment on low-resource machines.
