import cv2
import json
import numpy as np
from ultralytics import YOLO

VIDEO_PATH = "data/videos/sample.mp4"
MODEL_PATH = "models/best.pt"

OUT_VIDEO = "outputs/videos/hen_output.mp4"
OUT_JSON = "outputs/json/hen_counts.json"

FRAME_SKIP = 5   # much faster than heavy tracking
DIST_THRESHOLD = 50  # pixels for centroid matching

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (w, h))

next_id = 1
tracks = {}
counts_over_time = []

def distance(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    if frame_id % FRAME_SKIP != 0:
        out.write(frame)
        continue

    time_sec = round(frame_id / fps, 2)

    results = model(frame, conf=0.4, verbose=False)[0]
    detections = []

    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        detections.append((x1, y1, x2, y2, (cx, cy)))

    new_tracks = {}

    for det in detections:
        x1, y1, x2, y2, centroid = det
        matched = False

        for tid, old_centroid in tracks.items():
            if distance(centroid, old_centroid) < DIST_THRESHOLD:
                new_tracks[tid] = centroid
                matched = True
                break

        if not matched:
            new_tracks[next_id] = centroid
            next_id += 1

    tracks = new_tracks
    count = len(tracks)

    total_weight = 0
    for (x1, y1, x2, y2, centroid) in detections:
        area = (x2 - x1) * (y2 - y1)
        total_weight += area / (w * h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    avg_weight = round(total_weight / count, 4) if count > 0 else 0

    counts_over_time.append({
        "time_sec": time_sec,
        "count": count,
        "avg_weight_index": avg_weight
    })

    cv2.putText(frame, f"Count: {count}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(frame, f"Weight Index: {avg_weight}", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    out.write(frame)

cap.release()
out.release()

with open(OUT_JSON, "w") as f:
    json.dump(counts_over_time, f, indent=2)

print("Saved:", OUT_VIDEO)
print("Saved:", OUT_JSON)
