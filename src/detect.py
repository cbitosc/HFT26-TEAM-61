"""
PrismTech Wildlife Monitor - detection + triage + event grouping
Run: python src/detect.py
Output: ai_predictions.csv
"""
import os
import re
import glob
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from injury import check_injury

# ---------------- CONFIG ----------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT_DIR / "demo_images"
OUT_CSV = ROOT_DIR / "ai_predictions.csv"
MODEL_NAME = "yolov8n.pt"
CONF_THRESH = 0.25

# COCO animal class ids (YOLO pretrained)
ANIMAL_IDS = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23}

# ---------------- HELPERS ----------------
def parse_event_key(path):
    """Group burst frames into one event."""
    name = os.path.basename(str(path))
    stem = os.path.splitext(name)[0]
    m = re.search(r"(c\d{3})", stem)
    if m:
        return m.group(1)
    m = re.match(r"(.+?)[_-]?\d{3,}$", stem)
    if m:
        return m.group(1)
    return os.path.dirname(str(path)) or "batch_0"

def blank_score(img_path, animal_dets, conf_thresh=CONF_THRESH):
    """
    Heuristic blank triage:
      no confident animal detection + low edge density + low variance => blank
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"is_blank": True, "reason": "unreadable",
                "edge_density": 0.0, "variance": 0.0, "n_conf_dets": 0}

    small = cv2.resize(img, (256, 256))
    edges = cv2.Canny(small, 50, 150)
    edge_density = float(edges.mean() / 255.0)
    variance = float(small.var() / 255.0)

    n_conf = sum(1 for d in animal_dets if d["confidence"] >= conf_thresh)
    is_blank = (n_conf == 0) and (edge_density < 0.02) and (variance < 0.05)

    if is_blank:
        reason = "empty"
    elif n_conf == 0:
        reason = "no_animal_detection"
    else:
        reason = "has_animal"

    return {"is_blank": bool(is_blank), "reason": reason,
            "edge_density": round(edge_density, 4),
            "variance": round(variance, 4), "n_conf_dets": n_conf}

# ---------------- MAIN ----------------
def main():
    model = YOLO(MODEL_NAME)
    
    # Injury API tracking
    api_call_count = 0
    has_api_key = bool(os.environ.get("FEATHERLESS_API_KEY"))

    if not DEMO_DIR.exists():
        print(f"[!] Demo directory not found: {DEMO_DIR}")
        return

    images = sorted(glob.glob(str(DEMO_DIR / "*.*")))
    images = [p for p in images if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]

    if not images:
        print(f"[!] No images found in {DEMO_DIR}/")
        return

    rows = []
    for img_path in images:
        results = model(img_path, conf=CONF_THRESH, verbose=False)
        animal_dets = []
        
        # Use relative path for CSV to keep it clean
        rel_img_path = os.path.relpath(img_path, ROOT_DIR)

        for r in results:
            names = r.names
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]

                row = {
                    "image_path": rel_img_path,
                    "class": names[cls_id],
                    "class_id": cls_id,
                    "confidence": round(conf, 4),
                    "x1": round(x1, 2), "y1": round(y1, 2),
                    "x2": round(x2, 2), "y2": round(y2, 2),
                    "injury_suspected": False,
                    "injury_note": ""
                }

                if cls_id in ANIMAL_IDS:
                    animal_dets.append({"confidence": conf, "class": names[cls_id]})
                    
                    # Injury check: only for animals, conf >= 0.25, up to 20 images, if API key exists
                    if conf >= 0.25 and has_api_key and api_call_count < 20:
                        injury_res = check_injury(img_path, (x1, y1, x2, y2), names[cls_id])
                        row["injury_suspected"] = injury_res["injury_suspected"]
                        row["injury_note"] = injury_res["reason"]
                        api_call_count += 1

                rows.append(row)

        # Blank triage and event grouping logic
        b = blank_score(img_path, animal_dets)
        b["event_key"] = parse_event_key(img_path)

        if rows and rows[-1]["image_path"] == rel_img_path:
            for k, v in b.items():
                rows[-1][k] = v
        else:
            # Blank image with zero detections
            rows.append({
                "image_path": rel_img_path, "class": "blank", "class_id": -1,
                "confidence": 0.0, "x1": 0, "y1": 0, "x2": 0, "y2": 0, 
                "injury_suspected": False, "injury_note": "",
                **b
            })

    df = pd.DataFrame(rows)

    # Backfill per-image fields for multi-detection rows
    per_img = df.groupby("image_path")[["is_blank", "reason", "edge_density",
                                        "variance", "n_conf_dets", "event_key"]].first()
    df = df.drop(columns=["is_blank", "reason", "edge_density",
                          "variance", "n_conf_dets", "event_key"],
                 errors="ignore").merge(per_img, left_on="image_path",
                                        right_index=True, how="left")

    df.to_csv(OUT_CSV, index=False)

    total_imgs = df["image_path"].nunique()
    blanks = df.drop_duplicates("image_path")["is_blank"].sum()
    events = df["event_key"].nunique()
    
    print(f"\n[OK] {len(df)} detection rows -> {OUT_CSV.name}")
    print(f"[OK] Images: {total_imgs} | Blanks filtered: {blanks} "
          f"({100*blanks/max(total_imgs,1):.1f}%) | Events: {events}")
    
    if has_api_key:
        print(f"[OK] Injury API checks performed: {api_call_count}")
    else:
        print("[!] FEATHERLESS_API_KEY not set. Injury checks skipped.")

if __name__ == "__main__":
    main()