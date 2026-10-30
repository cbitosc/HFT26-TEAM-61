import os
import pandas as pd
from ultralytics import YOLO

# Load the pre-trained YOLOv8 nano model (lightweight and fast for Mac)
model = YOLO("yolov8n.pt") 

IMAGE_DIR = "demo_images"
OUTPUT_CSV = "ai_predictions.csv"

results_list = []

print("Starting AI detection...")

for filename in os.listdir(IMAGE_DIR):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
        
    filepath = os.path.join(IMAGE_DIR, filename)
    
    # Run AI inference on the image
    results = model(filepath, verbose=False)
    
    for result in results:
        boxes = result.boxes
        
        if len(boxes) == 0:
            # Mark as empty if the AI finds nothing
            results_list.append({
                "image": filename,
                "species": "Empty",
                "confidence": 1.00,
                "location": "Camera 1", 
                "hour": 12 
            })
        else:
            # Extract species and confidence for each detected object
            for box in boxes:
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                species_name = model.names[class_id]
                
                results_list.append({
                    "image": filename,
                    "species": species_name.capitalize(),
                    "confidence": round(conf, 2),
                    "location": "Camera 1",
                    "hour": 12
                })

df = pd.DataFrame(results_list)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Detection complete! Saved {len(df)} predictions to {OUTPUT_CSV}")