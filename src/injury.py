import os
import cv2
import base64
import json
import requests

# Change this to any Vision-Language Model supported by Featherless
MODEL_NAME = "llava-hf/llava-1.5-7b-hf" 
API_URL = "https://api.featherless.ai/v1/chat/completions"

def check_injury(image_path, box, class_name):
    api_key = os.environ.get("FEATHERLESS_API_KEY")
    if not api_key:
        return {"injury_suspected": False, "reason": "not checked (no key)"}
    
    try:
        # 1. Crop Image
        img = cv2.imread(str(image_path))
        if img is None:
            return {"injury_suspected": False, "reason": "not checked (unreadable)"}
            
        x1, y1, x2, y2 = map(int, box)
        h, w = img.shape[:2]
        # Ensure coordinates are within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return {"injury_suspected": False, "reason": "not checked (empty crop)"}
            
        # 2. Base64 Encode
        _, buffer = cv2.imencode('.jpg', crop)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        
        # 3. API Payload
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Analyze this image of a {class_name}. Look for visible signs of injury (like wounds, bleeding, broken limbs). Return ONLY a JSON object exactly like this: {{\"injury_suspected\": true/false, \"reason\": \"short text\"}}. If the image is unclear or you are not sure, return false."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.1
        }
        
        # 4. Call API with 20s timeout
        response = requests.post(API_URL, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        
        # 5. Parse JSON safely
        result_text = response.json()["choices"][0]["message"]["content"]
        
        # Extract JSON in case model returns markdown code blocks
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].strip()
            
        parsed = json.loads(result_text)
        return {
            "injury_suspected": bool(parsed.get("injury_suspected", False)),
            "reason": str(parsed.get("reason", "not checked"))
        }
        
    except Exception as e:
        # Never crash on error, timeout, or bad JSON
        return {"injury_suspected": False, "reason": "not checked (API error)"}