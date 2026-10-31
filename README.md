# 🐾 PrismTech Wildlife Monitor - AI Camera-Trap Analytics

**Team PrismTech** | AI-powered camera-trap analytics system designed to save researchers from manually reviewing thousands of images.

## 📖 Solution Overview
Camera traps generate massive amounts of data, but 60–80% of images are "blanks" (wind, grass, heat), and a single animal often triggers 10+ burst frames. 

Instead of just a basic object detector, our system acts as a complete **Analytics Workflow**:
- Runs YOLOv8n for animal detection.
- **Blank Triage:** Uses OpenCV heuristics (edge density + variance) to filter out empty images before human review.
- **Event Grouping:** Groups burst frames into unique "events" so researchers count animals, not photos.
- **Uncertainty Alerts:** Flags low-confidence or conflicting predictions.
- **Human-in-the-Loop:** A dedicated Review Queue where users can Confirm or Reject flagged detections, saving corrections back to the CSV.

## 🚀 Key Differentiators
1. **Analytics Tab:** Calculates "Review Time Saved" by filtering blanks.
2. **Detections Tab:** Interactive data table with confidence sliders and CSV export.
3. **Review Queue:** Routes low-confidence predictions (0.15–0.50) and conflicting classes to a human reviewer.
4. **Alerts Tab:** Highlights uncertain detections with visual thumbnails.

## 🛠 Architecture & Tech Stack
- **Frontend:** Streamlit (4 tabs: Analytics, Detections, Review Queue, Alerts)
- **Detection:** YOLOv8n (Ultralytics, pretrained)
- **Data Processing:** Pandas, OpenCV (for blank triage)
- **Language:** Python

## 💻 Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/cbitosc/HFT26-TEAM-61
   cd HFT26-TEAM-61
