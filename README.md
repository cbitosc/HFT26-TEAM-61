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

## 🎥 Demo Video
 https://drive.google.com/file/d/1onos3VTbAJm-jV3x7SusEGawjLUKxQAj/view?usp=drive_link

## 📸 Screenshots
<img width="1467" height="835" alt="Screenshot 2026-10-31 at 1 53 35 PM" src="https://github.com/user-attachments/assets/b612f119-16ac-4560-9968-160b758bff0d" />
<img width="1468" height="823" alt="Screenshot 2026-10-31 at 1 53 20 PM" src="https://github.com/user-attachments/assets/3c11e73c-e45d-46ed-b620-2a7f1797bc31" />
<img width="1450" height="709" alt="Screenshot 2026-10-31 at 1 52 55 PM" src="https://github.com/user-attachments/assets/6fe6d6be-ed0f-4129-a55a-2bf8b1d09ddd" />
<img width="1428" height="714" alt="Screenshot 2026-10-31 at 1 52 41 PM" src="https://github.com/user-attachments/assets/aff2ab8a-c3ba-4704-8095-53435f74cc57" />
<img width="1451" height="354" alt="Screenshot 2026-10-31 at 1 52 28 PM" src="https://github.com/user-attachments/assets/753f0953-59d0-4587-9061-aebfc6dddfe4" />
<img width="1466" height="821" alt="Screenshot 2026-10-31 at 1 52 16 PM" src="https://github.com/user-attachments/assets/2ae03e6a-11fc-4ca7-8fd2-4a552fbb3def" />
<img width="1466" height="835" alt="Screenshot 2026-10-31 at 1 51 51 PM" src="https://github.com/user-attachments/assets/117a3a0c-4ea3-4b39-a1dc-6a8dbfeba4cb" />







