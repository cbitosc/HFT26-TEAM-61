from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="PrismTech Wildlife Monitor", layout="wide")
st.title("🐾 PrismTech Wildlife Monitor")
st.markdown("**Team PrismTech** | AI-powered camera-trap analytics")
st.write("---")

# ---------- 1. Load AI predictions (written by detect.py) ----------
CSV_CANDIDATES = [
    Path(__file__).resolve().parent.parent / "ai_predictions.csv",
    Path("ai_predictions.csv"),
]


@st.cache_data
def load_data():
    for p in CSV_CANDIDATES:
        if p.exists():
            return pd.read_csv(p)
    return pd.DataFrame()


df = load_data()

if df.empty or "species" not in df.columns:
    st.warning("Data not found! Please run detect.py to create 'ai_predictions.csv'.")
else:
    # ---------- 2. Keep only wildlife classes ----------
    valid_animals = ["Sheep", "Bear", "Dog", "Cat", "Cow", "Horse", "Bird", "Elephant", "Zebra", "Giraffe"]
    df_animals = df[df["species"].isin(valid_animals)]

    loc_col = "Camera" if "Camera" in df.columns else "location"
    st.sidebar.header("Filters")
    selected_camera = st.sidebar.selectbox("Select Camera Location", df[loc_col].unique())
    filtered_df = df_animals[df_animals[loc_col] == selected_camera]

    # ---------- 3. Metrics ----------
    col1, col2 = st.columns(2)
    col1.metric("Total Valid Animal Sightings", len(filtered_df))
    col2.metric("Unique Species Detected", filtered_df["species"].nunique())

    # ---------- 4. Chart ----------
    st.write("### 📊 Species Distribution")
    species_counts = filtered_df["species"].value_counts().reset_index()
    species_counts.columns = ["species", "Count"]

    if not species_counts.empty:
        fig = px.bar(
            species_counts, x="species", y="Count", color="species",
            title=f"Wildlife Sightings at {selected_camera}",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No valid animals detected in this camera location.")

    # ---------- 5. Detections table ----------
    st.write("### 📸 Recent AI Detections")
    st.dataframe(filtered_df, use_container_width=True)

    # ---------- 6. Activity summary ----------
    st.write("---")
    st.write("### 🤖 Activity Summary")
    stats_text = f"{selected_camera} recorded {len(filtered_df)} animal sightings. Breakdown: "
    stats_text += ", ".join(f"{r['Count']} {r['species']}" for _, r in species_counts.iterrows())
    st.write("**Data ready for LLM:**", stats_text)

    if st.button("Generate Summary"):
        st.success(
            f"The camera trap system processed frames at {selected_camera} and logged "
            f"{len(filtered_df)} valid wildlife detections."
        )
        st.caption("Template-based summary. The LLM API is not connected yet.")

# ---------- 7. Live AI detection (runs the YOLOv8 model on an uploaded image) ----------
@st.cache_resource
def load_model():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")


st.write("---")
st.write("### Live AI Detection")
up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if up:
    from PIL import Image

    model = load_model()
    img = Image.open(up).convert("RGB")
    res = model(img, verbose=False)[0]
    st.image(res.plot()[:, :, ::-1], caption="YOLOv8 detections", use_container_width=True)
    for b in res.boxes:
        conf = float(b.conf[0])
        tag = " (Uncertain, low confidence)" if conf < 0.5 else ""
        st.write(f"{model.names[int(b.cls[0])]}: {conf:.2f}{tag}")
    if len(res.boxes) == 0:
        st.info("No animal detected: Empty frame")
