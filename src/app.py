"""
PrismTech Wildlife Monitor - Streamlit dashboard
Run: streamlit run src/app.py
"""
import os
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="PrismTech Wildlife Monitor", page_icon="🐾", layout="wide")

st.title("🐾 PrismTech Wildlife Monitor")
st.markdown("**Team PrismTech** | AI-powered camera-trap analytics")
st.write("---")

# 1. Load AI predictions
CSV_CANDIDATES = [
    Path(__file__).resolve().parent.parent / "ai_predictions.csv",
    Path("ai_predictions.csv")
]

CSV_PATH = next((p for p in CSV_CANDIDATES if p.exists()), None)

@st.cache_data
def load_data(path, mtime):
    df = pd.read_csv(path)
    # Ensure all required columns exist (handles old CSVs)
    for col, default in [
        ("is_blank", False), ("event_key", ""), ("n_conf_dets", 0), 
        ("review_status", ""), ("injury_suspected", False), ("injury_note", "")
    ]:
        if col not in df.columns:
            df[col] = default
            
    # Force correct data types to prevent errors
    df["is_blank"] = df["is_blank"].fillna(False).astype(bool)
    df["injury_suspected"] = df["injury_suspected"].fillna(False).astype(bool)
    df["review_status"] = df["review_status"].fillna("")
    return df

if not CSV_PATH:
    st.error("`ai_predictions.csv` not found. Run `python src/detect.py` first.")
    st.stop()

df = load_data(CSV_PATH, os.path.getmtime(CSV_PATH))

# Safety Check for Old CSV
if "image_path" not in df.columns:
    st.error("Purani CSV file mil rahi hai. Kripya terminal mein `python src/detect.py` chala kar nayi CSV generate karein.")
    st.stop()

# --- 2. Calculate Uncertainty ---
if "uncertain" not in df.columns:
    # A. Base uncertain: animal, not blank, conf 0.15 - 0.50
    is_animal = (df["class_id"] >= 14) & (df["class_id"] <= 23) & (~df["is_blank"])
    mask1 = is_animal & (df["confidence"] >= 0.15) & (df["confidence"] < 0.50)
    
    # B. Multi-class uncertain: same image, >1 animal class, conf diff < 0.15
    def check_multi_class(g):
        if g["class"].nunique() > 1:
            if (g["confidence"].max() - g["confidence"].min()) < 0.15:
                return True
        return False
        
    animal_df = df[is_animal]
    multi_class_flags = animal_df.groupby("image_path").apply(check_multi_class)
    mask2 = df["image_path"].map(multi_class_flags).fillna(False)
    
    df["uncertain"] = mask1 | mask2

# --- 3. Display Alerts ---
injury_rows = df[df["injury_suspected"] == True]
uncertain_count = len(df[df["uncertain"] == True])

if len(injury_rows) > 0:
    st.toast("Possible injury detected, please verify")
    st.error(f"🚨 **ALERT:** Possible injury detected in {len(injury_rows)} image(s), needs human verification. Check the Alerts tab.")
    
if uncertain_count > 0:
    st.warning(f"⚠️ **Uncertainty Alert:** {uncertain_count} detection(s) are uncertain. Please check the Review Queue or Alerts tab.")

# --- 4. Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Analytics", "🔍 Detections", "🧑‍🔬 Review Queue", "🚨 Alerts"])

# ---------------- TAB 1: ANALYTICS ----------------
with tab1:
    img_df = df.drop_duplicates("image_path")
    total_imgs = len(img_df)
    blanks = int(img_df["is_blank"].sum())
    events = img_df["event_key"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Images processed", total_imgs)
    c2.metric("Blank / unusable", f"{blanks}", f"{100*blanks/max(total_imgs,1):.0f}%")
    c3.metric("Distinct events", events)
    c4.metric("Review time saved", f"{100*blanks/max(total_imgs,1):.0f}%")

    st.markdown("#### Class distribution")
    animals_nonblank = df[(df["class_id"] >= 0) & (~df["is_blank"])]
    if len(animals_nonblank):
        st.bar_chart(animals_nonblank["class"].value_counts())
    else:
        st.info("No animal detections to plot.")

    st.markdown("#### Triage breakdown")
    st.dataframe(
        img_df["reason"].value_counts().rename_axis("reason").to_frame("images"),
        width='stretch'
    )

# ---------------- TAB 2: DETECTIONS ----------------
with tab2:
    conf = st.slider("Min confidence", 0.0, 1.0, 0.25, 0.05)
    show_blank = st.checkbox("Include blank images", value=False)

    view = df[df["confidence"] >= conf]
    if not show_blank:
        view = view[~view["is_blank"]]

    cols = [c for c in ["image_path", "class", "confidence", "event_key",
                        "n_conf_dets", "is_blank", "reason", "review_status", "injury_suspected"]
            if c in view.columns]
    
    st.write(f"**{len(view)}** detections found")
    st.dataframe(view[cols], width='stretch', height=400)

    st.download_button(
        "⬇️ Download filtered CSV",
        view[cols].to_csv(index=False).encode('utf-8'),
        "filtered_predictions.csv",
        "text/csv"
    )

# ---------------- TAB 3: REVIEW QUEUE ----------------
with tab3:
    st.markdown("Low-confidence detections (0.15–0.50) need human confirmation.")
    
    queue = df[(df["uncertain"] == True) & (df["review_status"] == "")]

    st.write(f"**{len(queue)}** detections in review queue")

    if len(queue):
        idx = st.selectbox(
            "Pick a detection to review",
            queue.index.tolist(),
            format_func=lambda i: f'{df.loc[i, "image_path"]} — {df.loc[i, "class"]} ({df.loc[i, "confidence"]:.2f})'
        )
        row = df.loc[idx]
        
        c_img, c_ctrl = st.columns([2, 1])
        with c_img:
            if os.path.exists(row["image_path"]):
                st.image(row["image_path"], caption=f'{row["class"]} @ {row["confidence"]:.2f}')
            else:
                st.warning(f"Image not found: {row['image_path']}")

        with c_ctrl:
            st.write(f'**Event:** `{row["event_key"]}`')
            st.write(f'**Status:** `{row["review_status"] or "pending"}`')
            
            if st.button("✅ Confirm"):
                df.loc[idx, "review_status"] = "confirmed"
                df.to_csv(CSV_PATH, index=False)
                st.cache_data.clear()
                st.rerun()
                
            if st.button("❌ Reject"):
                df.loc[idx, "review_status"] = "rejected"
                df.to_csv(CSV_PATH, index=False)
                st.cache_data.clear()
                st.rerun()
    else:
        st.success("Review queue empty! All good. 🎉")

# ---------------- TAB 4: ALERTS ----------------
with tab4:
    st.markdown("### 🚨 Flagged Detections (Possible Injury or Uncertain)")
    
    alert_df = df[(df["injury_suspected"] == True) | (df["uncertain"] == True)]
    
    if len(alert_df) == 0:
        st.success("No alerts! No uncertain or possible injuries detected.")
    else:
        for idx, row in alert_df.iterrows():
            with st.expander(f"{row['image_path']} — {row['class']} (Conf: {row['confidence']:.2f})"):
                if row["injury_suspected"]:
                    st.error(f"**Possible injury, needs human verification** | Note: {row.get('injury_note', '')}")
                if row["uncertain"]:
                    st.warning("Uncertain detection: Low confidence or conflicting classes.")
                
                # Check path before showing image
                if os.path.exists(row["image_path"]):
                    st.image(row["image_path"], caption=f"Class: {row['class']}")
                else:
                    st.text("Image file not found on disk.")