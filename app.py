import sys
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from severity import estimate_severity

st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🛣️",
    layout="wide"
)

st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(120deg, #f4f8ff 0%, #edf8f3 100%);
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .hero-card {
            background: rgba(255,255,255,0.82);
            border: 1px solid rgba(18, 38, 63, 0.08);
            border-radius: 18px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
            margin-bottom: 1rem;
        }
        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            line-height: 1.2;
            margin: 0;
        }
        .hero-subtitle {
            color: #475569;
            font-size: 1rem;
            margin-top: 0.4rem;
        }
        .metric-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
            height: 100%;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            line-height: 1.1;
            margin: 0.25rem 0;
            color: #0f172a;
        }
        .metric-label {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #64748b;
        }
        .status-box {
            background: #f8fafc;
            border: 1px solid #dbeafe;
            border-radius: 14px;
            padding: 0.9rem 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">🛣️ Road Damage Detection & Severity Assessment</div>
        <div class="hero-subtitle">Upload a road image to detect cracks, potholes, and other surface damage.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Model settings")
    pothole_only = st.checkbox("Detect potholes only", value=True)
    weights = st.text_input(
        "Model weights",
        str(ROOT / "runs" / "road_damage_full" / "weights" / "best.pt"),
    )
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    st.caption("Tip: lower the threshold for small or faint damage. Higher values reduce false positives.")

uploaded = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

if uploaded is None:
    st.markdown(
        """
        <div class="status-box">
            <strong>No image uploaded yet.</strong><br>
            Use the sidebar to set the model path and confidence threshold, then upload a road photo to begin detection.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("The detector expects a trained YOLO weights file such as runs/road_damage/weights/best.pt.")
    st.stop()

if not Path(weights).exists():
    fallback_weights = ROOT / "runs" / "road_damage" / "weights" / "best.pt"
    if pothole_only and fallback_weights.exists():
        weights = str(fallback_weights)
    else:
        st.error(
            "No trained model was found. Prepare the data and train it with: python scripts/prepare_archive.py --source-dir train --output prepared; python scripts/train.py --data prepared/data.yaml"
        )
        st.stop()

@st.cache_resource
def load_model(path):
    return YOLO(path)

model = load_model(weights)

data = uploaded.read()
tmp = Path(tempfile.gettempdir()) / uploaded.name
tmp.write_bytes(data)

result = model.predict(
    source=str(tmp),
    conf=confidence,
    imgsz=640,
    verbose=False
)[0]

image = result.orig_img.copy()
h, w = image.shape[:2]
rows = []

for box in result.boxes:
    cls = int(box.cls.item())
    conf = float(box.conf.item())
    x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
    name = model.names[cls]

    if pothole_only and "pothole" not in name.lower():
        continue

    severity, score = estimate_severity(name, x1, y1, x2, y2, w, h)

    cv2.rectangle(
        image,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        (0, 255, 0),
        2
    )

    cv2.putText(
        image,
        f"{name} | {conf:.2f} | {severity}",
        (int(x1), max(25, int(y1) - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2
    )

    rows.append(
        {
            "Defect": name,
            "Confidence": round(conf, 3),
            "Severity": severity,
            "Severity score": score,
            "X1": round(x1),
            "Y1": round(y1),
            "X2": round(x2),
            "Y2": round(y2),
        }
    )

summary_df = pd.DataFrame(rows)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Detections</div>
            <div class="metric-value">{}</div>
        </div>
        """.format(len(rows)),
        unsafe_allow_html=True,
    )
with col2:
    if rows:
        top_class = summary_df["Defect"].value_counts().idxmax()
        top_count = summary_df["Defect"].value_counts().max()
        value_text = f"{top_class} ({top_count})"
    else:
        value_text = "None"
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Most common defect</div>
            <div class="metric-value">{}</div>
        </div>
        """.format(value_text),
        unsafe_allow_html=True,
    )
with col3:
    if rows:
        avg_conf = round(summary_df["Confidence"].mean(), 3)
        value_text = f"{avg_conf}"
    else:
        value_text = "0.000"
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Avg confidence</div>
            <div class="metric-value">{}</div>
        </div>
        """.format(value_text),
        unsafe_allow_html=True,
    )

left, right = st.columns([1.5, 1])

with left:
    st.subheader("Detected image")
    st.image(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        caption="Annotated road damage output",
        use_container_width=True,
    )

with right:
    st.subheader("Results")
    if rows:
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No damage detected above the selected confidence threshold.")

st.divider()
st.warning(
    "Severity is estimated using a heuristic because the supplied RDD2022 annotations do not contain Low/Medium/High ground-truth severity labels."
)
