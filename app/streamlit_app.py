"""
IndicDoc AI - Streamlit Application
Deep Learning-Based Multilingual Indian Document Layout Understanding

Premium, Indic-inspired UI with full detection, evaluation, and model comparison.
"""

import streamlit as st
import numpy as np
import cv2
import json
import time
import os
import sys
from pathlib import Path
from PIL import Image
import io
import base64

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="IndicDoc AI",
    page_icon="assets/logo.jpg",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════
#  INDIC-INSPIRED DESIGN SYSTEM (CSS)
# ══════════════════════════════════════════════════════════════

INDIC_CSS = """
<style>
/* ─────── Google Fonts ─────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Playfair+Display:wght@500;600;700&display=swap');

/* ─────── CSS Custom Properties (Design Tokens) ─────── */
:root {
    --indigo: #1E2A5A;
    --indigo-light: #2a3d7a;
    --gold: #C89B3C;
    --gold-muted: #d4ad5a;
    --ivory: #F7F3EA;
    --ivory-warm: #FBF8F1;
    --terracotta: #8B4A3A;
    --terracotta-light: #a05e4e;
    --ink-dark: #111827;
    --ink-medium: #1f2937;
    --text-primary: #1a1a2e;
    --text-secondary: #4a4a6a;
    --text-muted: #7a7a9a;
    --border-subtle: rgba(30,42,90,0.08);
    --shadow-soft: 0 2px 12px rgba(30,42,90,0.06);
    --shadow-card: 0 4px 20px rgba(30,42,90,0.08);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --transition-fast: 0.2s cubic-bezier(0.4,0,0.2,1);
}

/* ─────── Global Overrides ─────── */
.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: var(--ivory) !important;
}

/* ─────── Sidebar ─────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--indigo) 0%, #152047 100%) !important;
    border-right: 1px solid rgba(200,155,60,0.2);
}
section[data-testid="stSidebar"] * {
    color: #e8e4d9 !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label {
    color: var(--gold-muted) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ─────── Headers ─────── */
.hero-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: var(--indigo);
    text-align: center;
    margin: 0;
    line-height: 1.15;
    letter-spacing: -0.02em;
}
.hero-title .gold { color: var(--gold); }

.hero-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 1.05rem;
    font-weight: 400;
    color: var(--text-secondary);
    text-align: center;
    margin: 0.4rem 0 1.2rem;
    letter-spacing: 0.01em;
}

.section-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.45rem;
    font-weight: 600;
    color: var(--indigo);
    margin: 1.5rem 0 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid var(--gold);
    display: inline-block;
}

/* ─────── Decorative Divider ─────── */
.indic-divider {
    text-align: center;
    margin: 0.6rem 0 1.2rem;
    color: var(--gold);
    font-size: 1.1rem;
    letter-spacing: 0.4em;
    opacity: 0.6;
}

/* ─────── Metric Cards ─────── */
.metric-card {
    background: white;
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 1.1rem 1rem;
    text-align: center;
    box-shadow: var(--shadow-soft);
    transition: transform var(--transition-fast), box-shadow var(--transition-fast);
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--indigo), var(--gold));
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-card);
}
.metric-value {
    font-family: 'Inter', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--indigo);
    line-height: 1.2;
}
.metric-label {
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 0.2rem;
}

/* ─────── Info Badges ─────── */
.lang-badge {
    display: inline-block;
    background: rgba(200,155,60,0.12);
    color: var(--gold);
    border: 1px solid rgba(200,155,60,0.25);
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 3px;
    transition: background var(--transition-fast);
}
.lang-badge:hover {
    background: rgba(200,155,60,0.22);
}

/* ─────── Result Box ─────── */
.result-panel {
    background: white;
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 1.2rem;
    box-shadow: var(--shadow-soft);
}

/* ─────── Detection Table ─────── */
.detection-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
}
.detection-table th {
    background: var(--indigo);
    color: white;
    padding: 8px 12px;
    text-align: left;
    font-weight: 600;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.detection-table td {
    padding: 7px 12px;
    border-bottom: 1px solid var(--border-subtle);
    color: var(--text-primary);
}
.detection-table tr:hover td {
    background: rgba(200,155,60,0.05);
}

/* ─────── Buttons ─────── */
.stButton>button {
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all var(--transition-fast) !important;
}
.stButton>button[kind="primary"],
.stButton>button[data-testid="stBaseButton-primary"] {
    background: var(--indigo) !important;
    border: none !important;
    color: white !important;
}
.stButton>button[kind="primary"]:hover,
.stButton>button[data-testid="stBaseButton-primary"]:hover {
    background: var(--indigo-light) !important;
    box-shadow: 0 4px 14px rgba(30,42,90,0.25) !important;
}

/* ─────── Tabs ─────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 2px solid var(--border-subtle);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    color: var(--text-muted);
    padding: 0.6rem 1.2rem;
    border-bottom: 3px solid transparent;
    transition: all var(--transition-fast);
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--indigo);
}
.stTabs [aria-selected="true"] {
    color: var(--indigo) !important;
    border-bottom-color: var(--gold) !important;
    font-weight: 600;
}

/* ─────── File Uploader ─────── */
.stFileUploader {
    border: 2px dashed rgba(30,42,90,0.15) !important;
    border-radius: var(--radius-md) !important;
    background: rgba(247,243,234,0.5) !important;
    transition: border-color var(--transition-fast);
}
.stFileUploader:hover {
    border-color: var(--gold) !important;
}

/* ─────── Warning / Info boxes ─────── */
.demo-banner {
    background: linear-gradient(135deg, rgba(200,155,60,0.08), rgba(139,74,58,0.06));
    border-left: 4px solid var(--gold);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 0.8rem 1rem;
    font-size: 0.88rem;
    color: var(--text-secondary);
    margin: 0.5rem 0;
}

/* ─────── Footer ─────── */
.app-footer {
    text-align: center;
    padding: 1.5rem 0 1rem;
    color: var(--text-muted);
    font-size: 0.78rem;
    border-top: 1px solid var(--border-subtle);
    margin-top: 2rem;
}
.app-footer .gold { color: var(--gold); }

/* ─────── Comparison Table ─────── */
.comparison-header {
    background: var(--indigo);
    color: white;
    padding: 0.6rem 1rem;
    border-radius: var(--radius-sm) var(--radius-sm) 0 0;
    font-weight: 600;
    font-size: 0.9rem;
}

/* ─────── About Card ─────── */
.about-card {
    background: white;
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 1.5rem;
    box-shadow: var(--shadow-soft);
    margin-bottom: 1rem;
}
.about-card h4 {
    color: var(--indigo);
    font-family: 'Playfair Display', serif;
    margin-bottom: 0.5rem;
}

/* ─────── Sidebar Footer ─────── */
.sidebar-footer {
    text-align: center;
    font-size: 0.72rem;
    color: rgba(232,228,217,0.5);
    padding: 1rem 0;
    border-top: 1px solid rgba(200,155,60,0.15);
    margin-top: 1rem;
}

/* ─────── Responsive ─────── */
@media (max-width: 768px) {
    .hero-title { font-size: 1.8rem; }
    .hero-subtitle { font-size: 0.9rem; }
    .metric-value { font-size: 1.3rem; }
}
</style>
"""

st.markdown(INDIC_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  CONSTANTS & UTILITIES
# ══════════════════════════════════════════════════════════════

DEFAULT_CLASSES = [
    'Paragraph', 'Image', 'Table', 'Header', 'Footer',
    'Title', 'Caption', 'Page Number', 'Footnote', 'List',
    'Figure', 'Equation', 'Logo', 'Stamp', 'Signature',
    'Handwriting', 'Chart', 'Map', 'Separator', 'Advertisement',
    'Watermark', 'Background', 'Margin Note', 'Column',
    'Section Header', 'Sub Header', 'Abstract', 'Author',
    'Affiliation', 'Date', 'Reference', 'Acknowledgment',
    'Appendix', 'Table of Contents', 'Index', 'Glossary',
    'Bibliography', 'Preface', 'Dedication', 'Colophon',
    'Running Header', 'Running Footer'
]

# Color palette for detection boxes
def get_class_colors(n):
    np.random.seed(42)
    colors = []
    for i in range(n):
        hue = int(180 * i / n)
        c = cv2.cvtColor(np.array([[[hue, 180, 220]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(x) for x in c))
    return colors

COLORS = get_class_colors(len(DEFAULT_CLASSES))


def draw_detections(image, boxes, labels, scores, class_names, conf_threshold=0.5):
    img = image.copy()
    h, w = img.shape[:2]
    for box, label, score in zip(boxes, labels, scores):
        if score < conf_threshold:
            continue
        idx = int(label) if int(label) < len(class_names) else 0
        color = COLORS[idx % len(COLORS)]
        name = class_names[idx] if idx < len(class_names) else f"cls_{idx}"
        x1, y1, x2, y2 = [int(c) for c in box[:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        thick = max(2, min(h, w) // 350)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        text = f"{name}: {score:.2f}"
        fs = max(0.38, min(h, w) / 1600)
        (tw, th_t), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        cv2.rectangle(img, (x1, y1 - th_t - bl - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, y1 - bl - 3), cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), 1, cv2.LINE_AA)
    return img


def load_metrics(metrics_dir):
    metrics = {}
    if metrics_dir.exists():
        for f in metrics_dir.glob("*.json"):
            try:
                with open(f) as fp:
                    metrics[f.stem] = json.load(fp)
            except Exception:
                pass
    return metrics


def run_inference_frcnn(image, model, device, conf_threshold):
    import torch
    from torchvision import transforms as T
    model.eval()
    
    # Prevent OOM on massive images by capping max dimension
    max_dim = 1024
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image_resized = cv2.resize(image, (int(w * scale), int(h * scale)))
    else:
        image_resized = image

    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)).to(device)
    start = time.time()
    with torch.no_grad():
        predictions = model([img_tensor])
    inference_time = time.time() - start
    pred = predictions[0]
    mask = pred['scores'] >= conf_threshold
    
    boxes = pred['boxes'][mask].cpu().numpy()
    
    # Scale boxes back if we resized
    if max(h, w) > max_dim:
        boxes = boxes / scale
        
    return {
        'boxes': boxes,
        'labels': pred['labels'][mask].cpu().numpy(),
        'scores': pred['scores'][mask].cpu().numpy(),
        'inference_time': inference_time
    }


def run_inference_yolo(image, model, conf_threshold):
    start = time.time()
    results = model(image, conf=conf_threshold, verbose=False)
    inference_time = time.time() - start
    result = results[0]
    boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) > 0 else np.array([])
    labels = result.boxes.cls.cpu().numpy() if len(result.boxes) > 0 else np.array([])
    scores = result.boxes.conf.cpu().numpy() if len(result.boxes) > 0 else np.array([])
    return {'boxes': boxes, 'labels': labels, 'scores': scores, 'inference_time': inference_time}


def demo_inference(image):
    h, w = image.shape[:2]
    np.random.seed(42)
    n = np.random.randint(4, 9)
    boxes, labels, scores = [], [], []
    for _ in range(n):
        bw = np.random.randint(w//6, w//2)
        bh = np.random.randint(h//8, h//3)
        x1 = np.random.randint(0, max(1, w-bw))
        y1 = np.random.randint(0, max(1, h-bh))
        boxes.append([x1, y1, x1+bw, y1+bh])
        labels.append(np.random.randint(0, min(10, len(DEFAULT_CLASSES))))
        scores.append(round(np.random.uniform(0.55, 0.97), 2))
    return {
        'boxes': np.array(boxes), 'labels': np.array(labels),
        'scores': np.array(scores), 'inference_time': 0.0, 'is_demo': True
    }


def metric_card_html(value, label, icon=""):
    return f"""
    <div class="metric-card">
        <div class="metric-value">{icon} {value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo
    logo_path = Path(__file__).parent / "assets" / "logo.jpg"
    if logo_path.exists():
        st.image(str(logo_path), use_container_width=True)
    else:
        st.markdown('<div style="text-align:center; font-size:1.5rem; font-weight:700; color:#C89B3C; margin:1rem 0;">IndicDoc AI</div>', unsafe_allow_html=True)

    st.markdown("---")

    model_choice = st.selectbox(
        "MODEL",
        ["Demo (No Model)", "Faster R-CNN (Baseline)", "YOLOv8 (Proposed)"],
        index=0,
        help="Select the model for inference"
    )

    conf_threshold = st.slider(
        "CONFIDENCE THRESHOLD",
        min_value=0.1, max_value=0.95, value=0.5, step=0.05,
    )

    st.markdown("---")
    st.markdown('<p style="font-size:0.72rem; color:rgba(200,155,60,0.7); font-weight:600; letter-spacing:0.08em;">DATASET</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.82rem; line-height:1.6;'>
    <b>IndicDLP</b> &mdash; AIKosh<br>
    119,806 images<br>
    12 languages &bull; 42 classes<br>
    COCO-format annotations
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="font-size:0.72rem; color:rgba(200,155,60,0.7); font-weight:600; letter-spacing:0.08em;">LANGUAGES</p>', unsafe_allow_html=True)
    languages = ["Assamese", "Bengali", "English", "Gujarati", "Hindi", "Kannada",
                 "Malayalam", "Marathi", "Odia", "Punjabi", "Tamil", "Telugu"]
    badge_html = "".join(f'<span class="lang-badge">{l}</span>' for l in languages)
    st.markdown(badge_html, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-footer">
        IndicDoc AI v1.0<br>
        B.E. AI&ML Deep Learning Project<br>
        SDG 9: Industry, Innovation & Infrastructure
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  HERO HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center; padding:0.8rem 0 0.3rem;">
    <div class="hero-title">Indic<span class="gold">Doc</span> AI</div>
    <div class="hero-subtitle">Understand Indian Documents. Digitally.</div>
    <div class="indic-divider">&loz; &loz; &loz;</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  NAVIGATION TABS
# ══════════════════════════════════════════════════════════════
tab_dash, tab_analyze, tab_insights, tab_eval, tab_about = st.tabs([
    "Dashboard",
    "Analyze",
    "Model Insights",
    "Evaluation",
    "About",
])


# ══════════════════════════════════════════════════════════════
#  TAB 1: DASHBOARD
# ══════════════════════════════════════════════════════════════
with tab_dash:
    # Metric row
    metrics_dir = PROJECT_ROOT / "outputs" / "metrics"
    saved_metrics = load_metrics(metrics_dir)
    baseline_m = saved_metrics.get("baseline_metrics", {})
    proposed_m = saved_metrics.get("proposed_metrics", {})

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(metric_card_html(
            baseline_m.get("total_predictions", "N/A"), "Total Predictions"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card_html(
            f"{baseline_m.get('precision', 'N/A')}", "Precision"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card_html(
            f"{baseline_m.get('recall', 'N/A')}", "Recall"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card_html(
            f"{baseline_m.get('f1', 'N/A')}", "F1-Score"), unsafe_allow_html=True)
    with c5:
        st.markdown(metric_card_html(
            f"{baseline_m.get('inference_time', 'N/A')}s", "Inference Time"), unsafe_allow_html=True)

    st.markdown("")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-title">Model Status</div>', unsafe_allow_html=True)

        model_info = {
            "Property": ["Current Model", "Architecture", "Dataset", "Classes", "Image Size", "Status"],
            "Baseline (Faster R-CNN)": [
                "Faster R-CNN",
                "ResNet-50 + FPN",
                "IndicDLP (Sample)",
                "42 classes + background",
                "640 x 640",
                "Trained" if (PROJECT_ROOT / "outputs" / "models" / "baseline_frcnn_best.pth").exists() else "Not trained",
            ],
            "Proposed (YOLOv8)": [
                "YOLOv8s",
                "CSPDarknet",
                "IndicDLP (Sample)",
                "42 classes",
                "640 x 640",
                "Trained" if (PROJECT_ROOT / "outputs" / "models" / "yolov8_best.pt").exists() else "Not trained",
            ],
        }
        st.dataframe(model_info, use_container_width=True, hide_index=True)

    with col_right:
        st.markdown('<div class="section-title">Latest Evaluation</div>', unsafe_allow_html=True)

        if baseline_m:
            eval_data = {
                "Metric": ["Precision", "Recall", "F1", "mAP@50", "Avg IoU"],
                "Value": [
                    baseline_m.get("precision", "N/A"),
                    baseline_m.get("recall", "N/A"),
                    baseline_m.get("f1", "N/A"),
                    baseline_m.get("map50", "N/A"),
                    baseline_m.get("avg_iou", "N/A"),
                ],
            }
            st.dataframe(eval_data, use_container_width=True, hide_index=True)
            if baseline_m.get("note"):
                st.caption(baseline_m["note"])
        else:
            st.info("No evaluation metrics available yet. Run training first.")

    # Show training plots if available
    plots_dir = PROJECT_ROOT / "outputs" / "plots"
    loss_plot = plots_dir / "training_loss.png"
    comp_plot = plots_dir / "model_comparison.png"

    if loss_plot.exists() or comp_plot.exists():
        st.markdown('<div class="section-title">Training Overview</div>', unsafe_allow_html=True)
        pc1, pc2 = st.columns(2)
        with pc1:
            if loss_plot.exists():
                st.image(str(loss_plot), caption="Training & Validation Loss", use_container_width=True)
        with pc2:
            if comp_plot.exists():
                st.image(str(comp_plot), caption="Model Performance Comparison", use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  TAB 2: ANALYZE
# ══════════════════════════════════════════════════════════════
with tab_analyze:
    col_upload, col_view, col_result = st.columns([1, 2, 1])

    with col_upload:
        st.markdown('<div class="section-title">Upload</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a document image",
            type=["png", "jpg", "jpeg", "tiff", "bmp"],
            help="Upload a scanned document image for layout analysis"
        )

        sample_dir = PROJECT_ROOT / "data" / "processed" / "sample" / "images"
        sample_images = sorted(list(sample_dir.glob("*.png")) + list(sample_dir.glob("*.jpg"))) if sample_dir.exists() else []

        if sample_images:
            use_sample = st.checkbox("Use a sample image")
            if use_sample:
                sample_choice = st.selectbox("Sample", [s.name for s in sample_images[:10]])
                uploaded_file = sample_dir / sample_choice

        if uploaded_file is not None:
            if isinstance(uploaded_file, Path):
                image = cv2.imread(str(uploaded_file))
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            h, w = image.shape[:2]
            st.markdown(f"**Dimensions:** {w} x {h} px")
            run_btn = st.button("Analyze Document", type="primary", use_container_width=True)
            reset_btn = st.button("Reset", use_container_width=True)
        else:
            run_btn = False
            reset_btn = False

    with col_view:
        st.markdown('<div class="section-title">Document Viewer</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            view_mode = st.radio("View", ["Original", "Prediction", "Comparison"], horizontal=True, label_visibility="collapsed")

            if run_btn or st.session_state.get("last_results") is not None:
                # Run inference
                if run_btn:
                    with st.spinner("Analyzing document..."):
                        if model_choice == "Faster R-CNN (Baseline)":
                            mp = PROJECT_ROOT / "outputs" / "models" / "baseline_frcnn_best.pth"
                            if mp.exists():
                                try:
                                    import torch
                                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                                    from ml.models.baseline_frcnn import get_baseline_model
                                    model = get_baseline_model(len(DEFAULT_CLASSES) + 1, pretrained=False)
                                    model.load_state_dict(torch.load(str(mp), map_location=device, weights_only=True))
                                    model.to(device)
                                    results = run_inference_frcnn(image, model, device, conf_threshold)
                                except Exception as e:
                                    st.warning(f"Model loading failed: {e}")
                                    results = demo_inference(image)
                            else:
                                results = demo_inference(image)
                        elif model_choice == "YOLOv8 (Proposed)":
                            mp = PROJECT_ROOT / "outputs" / "models" / "yolov8_best.pt"
                            if mp.exists():
                                try:
                                    from ultralytics import YOLO
                                    model = YOLO(str(mp))
                                    results = run_inference_yolo(image, model, conf_threshold)
                                except Exception as e:
                                    st.warning(f"Model loading failed: {e}")
                                    results = demo_inference(image)
                            else:
                                results = demo_inference(image)
                        else:
                            results = demo_inference(image)

                    st.session_state["last_results"] = results

                results = st.session_state.get("last_results")
                if results:
                    if results.get("is_demo"):
                        st.markdown('<div class="demo-banner"><b>Demo Mode</b> &mdash; Showing synthetic detections. Train a model for real predictions.</div>', unsafe_allow_html=True)

                    annotated = draw_detections(image_rgb.copy(), results["boxes"], results["labels"], results["scores"], DEFAULT_CLASSES, conf_threshold)

                    if view_mode == "Original":
                        st.image(image_rgb, caption="Original Document", use_container_width=True)
                    elif view_mode == "Prediction":
                        st.image(annotated, caption="Detected Layout Elements", use_container_width=True)
                    else:
                        c_l, c_r = st.columns(2)
                        with c_l:
                            st.image(image_rgb, caption="Original", use_container_width=True)
                        with c_r:
                            st.image(annotated, caption="Prediction", use_container_width=True)
            else:
                st.image(image_rgb, caption="Original Document", use_container_width=True)
        else:
            st.markdown("""
            <div class="result-panel" style="text-align:center; padding:4rem 2rem; color:var(--text-muted);">
                <div style="font-size:3rem; margin-bottom:0.5rem;">&#x1F4C4;</div>
                <div style="font-size:1.1rem; font-weight:500;">Upload a document image to begin analysis</div>
                <div style="font-size:0.85rem; margin-top:0.3rem;">Supports PNG, JPG, TIFF, BMP formats</div>
            </div>
            """, unsafe_allow_html=True)

    with col_result:
        st.markdown('<div class="section-title">Analysis</div>', unsafe_allow_html=True)
        results = st.session_state.get("last_results")

        if results and uploaded_file is not None:
            n_det = len(results.get("boxes", []))
            avg_conf = float(np.mean(results["scores"])) if n_det > 0 else 0.0
            inf_time = results.get("inference_time", 0)

            st.markdown(metric_card_html(str(n_det), "Elements Detected"), unsafe_allow_html=True)
            st.markdown(metric_card_html(f"{avg_conf:.2f}", "Avg Confidence"), unsafe_allow_html=True)
            st.markdown(metric_card_html(f"{inf_time:.3f}s", "Inference Time"), unsafe_allow_html=True)
            st.markdown(metric_card_html(model_choice.split("(")[0].strip(), "Model Used"), unsafe_allow_html=True)

            if n_det > 0:
                st.markdown("**Detected Elements:**")
                for i, (box, label, score) in enumerate(zip(results["boxes"], results["labels"], results["scores"])):
                    if score < conf_threshold:
                        continue
                    idx = int(label) if int(label) < len(DEFAULT_CLASSES) else 0
                    cls = DEFAULT_CLASSES[idx]
                    st.markdown(f"`{i+1}.` **{cls}** &mdash; {score:.2f}")

            # Download
            if n_det > 0:
                annotated_bgr = cv2.cvtColor(
                    draw_detections(image_rgb.copy(), results["boxes"], results["labels"], results["scores"], DEFAULT_CLASSES, conf_threshold),
                    cv2.COLOR_RGB2BGR
                )
                _, buf = cv2.imencode('.png', annotated_bgr)
                st.download_button("Download Result", data=buf.tobytes(), file_name="indicdoc_result.png", mime="image/png", use_container_width=True)
        else:
            st.markdown("""
            <div style="text-align:center; color:var(--text-muted); padding:2rem 0;">
                <p>Results will appear here after analysis.</p>
            </div>
            """, unsafe_allow_html=True)

    if reset_btn:
        st.session_state.pop("last_results", None)
        st.rerun()


# ══════════════════════════════════════════════════════════════
#  TAB 3: MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════
with tab_insights:
    st.markdown('<div class="section-title">Baseline vs Proposed Model</div>', unsafe_allow_html=True)

    metrics_dir = PROJECT_ROOT / "outputs" / "metrics"
    saved = load_metrics(metrics_dir)
    bm = saved.get("baseline_metrics", {})
    pm = saved.get("proposed_metrics", {})

    if bm or pm:
        comp = {
            "Metric": ["Precision", "Recall", "F1-Score", "mAP@50", "mAP@50:95", "Inference Time (s)", "Model Size (MB)"],
            "Faster R-CNN (Baseline)": [
                bm.get("precision", "N/A"), bm.get("recall", "N/A"), bm.get("f1", "N/A"),
                bm.get("map50", "N/A"), bm.get("map50_95", "N/A"),
                bm.get("inference_time", "N/A"), bm.get("model_size_mb", "N/A"),
            ],
            "YOLOv8 (Proposed)": [
                pm.get("precision", "N/A"), pm.get("recall", "N/A"), pm.get("f1", "N/A"),
                pm.get("map50", "N/A"), pm.get("map50_95", "N/A"),
                pm.get("inference_time", "N/A"), pm.get("model_size_mb", "N/A"),
            ],
        }
        st.dataframe(comp, use_container_width=True, hide_index=True)

        if bm.get("note"):
            st.caption(f"Baseline: {bm['note']}")
        if pm.get("note"):
            st.caption(f"Proposed: {pm['note']}")
    else:
        st.info("No evaluation metrics available yet. Train both models to see comparison.")

    # Architecture cards
    st.markdown('<div class="section-title">Architecture Details</div>', unsafe_allow_html=True)
    ac1, ac2 = st.columns(2)
    with ac1:
        st.markdown("""
        <div class="about-card">
            <h4>Faster R-CNN (Baseline)</h4>
            <table style="width:100%; font-size:0.85rem;">
                <tr><td><b>Backbone</b></td><td>ResNet-50 + FPN</td></tr>
                <tr><td><b>Type</b></td><td>Two-stage detector</td></tr>
                <tr><td><b>Pretrained</b></td><td>COCO (torchvision)</td></tr>
                <tr><td><b>Parameters</b></td><td>~41.8M</td></tr>
                <tr><td><b>Optimizer</b></td><td>SGD (lr=0.005, momentum=0.9)</td></tr>
                <tr><td><b>Scheduler</b></td><td>StepLR (step=3, gamma=0.1)</td></tr>
                <tr><td><b>Augmentation</b></td><td>HFlip, ColorJitter, Normalize</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    with ac2:
        st.markdown("""
        <div class="about-card">
            <h4>YOLOv8s (Proposed)</h4>
            <table style="width:100%; font-size:0.85rem;">
                <tr><td><b>Backbone</b></td><td>CSPDarknet</td></tr>
                <tr><td><b>Type</b></td><td>Single-stage detector</td></tr>
                <tr><td><b>Pretrained</b></td><td>COCO (ultralytics)</td></tr>
                <tr><td><b>Parameters</b></td><td>~11.2M</td></tr>
                <tr><td><b>Optimizer</b></td><td>SGD (ultralytics auto)</td></tr>
                <tr><td><b>Scheduler</b></td><td>Cosine annealing</td></tr>
                <tr><td><b>Augmentation</b></td><td>Mosaic, MixUp, HSV, Flip</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # Training curves & visualizations
    plots_dir = PROJECT_ROOT / "outputs" / "plots"
    st.markdown('<div class="section-title">Training Curves</div>', unsafe_allow_html=True)

    tc1, tc2 = st.columns(2)
    with tc1:
        lp = plots_dir / "training_loss.png"
        if lp.exists():
            st.image(str(lp), caption="Training & Validation Loss", use_container_width=True)
        else:
            st.info("Training loss plot will appear after training.")
    with tc2:
        pr = plots_dir / "precision_recall.png"
        if pr.exists():
            st.image(str(pr), caption="Precision-Recall Curve", use_container_width=True)
        else:
            st.info("PR curve will appear after evaluation.")

    # Sample predictions
    pred_dir = PROJECT_ROOT / "outputs" / "predictions"
    pred_images = sorted(list(pred_dir.glob("prediction_*.png"))) if pred_dir.exists() else []
    if pred_images:
        st.markdown('<div class="section-title">Sample Predictions</div>', unsafe_allow_html=True)
        pc = st.columns(min(3, len(pred_images)))
        for i, pi in enumerate(pred_images[:6]):
            with pc[i % len(pc)]:
                st.image(str(pi), caption=pi.stem.replace("_", " ").title(), use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  TAB 4: EVALUATION
# ══════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown('<div class="section-title">Evaluation Dashboard</div>', unsafe_allow_html=True)

    metrics_dir = PROJECT_ROOT / "outputs" / "metrics"
    saved = load_metrics(metrics_dir)
    bm = saved.get("baseline_metrics", {})

    if bm:
        ec1, ec2, ec3, ec4, ec5, ec6 = st.columns(6)
        with ec1: st.markdown(metric_card_html(f"{bm.get('precision', 'N/A')}", "Precision"), unsafe_allow_html=True)
        with ec2: st.markdown(metric_card_html(f"{bm.get('recall', 'N/A')}", "Recall"), unsafe_allow_html=True)
        with ec3: st.markdown(metric_card_html(f"{bm.get('f1', 'N/A')}", "F1-Score"), unsafe_allow_html=True)
        with ec4: st.markdown(metric_card_html(f"{bm.get('map50', 'N/A')}", "mAP@50"), unsafe_allow_html=True)
        with ec5: st.markdown(metric_card_html(f"{bm.get('avg_iou', 'N/A')}", "Avg IoU"), unsafe_allow_html=True)
        with ec6: st.markdown(metric_card_html(f"{bm.get('inference_time', 'N/A')}s", "Latency"), unsafe_allow_html=True)
    else:
        st.info("No evaluation metrics available yet.")

    plots_dir = PROJECT_ROOT / "outputs" / "plots"

    # Charts row
    ev1, ev2 = st.columns(2)
    with ev1:
        bm_plot = plots_dir / "baseline_metrics.png"
        if bm_plot.exists():
            st.image(str(bm_plot), caption="Baseline Metrics", use_container_width=True)
    with ev2:
        cm_plot = plots_dir / "confusion_matrix.png"
        if cm_plot.exists():
            st.image(str(cm_plot), caption="Confusion Matrix (Top 10 Classes)", use_container_width=True)

    # Class distribution
    cd_plot = plots_dir / "class_distribution.png"
    if cd_plot.exists():
        st.markdown('<div class="section-title">Class Distribution</div>', unsafe_allow_html=True)
        st.image(str(cd_plot), caption="Class Distribution (Sample Data)", use_container_width=True)

    # Error analysis
    err_dir = PROJECT_ROOT / "outputs" / "predictions" / "errors"
    err_images = list(err_dir.glob("*.png")) if err_dir.exists() else []
    if err_images:
        st.markdown('<div class="section-title">Error Analysis</div>', unsafe_allow_html=True)
        for ei in err_images:
            st.image(str(ei), caption=ei.stem.replace("_", " ").title(), use_container_width=True)

    # Per-language table
    st.markdown('<div class="section-title">Per-Language Evaluation</div>', unsafe_allow_html=True)
    lang_table = {
        "Language": languages,
        "Precision": ["N/A"] * 12,
        "Recall": ["N/A"] * 12,
        "F1": ["N/A"] * 12,
    }
    st.dataframe(lang_table, use_container_width=True, hide_index=True)
    st.caption("Per-language evaluation requires language metadata in the annotations. Results will be populated when evaluated on the full IndicDLP dataset with language labels.")


# ══════════════════════════════════════════════════════════════
#  TAB 5: ABOUT
# ══════════════════════════════════════════════════════════════
with tab_about:
    ab1, ab2 = st.columns([3, 2])

    with ab1:
        st.markdown("""
        <div class="about-card">
            <h4>About IndicDoc AI</h4>
            <p><b>IndicDoc AI</b> is a deep learning system for multilingual Indian document
            layout understanding. It detects and classifies structural elements &mdash;
            paragraphs, tables, headers, footers, figures, and more &mdash; from scanned
            document images across 12 Indian languages.</p>

            <h4>Problem Statement</h4>
            <p>Indian documents exhibit significant diversity in scripts (Devanagari, Tamil, Bengali, etc.),
            layouts (newspapers, textbooks, forms), and quality (scanned, photographed, degraded).
            Automated understanding requires robust deep learning models that can handle this diversity.</p>

            <h4>Methodology</h4>
            <ol>
                <li><b>Dataset:</b> IndicDLP from AIKosh &mdash; 119,806 images, 12 languages, 42 classes</li>
                <li><b>Baseline:</b> Faster R-CNN with ResNet-50 FPN backbone</li>
                <li><b>Proposed:</b> YOLOv8s fine-tuned for document layout detection</li>
                <li><b>Evaluation:</b> mAP, Precision, Recall, F1-Score, IoU</li>
            </ol>

            <h4>Pipeline</h4>
            <p><code>Document Image &rarr; Preprocessing &rarr; Deep Learning Model &rarr;
            NMS &rarr; Bounding Boxes + Classes + Confidence &rarr; Visualization</code></p>

            <h4>SDG Alignment</h4>
            <p><b>SDG 9: Industry, Innovation and Infrastructure</b> &mdash; Enables AI-based
            document digitization, supports multilingual information processing, and contributes
            to intelligent document infrastructure for India.</p>
        </div>
        """, unsafe_allow_html=True)

    with ab2:
        st.markdown("""
        <div class="about-card">
            <h4>Technology Stack</h4>
            <ul>
                <li>Python 3.13</li>
                <li>PyTorch + torchvision</li>
                <li>Ultralytics YOLOv8</li>
                <li>OpenCV</li>
                <li>Streamlit</li>
                <li>Albumentations</li>
                <li>scikit-learn</li>
                <li>matplotlib + seaborn</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card">
            <h4>Links</h4>
            <ul>
                <li><a href="https://aikosh.indiaai.gov.in/home" target="_blank">AIKosh Dataset Platform</a></li>
                <li><a href="https://github.com/Prashantj44/IndicDoc-AI" target="_blank">GitHub Repository</a></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card">
            <h4>Project Info</h4>
            <table style="width:100%; font-size:0.85rem;">
                <tr><td><b>Type</b></td><td>B.E. AI&ML Mini-Project</td></tr>
                <tr><td><b>Domain</b></td><td>Computer Vision / DL</td></tr>
                <tr><td><b>SDG</b></td><td>9 &mdash; Industry & Innovation</td></tr>
                <tr><td><b>Dataset</b></td><td>IndicDLP (AIKosh)</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-footer">
    <span class="gold">&loz;</span> &nbsp;
    <b>IndicDoc AI</b> &mdash; Deep Learning-Based Multilingual Indian Document Layout Understanding
    &nbsp; <span class="gold">&loz;</span><br>
    B.E. Artificial Intelligence & Machine Learning &bull; Deep Learning Mini-Project &bull; SDG 9
</div>
""", unsafe_allow_html=True)
