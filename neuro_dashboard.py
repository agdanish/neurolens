import streamlit as st
import tensorflow as tf
import tf_keras as legacy_keras
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import os
import time
import streamlit.components.v1 as components

# --- 1. DIRECT IMPORT BYPASS ---
try:
    from transformers.models.vit.modeling_tf_vit import TFViTModel
    from transformers.models.vit.configuration_vit import ViTConfig
except ImportError as e:
    st.error(f"Critical Library Error: {e}")
    st.stop()

# --- 2. Custom Layers ---
class TensorTranspose(legacy_keras.layers.Layer):
    def call(self, x): return tf.transpose(x, perm=[0, 3, 1, 2])
    def get_config(self): return super().get_config()

class VisionTransformerFeatureExtractor(legacy_keras.layers.Layer):
    def __init__(self, trainable=True, **kwargs):
        super().__init__(**kwargs)
        self.trainable_arg = trainable
        try:
            config = ViTConfig.from_pretrained('google/vit-base-patch16-224', image_size=224, num_channels=3)
            self.vit = TFViTModel.from_pretrained('google/vit-base-patch16-224', config=config)
        except Exception: self.vit = None
        if self.vit: self.vit.trainable = trainable
        self.transpose = TensorTranspose()

    def call(self, inputs, training=False):
        x = tf.cast(inputs, tf.float32)
        x = (x * 2.0) - 1.0
        x = self.transpose(x)
        if self.vit: return self.vit(x, training=training).last_hidden_state[:, 0]
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"trainable": self.trainable_arg})
        return config

# --- 3. ARCHITECTURE BLUEPRINTS ---
def build_resnet50():
    base = legacy_keras.applications.ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
    base.trainable = False
    return legacy_keras.Sequential([base, legacy_keras.layers.GlobalAveragePooling2D(), legacy_keras.layers.Dense(256, activation='relu'), legacy_keras.layers.Dropout(0.5), legacy_keras.layers.Dense(1, activation='sigmoid')])

def build_densenet121():
    inp = legacy_keras.Input(shape=(224, 224, 3))
    base = legacy_keras.applications.DenseNet121(weights=None, include_top=False, input_tensor=inp)
    x = legacy_keras.layers.Dropout(0.3)(legacy_keras.layers.Dense(256, activation='relu')(legacy_keras.layers.Dropout(0.5)(legacy_keras.layers.GlobalAveragePooling2D()(base.output))))
    return legacy_keras.Model(inputs=inp, outputs=legacy_keras.layers.Dense(1, activation='sigmoid')(x))

def build_inceptionv3():
    inp = legacy_keras.Input(shape=(224, 224, 3))
    base = legacy_keras.applications.InceptionV3(weights=None, include_top=False, input_tensor=inp)
    x = legacy_keras.layers.Dropout(0.3)(legacy_keras.layers.Dense(256, activation="relu")(legacy_keras.layers.Dropout(0.4)(legacy_keras.layers.concatenate([legacy_keras.layers.GlobalAveragePooling2D()(base.get_layer("mixed7").output), legacy_keras.layers.GlobalAveragePooling2D()(base.output)]))))
    return legacy_keras.Model(inputs=inp, outputs=legacy_keras.layers.Dense(1, activation="sigmoid")(x))

def build_vit():
    inp = legacy_keras.Input(shape=(224, 224, 3))
    x = VisionTransformerFeatureExtractor(trainable=False)(inp)
    x = legacy_keras.layers.Dropout(0.2)(legacy_keras.layers.Dense(512, activation="gelu", name="dense_1")(legacy_keras.layers.Dropout(0.2)(legacy_keras.layers.Dense(1024, activation="gelu", name="dense")(legacy_keras.layers.LayerNormalization(name="layer_normalization")(x)))))
    return legacy_keras.Model(inputs=inp, outputs=legacy_keras.layers.Dense(2, activation="softmax", name="dense_2")(x))

# --- 4. Smart Loader ---
def load_model_weights_safely(model, path, name):
    try: model.build((None, 224, 224, 3))
    except: pass
    try: model.load_weights(path); return True, "Strict"
    except:
        try: model.load_weights(path, by_name=True, skip_mismatch=True); return True, "Flexible"
        except Exception as e: return False, str(e)

# --- 5. UI CONFIGURATION ---
st.set_page_config(page_title="NeuroFundus Command Center", page_icon="👁️", layout="wide")

# --- 6. THE PARTICLE ENGINE (THE FIX) ---
# We inject JavaScript to force a canvas onto the parent window, bypassing Streamlit's iframe sandbox.
particle_js = """
<script>
    // 1. Clean up any existing canvas to prevent duplicates on rerun
    const existingCanvas = window.parent.document.getElementById('neural-canvas');
    if (existingCanvas) { existingCanvas.remove(); }

    // 2. Create the Canvas
    const canvas = window.parent.document.createElement('canvas');
    canvas.id = 'neural-canvas';
    canvas.style.position = 'fixed';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.zIndex = '-1'; // Behind everything
    canvas.style.background = 'radial-gradient(circle at center, #020617 0%, #000000 100%)';
    window.parent.document.body.appendChild(canvas);

    // 3. Initialize Context
    const ctx = canvas.getContext('2d');
    let width, height;

    function resize() {
        width = canvas.width = window.parent.innerWidth;
        height = canvas.height = window.parent.innerHeight;
    }
    window.parent.addEventListener('resize', resize);
    resize();

    // 4. Particle System
    const particles = [];
    const particleCount = 100; // Density
    const connectionDistance = 150;

    class Particle {
        constructor() {
            this.x = Math.random() * width;
            this.y = Math.random() * height;
            this.vx = (Math.random() - 0.5) * 1.5;
            this.vy = (Math.random() - 0.5) * 1.5;
            this.size = Math.random() * 2 + 1;
        }
        update() {
            this.x += this.vx;
            this.y += this.vy;
            if (this.x < 0 || this.x > width) this.vx *= -1;
            if (this.y < 0 || this.y > height) this.vy *= -1;
        }
        draw() {
            ctx.fillStyle = '#00f2ff'; // Cyan Dots
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    for (let i = 0; i < particleCount; i++) particles.push(new Particle());

    // 5. Animation Loop
    function animate() {
        ctx.clearRect(0, 0, width, height);

        // Draw Connections
        ctx.strokeStyle = 'rgba(0, 242, 255, 0.15)'; // Faint Cyan Lines
        ctx.lineWidth = 0.5;

        for (let i = 0; i < particles.length; i++) {
            particles[i].update();
            particles[i].draw();

            for (let j = i; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx*dx + dy*dy);

                if (dist < connectionDistance) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.stroke();
                }
            }
        }
        requestAnimationFrame(animate);
    }
    animate();
</script>
"""
components.html(particle_js, height=0, width=0)

# --- 7. CSS STYLING ---
st.markdown("""
<style>
    /* Make Streamlit Transparent so Canvas shows through */
    .stApp { background: transparent !important; }
    [data-testid="stAppViewContainer"] { background: transparent !important; }
    [data-testid="stHeader"] { background: rgba(0,0,0,0) !important; }
    [data-testid="stToolbar"] { visibility: hidden; } /* Hide the hamburger menu */

    /* Typography */
    * { font-family: 'Times New Roman', Times, serif !important; color: #e0f2fe; }

    /* Glass Cards */
    .glass-metric {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(56, 189, 248, 0.2);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: rgba(2, 6, 23, 0.9);
        border-right: 1px solid rgba(56, 189, 248, 0.1);
    }

    h1, h2, h3 { text-shadow: 0 0 15px rgba(56, 189, 248, 0.6); color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# Helper for Charts
def style_chart(fig, title):
    fig.update_layout(
        title={'text': title, 'font': {'family': 'Times New Roman', 'size': 18, 'color': 'white'}},
        paper_bgcolor='rgba(15, 23, 42, 0.6)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#e0f2fe', 'family': 'Times New Roman'},
        margin=dict(l=20, r=20, t=50, b=20),
        height=300
    )
    return fig

# --- 8. LOADING ENGINE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_CONFIG = {
    "ResNet50": {"path": os.path.join(BASE_DIR, "1-FundusR50-ResNet50", "FundusR50.h5"), "builder": build_resnet50},
    "InceptionV3": {"path": os.path.join(BASE_DIR, "2-FundusI3-InceptionV3", "FundusI3.h5"), "builder": build_inceptionv3},
    "DenseNet121": {"path": os.path.join(BASE_DIR, "3-FundusD121-DenseNet121", "FundusD121.h5"), "builder": build_densenet121},
    "ViT": {"path": os.path.join(BASE_DIR, "4-FundusViT-B16-VisionTransformer", "FundusViT-B16.weights.h5"), "builder": build_vit}
}

@st.cache_resource
def init_inference_engine():
    loaded, logs = {}, []
    for name, config in MODELS_CONFIG.items():
        if os.path.exists(config["path"]):
            try:
                model = config["builder"]()
                s, m = load_model_weights_safely(model, config["path"], name)
                if s: loaded[name] = model; logs.append(f"🟢 {name}: Online")
                else: logs.append(f"🔴 {name}: Failed")
            except: logs.append(f"🔴 {name}: Error")
        else: logs.append(f"⚪ {name}: Missing")
    return loaded, logs

def process_img(img_file):
    img = Image.open(img_file).convert("RGB")
    return img, np.expand_dims(np.array(img.resize((224, 224))) / 255.0, axis=0)

# --- 9. MAIN LOGIC ---
with st.sidebar:
    st.markdown("## ⚙️ SYSTEM PARAMETERS")
    models, logs = init_inference_engine()
    for log in logs: st.markdown(f"<small>{log}</small>", unsafe_allow_html=True)
    st.divider()
    w_vit = st.slider("ViT Weight Factor", 0.0, 3.0, 1.2)
    w_cnn = st.slider("CNN Weight Factor", 0.0, 3.0, 0.8)
    thresh = st.slider("Clinical Threshold", 0.0, 1.0, 0.5)

c1, c2 = st.columns([3, 1])
with c1: st.title("NeuroFundus Command Center")
with c2: st.markdown("<div class='glass-metric'><h5>SYSTEM ACTIVE</h5>v12.5 Particle Core</div>", unsafe_allow_html=True)

upload = st.file_uploader("Upload DICOM/Fundus Scan", type=['png', 'jpg', 'jpeg'])

if upload:
    img, data = process_img(upload)

    preds, speed = {}, {}
    with st.spinner("Analyzing Retinal Biomarkers..."):
        for name, model in models.items():
            t0 = time.time()
            try:
                raw = model.predict(data, verbose=0)
                preds[name] = float(raw[0][1] if raw.shape[-1] > 1 else raw[0][0])
            except: preds[name] = 0.0
            speed[name] = round((time.time()-t0)*1000, 1)

    vit_p = preds.get("ViT", 0)
    cnn_p = np.mean([v for k,v in preds.items() if k != "ViT"]) if len(preds)>1 else 0
    final_score = ((vit_p * w_vit) + (cnn_p * w_cnn)) / (w_vit + w_cnn + 1e-9)
    final_score = np.clip(final_score, 0, 1)

    status = "ALZHEIMER'S RISK DETECTED" if final_score > thresh else "LOW RISK / NORMAL"
    color = "#ff4444" if final_score > thresh else "#34d399"

    # ==================== VISUALIZATION DASHBOARD ====================

    col_main_1, col_main_2 = st.columns([1, 2])
    with col_main_1:
        st.image(img, caption="Processed Input Scan", use_column_width=True)
        st.markdown(f"<div class='glass-metric'><h2 style='color:{color}'>{status}</h2>{final_score:.2%} Probability</div>", unsafe_allow_html=True)

    with col_main_2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=final_score*100,
            title={'text': "Alzheimer's Probability Score"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color}, 'bgcolor': "rgba(0,0,0,0)"}
        ))
        st.plotly_chart(style_chart(fig, "Consensus Risk Meter"), use_column_width=True)

    st.divider()
    st.subheader("🧠 Deep Diagnostic Analytics")

    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        fig = go.Figure(go.Scatterpolar(r=[preds.get(k,0) for k in preds] + [list(preds.values())[0]], theta=list(preds.keys()) + [list(preds.keys())[0]], fill='toself', line_color='#34d399'))
        st.plotly_chart(style_chart(fig, "Architecture Consensus"), use_column_width=True)
    with r1c2:
        fig = go.Figure(data=[go.Pie(labels=["ViT (Global)", "CNN (Local)"], values=[(vit_p * w_vit), (cnn_p * w_cnn)], hole=.6)])
        st.plotly_chart(style_chart(fig, "Decision Factors"), use_column_width=True)
    with r1c3:
        df_conf = pd.DataFrame({"Model": list(preds.keys()), "Confidence": list(preds.values())})
        fig = px.bar(df_conf, x="Confidence", y="Model", orientation='h', color="Confidence", range_x=[0,1], color_continuous_scale="Bluered")
        st.plotly_chart(style_chart(fig, "Confidence Levels"), use_column_width=True)

    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        fig = px.scatter(x=list(speed.values()), y=list(preds.values()), size=[30]*4, color=list(preds.keys()), labels={'x':'ms', 'y':'Conf'})
        st.plotly_chart(style_chart(fig, "Speed vs Accuracy"), use_column_width=True)
    with r2c2:
        devs = [p - np.mean(list(preds.values())) for p in preds.values()]
        fig = go.Figure(go.Bar(x=list(preds.keys()), y=devs, marker_color=['#ff4444' if d>0 else '#34d399' for d in devs]))
        st.plotly_chart(style_chart(fig, "Divergence from Mean"), use_column_width=True)
    with r2c3:
        fig = go.Figure(data=go.Heatmap(z=[list(preds.values())], x=list(preds.keys()), y=['Risk'], colorscale='Viridis'))
        st.plotly_chart(style_chart(fig, "Risk Heatmap"), use_column_width=True)

    r3c1, r3c2, r3c3 = st.columns(3)
    with r3c1:
        img_arr = np.array(img)
        fig = go.Figure()
        for i, c in enumerate(['Red', 'Green', 'Blue']):
            h, b = np.histogram(img_arr[:,:,i], bins=64, range=(0, 256))
            fig.add_trace(go.Scatter(x=b[:-1], y=h, name=c, line=dict(color=c.lower())))
        st.plotly_chart(style_chart(fig, "RGB Spectrum"), use_column_width=True)
    with r3c2:
        fig = go.Figure(data=[go.Scatter3d(x=list(speed.values()), y=list(preds.values()), z=[1, 2, 3, 4], mode='markers', marker=dict(size=10, color=list(preds.values()), colorscale='Viridis'))])
        st.plotly_chart(style_chart(fig, "3D Diagnostic Manifold"), use_column_width=True)
    with r3c3:
        x_d = np.linspace(0, 1, 100); y_d = np.exp(-((x_d - final_score)**2) / 0.05)
        fig = go.Figure(go.Scatter(x=x_d, y=y_d, fill='tozeroy', line_color=color))
        st.plotly_chart(style_chart(fig, "Certainty Curve"), use_column_width=True)

else:
    st.markdown("<div class='glass-metric' style='margin-top:50px'><h2>Waiting for Retinal Scan...</h2></div>", unsafe_allow_html=True)
