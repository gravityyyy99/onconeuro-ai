"""Analyze a scan: upload or pick a sample, see the model's answer and where it looked (Grad-CAM)."""
import io
import os

import streamlit as st
from PIL import Image, UnidentifiedImageError

import model as M
import ui

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, "assets", "samples")
MAX_PIXELS = 25_000_000   # refuse absurdly large pictures instead of running out of memory

PLAIN = {
    "glioma": "A tumor that grows from the brain's supporting (glial) cells.",
    "meningioma": "A tumor that grows from the membranes that wrap the brain.",
    "notumor": "The model sees no sign of the three tumor types it knows.",
    "pituitary": "A tumor in the small pituitary gland at the base of the brain.",
}


@st.cache_data(show_spinner=False, max_entries=32)
def _analyze(data: bytes):
    img = Image.open(io.BytesIO(data))
    if img.width * img.height > MAX_PIXELS:
        raise ValueError("too large")
    return M.analyze_bytes(data)   # runs in the background worker process (PyTorch + Grad-CAM)


def run():
    _body()
    ui.footer()


def _body():
    st.markdown('<div class="chap">TRY IT</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero"><h1>OncoNeuro AI</h1>'
        '<div class="tag">A brain MRI classifier that shows its work</div>'
        "<p>Give it one brain MRI picture. It names the closest of four groups (glioma, meningioma, pituitary "
        "tumor or no tumor) and paints a heatmap over the places that pushed it to that answer.</p></div>",
        unsafe_allow_html=True)
    st.markdown(
        '<div class="warnbox"><b>Learning project, not a medical tool.</b> This cannot diagnose anyone and it '
        "is often wrong. Please do not use it for any health decision. See a doctor instead.</div>",
        unsafe_allow_html=True)

    if not M.weights_available():
        st.error("The model file `onconeuro_resnet18.pt` is missing. Train it with `train_local.py` or the "
                 "Colab notebook, then place it next to `app.py`.")
        return

    st.markdown('<div class="section">1. Choose a scan</div>', unsafe_allow_html=True)
    st.markdown('<div class="cap">Upload your own picture (JPG or PNG), or try one of four sample scans '
                "from the public test set.</div>", unsafe_allow_html=True)
    up = st.file_uploader("Upload a brain MRI picture", type=["jpg", "jpeg", "png"],
                          label_visibility="collapsed")
    st.caption("Or try a sample scan:")
    cols = st.columns(4)
    for col, c in zip(cols, M.DEFAULT_CLASSES):
        if col.button(M.LABELS[c], key=f"s_{c}", width="stretch"):
            st.session_state["sample"] = c
            st.session_state["use_sample"] = True

    data, name = None, None
    if up is not None:
        data, name = up.getvalue(), up.name
        st.session_state["use_sample"] = False
    elif st.session_state.get("sample") and st.session_state.get("use_sample", True):
        c = st.session_state["sample"]
        with open(os.path.join(SAMPLES, f"{c}.jpg"), "rb") as f:
            data, name = f.read(), f"sample scan ({M.LABELS[c]})"

    if data is None:
        st.info("Pick a sample or upload a picture to see the result here.")
        return

    try:
        with st.spinner("Looking at the scan (the first one can take a few seconds while the model wakes up)..."):
            probs, overlay, resized = _analyze(data)
    except (UnidentifiedImageError, ValueError, OSError):
        st.error("That file could not be read as a normal picture, or it is too large. Try a JPG or PNG "
                 "under 25 megapixels.")
        return

    top = max(probs, key=probs.get)
    st.markdown('<div class="section">2. What the model saw</div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    a.image(resized, caption=f"Your scan, resized to 224 x 224 ({name})", width="stretch")
    b.image(overlay, caption="Grad-CAM heatmap: red and yellow pushed hardest toward the answer", width="stretch")
    st.markdown('<div class="legend2"><span>less</span><span class="grad"></span><span>more influence</span></div>',
                unsafe_allow_html=True)

    st.markdown('<div class="section">3. The answer, and how sure it is</div>', unsafe_allow_html=True)
    st.markdown(f"**Closest group: {M.LABELS.get(top, top)}** ({probs[top]:.1%})")
    st.caption(PLAIN.get(top, ""))
    if probs[top] < 0.7:
        st.warning("The model is not confident about this one. Treat the answer as a coin toss between the top groups.")
    else:
        st.markdown('<div class="warnbox">High confidence is not the same as correct. On scans it had never seen, '
                    "this model was wrong about 1 time in 8, and it was sometimes very sure when it was wrong.</div>",
                    unsafe_allow_html=True)
    rows = "".join(
        f'<div class="prob"><div style="display:flex;justify-content:space-between"><b>{M.LABELS.get(c, c)}</b>'
        f'<span>{p:.1%}</span></div><div class="bar {"" if c == top else "g"}"><i style="--w:{p * 100:.1f}%"></i></div></div>'
        for c, p in sorted(probs.items(), key=lambda kv: -kv[1]))
    st.markdown(rows, unsafe_allow_html=True)

    with st.expander("How do I read the heatmap?"):
        st.markdown(
            "- The colors show where the model's attention was strongest. They are **not** an outline of a tumor.\n"
            "- If the red sits on a bright, odd-looking patch inside the brain, the model may be reacting to something real.\n"
            "- If the red sits on the skull edge, the black background or a corner, the model is probably "
            "reacting to something unrelated. Do not trust that answer.\n"
            "- Even a sensible-looking heatmap proves nothing about a diagnosis.")
    st.caption("The picture is processed in memory to make this result and is not saved by this app.")
