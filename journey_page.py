"""Our journey: how OncoNeuro AI was made, step by step, with graphics and results."""
import json
import os

import streamlit as st

import labels as M
import ui

HERE = os.path.dirname(os.path.abspath(__file__))
BOOK = []   # documentation pages, set by app.py


def _bar(label, text, width, cls=""):
    return (f'<div style="display:flex;justify-content:space-between"><b>{label}</b><span>{text}</span></div>'
            f'<div class="bar {cls}"><i style="--w:{width:.1f}%"></i></div>')


def run():
    ev = ui.evaluation()

    st.markdown('<div class="chap">THE WHOLE STORY</div>', unsafe_allow_html=True)
    st.title("Our journey")
    st.markdown("How OncoNeuro AI was made, from the first question to what you can use now. This page is the "
                "short, visual version. The full reasoning is in the Documentation chapters in the menu.")

    # 1. path
    st.markdown('<div class="section">1. The path of a scan</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="journey">'
        '<div class="jn"><b>STEP 1</b><span>Scan</span><small>a brain MRI picture</small></div><div class="jl"></div>'
        '<div class="jn"><b>STEP 2</b><span>Prepare</span><small>gray, 224 x 224, scaled</small></div><div class="jl"></div>'
        '<div class="jn"><b>STEP 3</b><span>ResNet-18</span><small>finds edges and shapes</small></div><div class="jl"></div>'
        '<div class="jn"><b>STEP 4</b><span>Answer + heatmap</span><small>one of 4 groups</small></div>'
        '</div>', unsafe_allow_html=True)
    st.caption("Every picture follows this path. The heatmap is made from the same pass, so it shows what "
               "really drove the answer.")

    # 2. numbers
    st.markdown('<div class="section">2. The project in numbers</div>', unsafe_allow_html=True)
    tr = sum(ev["train_counts"].values()) if ev else 5600
    te = ev["test_size"] if ev else 1600
    nums = [(str(tr), "scans used to teach the model"), (str(te), "scans kept hidden for the final test"),
            ("4", "groups to choose from"), ("2052", "numbers the model actually learned")]
    for col, (n, label) in zip(st.columns(4), nums):
        col.markdown(f'<div class="stat"><div class="n cnt" style="--to:{n}"></div><div class="l">{label}</div></div>',
                     unsafe_allow_html=True)
    st.caption("ResNet-18 has about 11 million numbers inside it. We froze nearly all of them and trained only "
               "the last 2,052, which is what makes this a small, fast, beginner-friendly project.")

    # 3. steps
    st.markdown('<div class="section">3. How we built it, step by step</div>', unsafe_allow_html=True)
    steps = [
        ("Pick a question we can answer", "Can a computer sort brain MRI pictures into glioma, meningioma, "
         "pituitary tumor and no tumor, and can it show us why it chose?"),
        ("Find honest data", "The public Brain Tumor MRI Dataset on Kaggle: 7,200 scans already split into a "
         "training folder and a testing folder the model never sees while learning."),
        ("Borrow a pretrained eye", "ResNet-18 already learned to see edges, textures and shapes from over a "
         "million everyday photos. We reuse that skill instead of starting from zero (transfer learning)."),
        ("Freeze it and train only the last step", "Everything it knew stays fixed. A new final layer learns to "
         "map those features to our four groups."),
        ("Add the heatmap", "Grad-CAM paints the parts of the scan that pushed the model toward its answer, so "
         "we can check it is looking at the brain and not at an accident of the picture."),
        ("Test on scans it has never seen", "The score on the hidden 1,600 scans is the real score: "
         f"{ev['accuracy'] + 1e-9:.1%}." if ev else "The score on the hidden scans is the real score."),
        ("Be honest about the limits", "We list what it gets wrong, which groups it confuses, and why it must "
         "never be used for real decisions."),
    ]
    html = '<div class="tl">' + "".join(
        f'<div class="tli" style="animation-delay:{i * .12:.2f}s"><div class="dot">{i + 1}</div>'
        f'<div class="tlb"><b>{t}</b><p>{d}</p></div></div>' for i, (t, d) in enumerate(steps)) + "</div>"
    st.markdown(html, unsafe_allow_html=True)

    # 4. results
    st.markdown('<div class="section">4. How well did it do?</div>', unsafe_allow_html=True)
    if not ev:
        st.info("Run `evaluate.py` to create the results.")
    else:
        st.markdown(f'<div class="cap">Overall accuracy on the {ev["test_size"]} hidden scans: '
                    f'<b>{ev["accuracy"] + 1e-9:.1%}</b>. A random guess among 4 groups would score 25%.</div>',
                    unsafe_allow_html=True)
        st.markdown(_bar("Overall accuracy", f"{ev['accuracy'] + 1e-9:.1%}", ev["accuracy"] * 100), unsafe_allow_html=True)
        st.markdown("**Of the real cases in each group, how many did it find?** (recall)")
        for c in ev["classes"]:
            r = ev["per_class"][c]["recall"]
            st.markdown(_bar(M.LABELS[c], f"{r:.0%}", r * 100, "g" if r < .8 else ""), unsafe_allow_html=True)
        st.caption("Glioma is the weak spot: about 1 in 3 real gliomas were missed. For a medical question that "
                   "is exactly the number that matters most, and it is one more reason this is not a medical tool.")

        # plain HTML table: no charting library to load, so this page opens instantly
        names = [M.LABELS[c] for c in ev["classes"]]
        head = "".join(f"<th>{n}</th>" for n in names)
        body = ""
        for i, row in enumerate(ev["confusion"]):
            cells = ""
            for j, v in enumerate(row):
                pct = min(100, round(v / max(sum(row), 1) * 100))
                cls = "ok" if i == j else ""
                cells += (f'<td class="{cls}" style="background:color-mix(in srgb, var(--accent) {pct}%, var(--soft))">'
                          f'{v}</td>')
            body += f"<tr><th>{names[i]}</th>{cells}</tr>"
        st.markdown(
            '<div class="cmwrap"><div class="cmtitle">Confusion matrix: where it was right and wrong</div>'
            '<div class="cmhint">Rows are what the scan really was. Columns are what the model said.</div>'
            f'<table class="cm"><tr><th></th>{head}</tr>{body}</table></div>', unsafe_allow_html=True)
        st.caption("The diagonal is correct answers. Anything off the diagonal is a mistake. Look at the glioma "
                   "row: many gliomas were called meningioma.")

    # 5. heatmaps
    st.markdown('<div class="section">5. Looking where it looked</div>', unsafe_allow_html=True)
    for fname, cap in (("gradcam_right.jpg", "Four correct answers and their heatmaps."),
                       ("gradcam_wrong.jpg", "Four mistakes. Notice the heat is often away from any obvious tumor.")):
        path = os.path.join(HERE, "assets", "examples", fname)
        if os.path.exists(path):
            st.image(path, caption=cap, width="stretch")

    st.markdown('<div class="section">6. Keep reading</div>', unsafe_allow_html=True)
    st.markdown("The Documentation chapters explain each part in plain words, including the thought process "
                "behind every choice.")
    if BOOK:
        st.page_link(BOOK[0], label="Open the documentation")
    ui.footer()
