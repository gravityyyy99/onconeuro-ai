"""Builds the documentation 'book': one Streamlit page per chapter (markdown in docs_content/)."""
import json
import os

import streamlit as st

import ui

HERE = os.path.dirname(os.path.abspath(__file__))
CONTENT = os.path.join(HERE, "docs_content")

CONTENTS = ("00_contents.md", "Contents", "contents")
CHAPTERS = [
    ("01_plain_words.md", "In plain words", "plain-words"),
    ("02_problem.md", "The problem", "the-problem"),
    ("03_data.md", "The data", "the-data"),
    ("04_model.md", "The model", "the-model"),
    ("05_gradcam.md", "Seeing inside: Grad-CAM", "grad-cam"),
    ("06_training.md", "Training and testing", "training-and-testing"),
    ("07_app.md", "The app", "the-app"),
    ("08_results.md", "Results and limits", "results"),
    ("09_safety.md", "Safety and honesty", "safety"),
    ("10_credits.md", "Credits", "credits"),
    ("11_plans.md", "Future plans", "future-plans"),
]
BOOK = [CONTENTS] + CHAPTERS
_pages = []


def label(i):
    return BOOK[i][1] if i == 0 else f"Chapter {i}: {BOOK[i][1]}"


def _values():
    ev = ui.evaluation()
    if not ev:
        return {}
    v = {"acc": f"{ev['accuracy'] + 1e-9:.1%}", "n_test": f"{ev['test_size']:,}",
         "n_train": f"{sum(ev['train_counts'].values()):,}",
         "conf_right": f"{ev['mean_conf_correct']:.0%}", "conf_wrong": f"{ev['mean_conf_wrong']:.0%}"}
    for c, d in ev["per_class"].items():
        v[f"rec_{c}"], v[f"prec_{c}"] = f"{d['recall']:.0%}", f"{d['precision']:.0%}"
    return v


@st.cache_data(show_spinner=False)
def _chapter(fname):
    with open(os.path.join(CONTENT, fname), encoding="utf-8") as f:
        return _fill(f.read())


def _fill(text):
    for k, val in _values().items():
        text = text.replace("{{" + k + "}}", val)
    return text


def _results_extra():
    for fname, cap in (("gradcam_right.jpg", "Correct answers and their heatmaps."),
                       ("gradcam_wrong.jpg", "Mistakes. The heat is often away from any obvious tumor.")):
        path = os.path.join(HERE, "assets", "examples", fname)
        if os.path.exists(path):
            st.image(path, caption=cap, width="stretch")



def _plans_visual():
    svg = (
        '<svg viewBox="0 0 700 200" role="img" aria-label="Roadmap from today to expert review" xmlns="http://www.w3.org/2000/svg">'
        '<path class="rbase" d="M40 150 C140 150 150 120 230 112 S350 100 400 80 S560 60 660 40"/>'
        '<path id="rp" class="rpath" d="M40 150 C140 150 150 120 230 112 S350 100 400 80 S560 60 660 40"/>'
        '<circle class="rnode n1" cx="40" cy="150" r="9"/><circle class="rnode n2" cx="190" cy="118" r="9"/>'
        '<circle class="rnode n3" cx="330" cy="98" r="9"/><circle class="rnode n4" cx="490" cy="68" r="9"/>'
        '<circle class="rnode n5" cx="660" cy="40" r="9"/>'
        '<text class="t1" x="50" y="180" text-anchor="middle">Today</text><text class="sub t1" x="50" y="194" text-anchor="middle">frozen ResNet-18</text>'
        '<text class="t2" x="190" y="96" text-anchor="middle">Fine-tune</text>'
        '<text class="t3" x="330" y="76" text-anchor="middle">Better data</text>'
        '<text class="t4" x="490" y="46" text-anchor="middle">Safer answers</text>'
        '<text class="t5" x="640" y="20" text-anchor="middle">Expert review</text>'
        '<circle class="rider" r="6"><animateMotion dur="5s" begin="3.6s" repeatCount="indefinite"><mpath href="#rp"/></animateMotion></circle>'
        '</svg>')
    st.markdown(f'<div class="road">{svg}</div>', unsafe_allow_html=True)
    st.caption("The path ahead. The line draws itself, and the dot keeps travelling because the work is not finished.")


def _plans_phases():
    phases = [
        ("PHASE 1", "Improve the model", "Fine-tune the whole network, compare other networks, and report how much glioma detection improves.", "NEXT"),
        ("PHASE 2", "Better data", "Use scans from several sources, and several slices per patient instead of a single picture.", "PLANNED"),
        ("PHASE 3", "Safer answers", "Add an 'I am not sure' answer, calibrate the confidence, and check fairness across groups.", "PLANNED"),
        ("PHASE 4", "Explain and verify", "Publish a gallery of the hardest mistakes, and talk to doctors before any step toward real use.", "LATER"),
    ]
    for col, (i, (k, t, dsc, chip)) in zip(st.columns(4), enumerate(phases)):
        cls = "chip next" if chip == "NEXT" else "chip"
        col.markdown(f'<div class="plan" style="animation-delay:{.3 + i * .25:.2f}s"><div class="k">{k}</div><div class="t">{t}</div>'
                     f'<div class="d">{dsc}</div><span class="{cls}">{chip}</span></div>', unsafe_allow_html=True)
    st.caption("Phase 1 is where I start. Hover a card to lift it.")
    import site_config as cfg
    st.link_button("Follow the project repository for updates", cfg.PROJECT_REPO_URL)


EXTRAS = {"08_results.md": _results_extra, "11_plans.md": _plans_phases}
PRE = {"11_plans.md": _plans_visual}


def _make(i):
    fname = BOOK[i][0]

    def page():
        if i > 0:
            st.markdown(f'<div class="chap">CHAPTER {i}</div>', unsafe_allow_html=True)
        text = _chapter(fname)
        if fname in PRE:   # motion first, then the words
            head, _, rest = text.partition("\n")
            st.markdown(head)
            PRE[fname]()
            st.markdown(rest)
        else:
            st.markdown(text)
        if i == 0:
            for j in range(1, len(BOOK)):
                st.page_link(_pages[j], label=label(j))
        if fname in EXTRAS:
            EXTRAS[fname]()
        st.divider()
        left, right = st.columns(2)
        if i > 0:
            left.page_link(_pages[i - 1], label=f"Previous: {label(i - 1)}")
        if i < len(BOOK) - 1:
            right.page_link(_pages[i + 1], label=f"Next: {label(i + 1)}")
        ui.footer()

    page.__name__ = f"chapter_{i}"
    return page


def build():
    _pages.clear()
    for i, (_, _, slug) in enumerate(BOOK):
        _pages.append(st.Page(_make(i), title=label(i), url_path=slug))
    return list(_pages)
