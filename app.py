"""OncoNeuro AI: brain MRI classifier with Grad-CAM. Learning project, not a medical tool.
Run with `streamlit run app.py`."""
import json
import os

import streamlit as st
import streamlit.components.v1 as components

import about_page
import analyze_page
import docs_pages
import journey_page
import site_config as cfg
import model
import ui

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
st.set_page_config(page_title="OncoNeuro AI | Brain MRI classifier", layout="wide",
                   page_icon=os.path.join(ASSETS, "mark.svg"), initial_sidebar_state="collapsed")
st.session_state.setdefault("dark", True)   # new visitors land in dark mode
ui.inject_css()


logo = "logo_dark.svg" if ui.is_dark() else "logo.svg"
st.logo(os.path.join(ASSETS, logo), icon_image=os.path.join(ASSETS, "mark.svg"), size="large")

PAGE_JS = r"""
(function () {
  if (window.__onconeuroBoot) return; window.__onconeuroBoot = true;
  if (!document.getElementById('onco-font')) {
    var f = document.createElement('link'); f.id = 'onco-font'; f.rel = 'stylesheet';
    f.href = 'https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;500;600&display=swap';
    document.head.appendChild(f);
  }

  // ---- welcome / concern message: pure browser code, so the button always works ----
  var showConcern = function () {
    if (document.getElementById('onco-concern')) return;
    var st = document.createElement('style');
    st.textContent = '#onco-concern{position:fixed;inset:0;z-index:2000000;background:rgba(0,0,0,.6);display:flex;align-items:center;justify-content:center;padding:12px;overflow-y:auto;animation:ocIn .35s ease}'
      + '#onco-concern .box{background:var(--card,#2a2927);color:var(--body,#d6d4cb);border:1px solid var(--line,#3d3c39);border-radius:20px;max-width:640px;width:100%;max-height:94vh;overflow-y:auto;padding:1.5rem 1.6rem;box-shadow:0 20px 60px rgba(0,0,0,.5);font-family:"Source Sans 4","Source Sans Pro",sans-serif;line-height:1.6;animation:ocUp .4s cubic-bezier(.22,.61,.36,1)}'
      + '#onco-concern h2{font-family:"Source Serif 4",Georgia,serif;font-weight:500;font-size:1.6rem;margin:0 0 .6rem;color:var(--ink,#f5f4ee)}'
      + '#onco-concern p{margin:.5rem 0;color:var(--body,#d6d4cb)} #onco-concern b{color:var(--ink,#f5f4ee)}'
      + '#onco-concern .wb{background:var(--soft,#2f2e2b);border:1px solid var(--line,#3d3c39);border-left:4px solid var(--accent,#d97757);border-radius:14px;padding:.8rem 1rem;margin:.8rem 0}'
      + '#onco-concern ul{margin:.4rem 0 0 1.1rem;padding:0} #onco-concern li{margin:.25rem 0}'
      + '#onco-concern button{width:100%;margin-top:.8rem;min-height:2.8rem;border:0;border-radius:999px;background:var(--accent,#d97757);color:#fff;font-size:1rem;font-weight:600;cursor:pointer}'
      + '#onco-concern button:hover{filter:brightness(1.08)} #onco-concern button:focus-visible{outline:3px solid var(--ink,#fff);outline-offset:2px}'
      + '@keyframes ocIn{from{opacity:0}to{opacity:1}} @keyframes ocUp{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:none}}';
    document.head.appendChild(st);
    var o = document.createElement('div'); o.id = 'onco-concern';
    o.setAttribute('role', 'dialog'); o.setAttribute('aria-modal', 'true'); o.setAttribute('aria-labelledby', 'onco-ct');
    o.innerHTML = '<div class="box"><h2 id="onco-ct">Before you begin</h2>'
      + '<p><b>This is a learning project, not a medical tool.</b> OncoNeuro AI cannot diagnose anyone. It is a student project that shows how a computer can be taught to sort brain MRI pictures into four groups.</p>'
      + '<div class="wb"><b>Why you must not rely on it</b><ul>'
      + '<li>It was trained on one public set of pictures, not on a real hospital\'s patients.</li>'
      + '<li>It is wrong about 1 in 8 scans it has never seen, and it can be wrong while sounding very sure.</li>'
      + '<li>It misses many gliomas: it finds only about 68 in 100 of them.</li>'
      + '<li>It looks at a single flat picture. A doctor looks at the whole scan, your history, your symptoms and other tests.</li>'
      + '<li>A picture unlike the ones it learned from can make it fail in strange ways.</li></ul></div>'
      + '<p><b>If you or someone you know has a health concern, please see a qualified doctor.</b> Do not use anything on this website to decide about a diagnosis, a treatment or whether to seek care. Uploaded pictures are processed in memory and are not saved.</p>'
      + '<button type="button" id="onco-ok">I understand, take me to the project</button></div>';
    var close = function () { o.style.transition = 'opacity .3s'; o.style.opacity = '0'; setTimeout(function () { o.remove(); st.remove(); }, 300); document.removeEventListener('keydown', esc); };
    var esc = function (e) { if (e.key === 'Escape') close(); };
    document.body.appendChild(o);
    document.getElementById('onco-ok').addEventListener('click', close);
    document.addEventListener('keydown', esc);
    setTimeout(function () { var b = document.getElementById('onco-ok'); if (b) b.focus(); }, 50);
  };
  if (document.body) showConcern(); else window.addEventListener('DOMContentLoaded', showConcern);
  document.addEventListener('click', function (e) {
    var t = e.target && e.target.closest && e.target.closest(
      'img[data-testid="stLogo"],[data-testid="stLogoLink"],[data-testid="stSidebarLogo"],[data-testid="stHeaderLogo"]');
    if (!t) return; e.preventDefault(); e.stopPropagation(); window.location.assign(window.location.origin + '/');
  }, true);
  // choosing a section in the menu closes the menu (it can be reopened from the header)
  var closeMenu = function () {
    var b = document.querySelector('[data-testid="stSidebarCollapseButton"] button') || document.querySelector('[data-testid="stSidebarCollapseButton"]');
    if (b && b.offsetParent !== null) b.click();
  };
  document.addEventListener('click', function (e) {
    var a = e.target && e.target.closest && e.target.closest('[data-testid="stSidebarNav"] a, [data-testid="stSidebarNavLink"]');
    if (a) setTimeout(closeMenu, 120);   // once, right after the click, so it glides shut with the page change
  }, true);
  var n = 0, iv = setInterval(function () {
    var b = document.querySelector('[data-testid="stSidebarCollapseButton"] button') || document.querySelector('[data-testid="stSidebarCollapseButton"]');
    if (b && b.offsetParent !== null) { b.click(); clearInterval(iv); }
    else if (document.querySelector('[data-testid="stExpandSidebarButton"]') || ++n > 40) clearInterval(iv);
  }, 150);
  var last = 0;
  var hide = function (v) {
    [document.querySelector('[data-testid="stHeader"]'), document.querySelector('.st-key-themesw')]
      .forEach(function (el) { if (el) el.style.transform = v ? 'translateY(-130%)' : ''; });
  };
  document.addEventListener('scroll', function (e) {
    var m = e.target; if (!m || !m.getAttribute || m.getAttribute('data-testid') !== 'stMain') return;
    var t = m.scrollTop, d = t - last;
    if (t > 60 && d > 4) hide(true); else if (d < -4 || t <= 10) hide(false);
    last = t;
  }, true);
})();
"""
components.html(
    "<script>var d=window.parent.document;if(!window.parent.__onconeuroBoot){var s=d.createElement('script');"
    "s.textContent=" + json.dumps(PAGE_JS) + ";d.head.appendChild(s);}</script>", height=0)

analyze = st.Page(analyze_page.run, title="Analyze a scan", url_path="analyze", default=True)
journey = st.Page(journey_page.run, title="Our journey", url_path="journey")
about = st.Page(about_page.run, title="About me", url_path="about")
chapters = docs_pages.build()
journey_page.BOOK = chapters
pg = st.navigation({"Try it": [analyze], "The story": [journey, about],
                    "Documentation": chapters}, expanded=True)

ui.hidden_theme_toggle()
ui.theme_switch()

with st.sidebar:
    st.caption("Learning project. Not a medical tool.")
    st.link_button("Project repository", cfg.PROJECT_REPO_URL, width="stretch")
    st.link_button("GitHub profile", cfg.GITHUB_URL, width="stretch")
    st.link_button("LinkedIn", cfg.LINKEDIN_URL, width="stretch")
    st.caption(f"Created by {cfg.AUTHOR}")

pg.run()

# only after the page has been drawn: load PyTorch + the model in the background while the visitor reads
model.warmup()
