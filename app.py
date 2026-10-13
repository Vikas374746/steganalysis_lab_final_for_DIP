"""
app.py  -  Streamlit GUI for the Image Steganalysis Lab.
Run with:  streamlit run app.py
"""
import hashlib
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import stego_core as sc

st.set_page_config(page_title="DIP-Guided Adaptive LSB Steganography", page_icon="🖼️", layout="wide")

PROPOSED = "1. Proposed Method: Baseline vs Proposed"
OPS = [PROPOSED, "2. Embed Text", "3. Embed Image", "4. Detect (Analyze Image)", "5. Extract Hidden Information"]
BASELINE_LABEL = "Baseline: LSB Replacement (sequential)"
PROPOSED_LABEL = "Proposed: DIP-Guided Adaptive LSB"
TECH_UI = {PROPOSED_LABEL: "dip", BASELINE_LABEL: "seq"}
IMG_TYPES = ["png", "bmp", "tif", "tiff", "jpg", "jpeg"]


@st.cache_data(show_spinner="Running steganalysis...")
def cached_analysis(img):
    return sc.analyze(img)


# ============================================================================ sidebar
with st.sidebar:
    st.title("🖼️ DIP-Guided Stego Lab")
    st.caption("Plain LSB steganography vs a DIP-guided adaptive LSB method")
    op = st.radio("Operation", OPS)
    password = st.text_input("Password (optional, text payloads)", type="password")
    st.caption("New password-protected text payloads use AES-GCM; encryption does not make steganography undetectable.")
    tech_label = st.selectbox("Embedding method (Embed pages)", list(TECH_UI))
    tech = TECH_UI[tech_label]
    st.caption("Used when **embedding**. Extraction recognises both methods automatically.")
    use_demo = st.checkbox("Use a synthetic demo cover when no cover is uploaded")
    st.caption("Use lossless images (PNG/BMP). JPEG destroys LSB data.")


# ============================================================================ helpers
def pick_image(label, key, allow_demo=False, force_gray=False):
    """File uploader returning (uint8 array, file-meta dict). An uploaded file always wins
    over the demo cover."""
    f = st.file_uploader(label, type=IMG_TYPES, key=key)
    if f is None:
        if allow_demo and use_demo:
            img = sc.synthetic_cover(384, 1)
            st.caption("Using the synthetic demo cover. Upload a file to use your own image instead.")
            return (sc.to_gray(img) if force_gray else img), None
        return None, None
    raw = f.getvalue()
    try:
        meta = sc.describe_file(raw)
    except Exception as e:
        st.error(f"Could not read this file as an image. Reason: {type(e).__name__}: {e}")
        st.caption("Typical causes: the file is truncated/corrupted (re-save or re-download it), "
                   "or it is not really a PNG/BMP. Open it in an image viewer to check.")
        return None, None
    meta["name"] = f.name
    for w in meta["warnings"]:
        st.warning(w)
    return sc.load_image(io.BytesIO(raw), force_gray), meta


def show_quality(cover, stego):
    q, diff = sc.quality(cover, stego)
    changed = int(np.count_nonzero(diff))
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MSE", f"{q['MSE']:.6f}")
    c2.metric("PSNR", "∞ dB" if np.isinf(q["PSNR (dB)"]) else f"{q['PSNR (dB)']:.2f} dB")
    c3.metric("SSIM", f"{q['SSIM']:.6f}")
    c4.metric("Samples changed", f"{changed:,}")
    c5.metric("Changed", f"{100 * changed / max(1, diff.size):.3f}%")
    a, b, c = st.columns(3)
    a.image(cover, caption="Cover")
    b.image(stego, caption="Stego")
    c.image(np.clip(diff * 255, 0, 255).astype(np.uint8), caption="Changed samples (amplified; white = changed)")


def _embed_signature(kind, cover, tech, message, password, secret_bytes, secret_gray):
    digest = hashlib.sha256()
    digest.update(bytes([kind]))
    digest.update(tech.encode("utf-8"))
    digest.update(cover.tobytes())
    if kind == sc.KIND_TEXT:
        digest.update(message.encode("utf-8"))
        digest.update(password.encode("utf-8"))
    else:
        digest.update(secret_bytes)
        digest.update(bytes([secret_gray]))
    return digest.digest()


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


def lsb_plane(img):
    g = sc.to_gray(img)
    return (g & 1) * 255


# ============================================================================ analysis view
def render_analysis(img, cover=None):
    m = cached_analysis(img)
    level, head, reasons = sc.verdict(m)
    {"HIGH": st.error, "MEDIUM": st.warning, "LOW": st.success}[level](f"**{head}**  (suspicion: {level})")
    for r in reasons:
        st.write("• " + r)
    st.caption("A LOW result is not proof of a clean image: short payloads, LSB matching (±1) and "
               "edge-adaptive embedding can evade these statistical tests. Try extraction as well.")

    c = st.columns(4)
    c[0].metric("Chi-square p (first window)", f"{m['p_first']:.4f}")
    c[1].metric("RS rate estimate", "n/a" if m["rs_rate"] is None else f"{m['rs_rate']:.3f}")
    c[2].metric("LSB entropy", f"{m['lsb_entropy']:.5f}")
    c[3].metric("LSB transition rate", f"{m['lsb_transition']:.4f}")

    tabs = st.tabs(["Bit planes", "Histogram & chi-square", "Spatial (Laplacian/Sobel)"])

    with tabs[0]:
        g = sc.to_gray(img)
        fig, ax = plt.subplots(2, 4, figsize=(12, 6))
        for b in range(8):
            a = ax[b // 4, b % 4]
            a.imshow((g >> b) & 1, cmap="gray")
            a.set_title(f"Bit plane {b + 1}" + (" (LSB)" if b == 0 else ""))
            a.axis("off")
        plt.tight_layout()
        show_fig(fig)
        if cover is not None and cover.shape == img.shape:
            x, y = st.columns(2)
            x.image(lsb_plane(cover), caption="Cover LSB plane")
            y.image(lsb_plane(img), caption="Suspect LSB plane")
        else:
            st.caption("Sequentially embedded data shows up as a noisy band at the top of the LSB plane.")

    with tabs[1]:
        flat = img.flatten()
        h = np.bincount(flat, minlength=256)
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].bar(range(256), h, width=1.0)
        ax[0].set_title("Histogram")
        ax[1].bar(range(96, 128), h[96:128], width=0.9,
                  color=["tab:blue" if i % 2 == 0 else "tab:orange" for i in range(96, 128)])
        ax[1].set_title("Zoom: even (blue) vs odd (orange) values")
        ax[2].plot(m["lengths"] / flat.size * 100, m["ps"], "o-")
        ax[2].axhline(0.5, ls="--", c="r")
        ax[2].set_xlabel("% of pixel stream analysed")
        ax[2].set_ylabel("chi-square p-value")
        ax[2].set_title("Progressive chi-square")
        plt.tight_layout()
        show_fig(fig)
        x = st.columns(4)
        x[0].metric("Chi-square statistic", f"{m['chi_stat']:.1f}")
        x[1].metric("Degrees of freedom", m["dof"])
        x[2].metric("p-value (whole image)", f"{m['p_all']:.4f}")
        x[3].metric("Rough payload (chi-square)", f"{m['chi_est_bytes']} B")
        st.info("**How to read it:** LSB replacement equalises the counts of each value pair (2k, 2k+1). "
                "A **high** p-value (close to 1) therefore means the pairs are suspiciously equal → "
                "embedding. A **low** p-value means natural, unequal pairs → no evidence. "
                "The progressive curve stays high while it is inside the embedded region and "
                "drops where the message ends.")

    with tabs[2]:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].imshow(np.abs(m["_lap"]), cmap="gray", vmax=np.percentile(np.abs(m["_lap"]), 99))
        ax[0].set_title("|Laplacian| high-frequency residual")
        ax[1].imshow(m["_sobel"], cmap="gray")
        ax[1].set_title("Sobel gradient magnitude")
        for a in ax:
            a.axis("off")
        plt.tight_layout()
        show_fig(fig)
        x = st.columns(2)
        x[0].metric("Laplacian HF energy", f"{m['hf_spatial']:.3f}")
        x[1].metric("Edge pixels", f"{m['edge_pct']:.2f} %")

    return m, level


# ============================================================================ extraction view
def render_payload(kind, data, key):
    if kind == sc.KIND_TEXT:
        st.text_area("Extracted message", data.decode("utf-8", errors="replace"), height=120, key=f"t_{key}")
        st.download_button("⬇️ Save as .txt", data, "extracted_message.txt", key=f"dt_{key}")
    else:
        try:
            st.image(Image.open(io.BytesIO(data)), caption="Extracted secret image")
        except Exception:
            st.warning("Payload marked as image could not be decoded.")
        st.download_button("⬇️ Save extracted image", data, "extracted_image.png", "image/png", key=f"di_{key}")


def render_diagnostics(img, meta, expanded):
    with st.expander("🔎 Image diagnostics: why might extraction fail?", expanded=expanded):
        if meta:
            st.table(pd.DataFrame([("File", meta.get("name", "")), ("Format / mode", f"{meta['format']} / {meta['mode']}"),
                                   ("Size", f"{meta['width']}×{meta['height']}"),
                                   ("File size", f"{meta['file_kb']:.1f} KB"),
                                   ("Distinct colours", f"{meta['colours']:,}")],
                                  columns=["Property", "Value"]).set_index("Property"))
        st.write("**Low bit planes** (noise-like ≈ 50 % / 50 % can carry data, structured planes cannot):")
        st.dataframe(pd.DataFrame(sc.plane_profile(img)), hide_index=True)
        st.write("**What each extractor read where it expects the header** "
                 "(a valid stego image from this app starts with `53 54 47 32` = \"STG2\"):")
        st.dataframe(pd.DataFrame(sc.probe_headers(img, password)), hide_index=True)



# ============================================================================ difference analysis
def render_cover_diff(cover, stego):
    """Cover vs stego: what exactly did embedding change?"""
    xor = np.bitwise_xor(cover, stego)
    changed = int(np.count_nonzero(xor))
    total = int(xor.size)
    q, diff = sc.quality(cover, stego)

    c = st.columns(5)
    c[0].metric("Samples changed", f"{changed:,}")
    c[1].metric("% of samples", f"{changed / total * 100:.3f} %")
    c[2].metric("MSE", f"{q['MSE']:.6f}")
    c[3].metric("PSNR", "inf dB" if np.isinf(q["PSNR (dB)"]) else f"{q['PSNR (dB)']:.2f} dB")
    c[4].metric("SSIM", f"{q['SSIM']:.6f}")
    if changed == 0:
        st.info("Cover and suspect image are identical: nothing was changed.")
        return
    st.caption(f"Largest per-sample change: {int(diff.max())} grey level(s). "
               "A maximum of 1 means only the LSB was touched (invisible to the eye).")
    est_bits = 2 * changed          # a random payload bit flips an LSB with probability 1/2
    st.info(f"Estimated payload from the difference: about {est_bits:,} bits (~{est_bits // 8:,} bytes), "
            f"roughly {100 * est_bits / total:.0f} % of the LSB capacity. "
            "(Each random payload bit flips the LSB only half of the time, hence changed x 2.)")

    # detection scan, cover (before) vs stego (after)
    fc, fs = cover.flatten(), stego.flatten()
    lc, pc = sc.progressive_chi(fc)
    ls, ps = sc.progressive_chi(fs)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(lc / fc.size * 100, pc, "g.-", label="cover (before)")
    ax.plot(ls / fs.size * 100, ps, "r.-", label="stego (after)")
    ax.set_xlabel("% of pixel stream (raster order)")
    ax.set_ylabel("chi-square p")
    ax.set_title("Detection scan: before vs after")
    ax.legend()
    show_fig(fig)

    # which bit planes changed
    per_plane = [int(np.count_nonzero((xor >> b) & 1)) for b in range(8)]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(range(8), per_plane)
    ax.set_xlabel("bit plane (0 = LSB)")
    ax.set_ylabel("samples changed")
    ax.set_title("Which bit planes were modified")
    show_fig(fig)

    # difference map
    dmap = (np.clip(diff * 255, 0, 255)).astype(np.uint8)
    st.image(dmap, caption=f"Difference (x255): {changed:,} samples changed", use_container_width=True)

    # LSB planes
    a, b = st.columns(2)
    a.image(lsb_plane(cover), caption="Cover LSB plane")
    b.image(lsb_plane(stego), caption="Stego LSB plane")

    # histograms
    hc, hs = np.bincount(fc, minlength=256), np.bincount(fs, minlength=256)
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    ax[0].bar(range(256), hc, width=1.0); ax[0].set_title("Cover histogram")
    ax[1].bar(range(256), hs, width=1.0, color="tab:orange"); ax[1].set_title("Stego histogram")
    ax[2].bar(range(256), hs.astype(int) - hc.astype(int), width=1.0, color="tab:red")
    ax[2].set_title("Histogram difference (stego - cover)")
    plt.tight_layout()
    show_fig(fig)


def _text_diff_html(orig, ext):
    import difflib, html
    sm = difflib.SequenceMatcher(None, orig, ext, autojunk=False)
    out = []
    for op, a0, a1, b0, b1 in sm.get_opcodes():
        o, e = html.escape(orig[a0:a1]), html.escape(ext[b0:b1])
        if op == "equal":
            out.append(e)
        else:
            if o:
                out.append(f"<span style='background:#7a1f1f;text-decoration:line-through'>{o}</span>")
            if e:
                out.append(f"<span style='background:#1f6f3a'>{e}</span>")
    return "<pre style='white-space:pre-wrap'>" + "".join(out) + "</pre>", sm.ratio()


def render_payload_diff(kind, data, key):
    """Original secret vs extracted payload."""
    st.markdown("**Original secret vs extracted payload**")
    if kind == sc.KIND_TEXT:
        orig = st.text_area("Paste the original secret message", key=f"po_t_{key}")
        if not orig:
            st.caption("Paste the original message to compare it with the extracted one.")
            return
        ext = data.decode("utf-8", errors="replace")
        html_diff, ratio = _text_diff_html(orig, ext)
        c = st.columns(4)
        c[0].metric("Exact match", "YES" if orig.encode("utf-8") == data else "NO")
        c[1].metric("Similarity", f"{ratio * 100:.2f} %")
        c[2].metric("Original length", f"{len(orig.encode('utf-8'))} B")
        c[3].metric("Extracted length", f"{len(data)} B")
        if orig.encode("utf-8") != data:
            st.markdown(html_diff, unsafe_allow_html=True)
            st.caption("Red/struck-through = in the original only, green = in the extracted only.")
        else:
            st.success("Extracted message is identical to the original, bit for bit.")
    else:
        f = st.file_uploader("Upload the original secret image", type=IMG_TYPES, key=f"po_i_{key}")
        if f is None:
            st.caption("Upload the original secret image to compare it with the extracted one.")
            return
        try:
            ext_img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            st.warning("Extracted payload could not be decoded as an image.")
            return
        org_img = Image.open(f).convert("RGB")
        resized = org_img.size != ext_img.size
        if resized:
            org_img = org_img.resize(ext_img.size)
        o, e = np.array(org_img), np.array(ext_img)
        q, diff = sc.quality(o, e)
        c = st.columns(4)
        c[0].metric("Exact match", "YES" if np.array_equal(o, e) else "NO")
        c[1].metric("MSE", f"{q['MSE']:.6f}")
        c[2].metric("PSNR", "inf dB" if np.isinf(q["PSNR (dB)"]) else f"{q['PSNR (dB)']:.2f} dB")
        c[3].metric("SSIM", f"{q['SSIM']:.6f}")
        if resized:
            st.caption("Original was resized to the extracted size (the app may shrink secrets to fit capacity).")
        a, b, d = st.columns(3)
        a.image(o, caption="Original secret")
        b.image(e, caption="Extracted secret")
        d.image(np.clip(diff * 255, 0, 255).astype(np.uint8), caption="Difference (x255)")


def render_diff_section(img, cover, payloads):
    st.subheader("Difference analysis")
    t1, t2 = st.tabs(["Cover vs stego", "Original secret vs extracted"])
    with t1:
        if cover is None:
            st.info("Provide the original cover image (uploader above) to see what embedding changed.")
        elif cover.shape != img.shape:
            st.warning("Cover and suspect image must have the same size and colour mode.")
        else:
            render_cover_diff(cover, img)
    with t2:
        if not payloads:
            st.info("No payload was extracted, so there is nothing to compare.")
        for i, (kind, data) in enumerate(payloads):
            render_payload_diff(kind, data, f"{i}")


def render_extraction(img, meta=None, cover=None):
    results = sc.extract_all(img, password)
    payloads = []
    if results:
        for i, r in enumerate(results):
            label = "Legacy MATLAB prototype format" if r["tech"] == "legacy" else sc.TECH_LABEL[r["tech"]]
            st.success(f"Hidden data found  —  {label}")
            if r["tech"] == "dip":
                st.caption("Extraction rebuilt the same texture map from this image (variance + Sobel + Laplacian → "
                           "5x5 energy → Otsu → morphology), followed the same embedding order, read the LSBs "
                           "and verified the CRC32. No side information was needed.")
            if not r["crc_ok"]:
                if r["encrypted"]:
                    st.warning("The payload is encrypted and its integrity check failed: "
                               "enter the correct password in the sidebar.")
                else:
                    st.warning("Integrity check failed: the payload may be corrupted.")
                continue
            render_payload(r["kind"], r["data"], f"k{i}")
            payloads.append((r["kind"], r["data"]))
            st.caption(f"Payload size: {r['length']} bytes" + (" · decrypted with your password" if r["encrypted"] else ""))
        render_diagnostics(img, meta, expanded=False)
        render_diff_section(img, cover, payloads)
        return results

    st.error("No hidden message in this app's own formats was found.")
    st.write("Running the **universal scan** for payloads written by other tools "
             "(other channels, bit planes, bit orders, length headers, embedded PNGs)...")
    hits = sc.universal_scan(img)
    if hits:
        for i, h in enumerate(hits):
            st.success(f"Payload found by universal scan: {h['where']}")
            render_payload(h["kind"], h["data"], f"u{i}")
            payloads.append((h["kind"], h["data"]))
        render_diagnostics(img, meta, expanded=False)
        render_diff_section(img, cover, payloads)
        return [{"crc_ok": True, "tech": "universal", "kind": hits[0]["kind"], "length": hits[0]["length"]}]

    st.warning("Nothing recognisable was found. Check the diagnostics below: the most common causes are "
               "a colour-quantised / re-compressed file, a password-keyed scatter embed with a different "
               "password, or a tool that uses its own container format.")
    render_diagnostics(img, meta, expanded=True)
    render_diff_section(img, cover, [])
    return []



# ============================================================================ proposed method page

def _sample_photo():
    try:
        import matplotlib, os
        p = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "sample_data", "grace_hopper.jpg")
        im = Image.open(p).convert("RGB")
        return np.array(im.resize((384, int(384 * im.height / im.width))))
    except Exception:
        return None


def _show_gray(ax, x, title, binary=False):
    x = np.asarray(x, dtype=np.float32)
    if not binary:
        top = np.percentile(x, 99)
        x = np.clip(x / top, 0, 1) if top > 0 else x
    ax.imshow(x, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _proposed_metrics(cover, stego, smooth):
    q, _ = sc.quality(cover, stego)
    chg = stego != cover
    m = sc.analyze(stego)
    level, _, _ = sc.verdict(m)
    return {"PSNR (dB)": q["PSNR (dB)"], "SSIM": q["SSIM"], "Samples changed": int(chg.sum()),
            "Changes inside SMOOTH areas (%)": 100.0 * float((chg & smooth).sum()) / max(1, int(chg.sum())),
            "RS rate estimate": np.nan if m["rs_rate"] is None else m["rs_rate"],
            "Chi-square p (first window)": m["p_first"], "Detector suspicion": level}


def page_proposed():
    st.header("Proposed method: DIP-guided adaptive LSB")
    st.markdown(
        "**Existing method:** LSB replacement writes the message into pixels in raster order, so noise lands in smooth "
        "areas, where it is visible and where statistical detectors pick it up.  \n"
        "**Proposal:** let image processing decide *where* to hide. Find the regions of high texture energy and hide the "
        "message there first. The receiver recomputes the same map, so nothing extra is transmitted.")
    with st.expander("The algorithm, step by step", expanded=False):
        st.markdown("""
1. Remove the LSB plane: `base = img & 0xFE`. The map uses only these 7 planes, which embedding never changes, so sender and receiver get **identical** maps.
2. Local variance (3x3 uniform filter), Sobel gradient magnitude and Laplacian response, each normalised to [0,1].
3. Fuse them: `complexity = (variance + Sobel + Laplacian) / 3`.
4. **Regional texture energy:** average `complexity` over a 5x5 window (low-pass filter). This rates a *region*, not a single edge pixel.
5. **Otsu thresholding** splits the energy map into smooth and textured regions automatically (no hand-tuned threshold).
6. **Morphological opening then closing** removes isolated specks and fills holes, giving a clean textured region.
7. Embedding order: samples inside the region first (strongest energy first), then the rest. Embedding is plain LSB replacement.
8. Extraction: rebuild the map from the received image, follow the same order, read the LSBs, verify the CRC32.

**Why it helps:** the same number of bits changes the same number of samples, but they now sit where natural variation already hides them.
**Cost:** the safe (textured) capacity is smaller than the full capacity and depends on the image content.
""")

    use_sample = st.checkbox("Use built-in sample photo (Grace Hopper)", value=False)
    cover, _ = pick_image("Cover image", "prop_cover", allow_demo=True)
    if cover is None and use_sample:
        cover = _sample_photo()
        if cover is None:
            st.error("Built-in sample photo not found. Upload a cover image instead.")
    if cover is None:
        st.info("Upload a cover image (a real photo such as Lena works best), or tick the sample photo above.")
        return
    if cover.shape[0] * cover.shape[1] > 700 * 700:
        st.warning("Large image: the analysis may take a while. A 512x512 image is plenty.")

    # ---------------------------------------------------------------- step 1: pipeline
    st.subheader("1. The DIP pipeline decides where to hide")
    stg = sc.dip_stages(cover)
    ch = 0
    fig, ax = plt.subplots(2, 4, figsize=(14, 7.2))
    panels = [("Cover (7 MSB planes)", stg["base"][..., ch], False), ("Local variance", stg["variance"][..., ch], False),
              ("Sobel magnitude", stg["sobel"][..., ch], False), ("Laplacian response", stg["laplacian"][..., ch], False),
              ("Fused complexity", stg["complexity"][..., ch], False), ("Texture energy (5x5 mean)", stg["energy"][..., ch], False),
              (f"Otsu threshold (T = {float(np.ravel(stg['otsu_T'])[ch]):.2f})", stg["raw_mask"][..., ch], True),
              ("After morphology = hiding region", stg["mask"][..., ch], True)]
    for a, (t, x, b) in zip(ax.ravel(), panels):
        _show_gray(a, x, t, b)
    plt.tight_layout()
    show_fig(fig)
    mask = stg["mask"]
    smooth = ~mask
    full_cap = sc.capacity_bytes(cover, "seq")
    sec_cap = sc.secure_capacity_bytes(cover, "dip")
    c = st.columns(3)
    c[0].metric("Textured (hiding) region", f"{100 * mask.mean():.1f} % of samples")
    c[1].metric("Safe capacity (inside region)", f"{sec_cap:,} B")
    c[2].metric("Full capacity", f"{full_cap:,} B")

    # ---------------------------------------------------------------- step 2: embed with both
    st.subheader("2. Same message, two methods")
    mode = st.radio("Payload", ["Text message", "Random data (% of full capacity)"], horizontal=True)
    if mode == "Text message":
        msg = st.text_area("Secret message", "DIP_PROJECT_SECRET_KEY_2026", key="prop_msg")
        data, kind = msg.encode("utf-8"), sc.KIND_TEXT
    else:
        pct = st.slider("Payload size (% of full capacity)", 1, 90, 20)
        data = np.random.default_rng(7).integers(0, 256, max(8, int(full_cap * pct / 100)), dtype=np.uint8).tobytes()
        kind = sc.KIND_IMAGE
    if not data:
        st.info("Enter a message.")
        return
    try:
        s_base, _ = sc.embed(cover, data, kind, "seq", "")
        s_prop, _ = sc.embed(cover, data, kind, "dip", "")
    except ValueError as e:
        st.error(str(e))
        return
    if len(data) > sec_cap:
        st.warning(f"The payload ({len(data):,} B) is larger than the safe capacity ({sec_cap:,} B). The proposed method "
                   "spills into smooth areas after filling the textured region, so its advantage shrinks.")
    else:
        st.success(f"The payload ({len(data):,} B) fits entirely inside the textured region ({sec_cap:,} B safe capacity).")

    def overlay(stego):
        g = sc.to_gray(cover).astype(np.float32) * 0.45
        rgb = np.stack([g, g, g], -1)
        chg = (stego != cover)
        chg = chg.any(axis=2) if chg.ndim == 3 else chg
        rgb[chg] = [255, 40, 40]
        return rgb.astype(np.uint8)

    a, b = st.columns(2)
    for col, name, stg_img in ((a, "Baseline: LSB replacement (sequential)", s_base), (b, "Proposed: DIP-guided adaptive LSB", s_prop)):
        with col:
            st.markdown(f"**{name}**")
            st.image(stg_img, caption="Stego image")
            st.image(overlay(stg_img), caption="Where the changes landed (red)")
            st.image(lsb_plane(stg_img), caption="LSB plane")

    mb, mp = _proposed_metrics(cover, s_base, smooth), _proposed_metrics(cover, s_prop, smooth)
    ex_b = any(r["crc_ok"] and r["data"] == data for r in sc.extract_all(s_base, ""))
    ex_p = any(r["crc_ok"] and r["data"] == data and r["tech"] == "dip" for r in sc.extract_all(s_prop, ""))
    mb["Extracted exactly"], mp["Extracted exactly"] = ex_b, ex_p
    tbl = pd.DataFrame({"Baseline (LSB)": mb, "Proposed": mp})
    st.markdown("**Comparison table**")
    st.dataframe(tbl.astype(str), use_container_width=True)
    st.caption("Higher PSNR/SSIM = less visible. Lower RS estimate and 'LOW' suspicion = harder to detect. "
               "'Changes inside SMOOTH areas' = share of modified samples that fell in the smooth region of the map.")

    # ---------------------------------------------------------------- step 3: payload sweep
    st.subheader("3. Effect of payload size")
    fr = st.multiselect("Payload sizes (% of full capacity)", [5, 10, 20, 30, 40, 60, 80], default=[5, 10, 20, 40, 60], key="prop_fr")
    if st.button("Run payload sweep") and fr:
        rows = []
        bar = st.progress(0.0)
        techs = {"Baseline LSB": "seq", "Proposed": "dip"}
        total = len(fr) * len(techs)
        k = 0
        for f in sorted(fr):
            d = np.random.default_rng(f).integers(0, 256, max(8, int(full_cap * f / 100)), dtype=np.uint8).tobytes()
            for nm, t in techs.items():
                k += 1
                bar.progress(k / total, text=f"{nm} @ {f}%")
                sg, _ = sc.embed(cover, d, sc.KIND_IMAGE, t, "")
                q, _ = sc.quality(cover, sg)
                chg = sg != cover
                rs = sc.rs_analysis(sg)
                rows.append({"Method": nm, "Payload %": f, "PSNR (dB)": q["PSNR (dB)"], "SSIM": q["SSIM"],
                             "Changes in smooth areas (%)": 100.0 * float((chg & smooth).sum()) / max(1, int(chg.sum())),
                             "RS estimate": np.nan if rs is None else rs})
        bar.empty()
        st.session_state["prop_sweep"] = pd.DataFrame(rows)
    sw = st.session_state.get("prop_sweep")
    if sw is not None:
        fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
        for nm, d in sw.groupby("Method"):
            ax[0].plot(d["Payload %"], 1 - d["SSIM"], "o-", label=nm)
            ax[1].plot(d["Payload %"], d["Changes in smooth areas (%)"], "o-", label=nm)
            ax[2].plot(d["Payload %"], d["RS estimate"], "o-", label=nm)
        ax[0].set(title="Visible distortion: 1 - SSIM (lower is better)", xlabel="payload % of capacity", yscale="log")
        ax[1].set(title="Changes landing in smooth areas (%)", xlabel="payload % of capacity")
        ax[2].set(title="RS detector estimate (lower is better)", xlabel="payload % of capacity")
        ax[0].legend()
        plt.tight_layout()
        show_fig(fig)
        st.dataframe(sw.round(5), hide_index=True)
        st.download_button("Download sweep (CSV)", sw.to_csv(index=False).encode(), "proposed_sweep.csv", "text/csv")

    st.subheader("Summary")
    st.markdown("""
- **Problem:** plain LSB replacement writes in raster order, so noise lands in smooth areas where it is visible and easier to detect.
- **Idea:** choose the pixels with a DIP pipeline (variance + Sobel + Laplacian, 5x5 low-pass, Otsu, morphology) that the receiver can recompute from the image itself.
- **Expected behavior:** PSNR should be close to sequential LSB for the same payload; changes may be shifted away from smooth areas when the payload fits the textured capacity.
- **Evidence:** the table and sweep describe this cover only. Run `evaluate_proposed.py` across several covers and detector seeds before making general performance claims; lower detection accuracy is not guaranteed.
- **Limitation:** textured capacity is image-dependent. Once it is exceeded, the method spills into smooth areas and its advantage can shrink.
""")


# ============================================================================ pages
def page_embed(kind):
    st.header("Embed Text" if kind == sc.KIND_TEXT else "Embed Image")
    st.caption(f"Method: **{tech_label}** (change it in the sidebar). Use page 1 for the full baseline-vs-proposed comparison.")
    gray = st.checkbox("Convert cover to grayscale (as in the MATLAB prototype)")
    cover, _ = pick_image("Cover image", "cover", allow_demo=True, force_gray=gray)

    if kind == sc.KIND_TEXT:
        message = st.text_area("Secret message", "DIP_PROJECT_SECRET_KEY_2026")
        secret_file, secret_gray, secret_bytes = None, False, b""
    else:
        secret_file = st.file_uploader("Secret image to hide", type=IMG_TYPES, key="secret")
        secret_gray = st.checkbox("Store secret as grayscale (smaller, fits better)", value=True)
        secret_bytes = secret_file.getvalue() if secret_file is not None else b""
        if password:
            st.caption("Note: the password encrypts text payloads only.")

    if cover is None:
        st.info("Upload a cover image (or tick the demo cover in the sidebar).")
        return
    st.caption(f"Capacity with **{tech_label}**: {sc.capacity_bytes(cover, tech):,} bytes "
               f"({cover.shape[1]}×{cover.shape[0]}, {'gray' if cover.ndim == 2 else 'RGB'})")
    if tech == "dip":
        stg = sc.dip_stages(cover)
        sec = sc.secure_capacity_bytes(cover, tech)
        st.info(f"Proposed method: the message is hidden in the textured region first "
                f"({100 * stg['mask'].mean():.1f} % of samples, safe capacity {sec:,} bytes). "
                "Smooth areas are left untouched while the payload fits.")
        with st.expander("Show the hiding region chosen by the DIP pipeline"):
            a, b = st.columns(2)
            a.image(cover, caption="Cover")
            b.image((stg["mask"].any(axis=2) if stg["mask"].shape[2] > 1 else stg["mask"][..., 0]).astype(np.uint8) * 255,
                    caption="Hiding region (white = textured, used first)")
    else:
        st.caption("Baseline: bits are written pixel by pixel in raster order, smooth areas included. "
                   "Switch the sidebar to the proposed method to compare.")
    signature = _embed_signature(kind, cover, tech_label, message if kind == sc.KIND_TEXT else "",
                                 password, secret_bytes, secret_gray)

    if st.button("Embed", type="primary"):
        try:
            if kind == sc.KIND_TEXT:
                data = message.encode("utf-8")
                if not data:
                    st.error("Message is empty.")
                    return
                note = None
            else:
                if secret_file is None:
                    st.error("Upload the secret image first.")
                    return
                data, size = sc.fit_secret_image(Image.open(io.BytesIO(secret_bytes)), sc.capacity_bytes(cover, tech), secret_gray)
                note = f"Secret image stored at {size[0]}×{size[1]} ({len(data):,} bytes as PNG)."
            stego, info = sc.embed(cover, data, kind, tech, password)
        except ValueError as e:
            st.error(str(e))
            return
        st.session_state["embed_result"] = {"kind": kind, "cover": cover, "stego": stego, "info": info,
                                            "png": sc.to_png_bytes(stego), "tech": tech_label, "note": note,
                                            "signature": signature}

    res = st.session_state.get("embed_result")
    current_result = res and res["kind"] == kind and res.get("signature") == signature
    if res and res["kind"] == kind and not current_result:
        st.info("Embedding settings changed. Click **Embed** to create a result with the current method and inputs.")
    if current_result:
        st.success(f"Stego image created with {res['tech']}.")
        if res["note"]:
            st.info(res["note"])
        i = res["info"]
        st.caption(f"Payload {i['payload_bytes']:,} B of {i['capacity_bytes']:,} B capacity · "
                   f"{i['changed_samples_pct']:.3f}% of samples changed")
        st.download_button("⬇️ Download stego image (PNG)", res["png"], "stego_image.png", "image/png")
        show_quality(res["cover"], res["stego"])
        st.caption("Embedding changes are intentionally tiny. Use the high-contrast changed-sample map above to see where bits were written.")
        if st.checkbox("Run steganalysis on this stego image"):
            render_analysis(res["stego"], res["cover"])


def optional_cover(suspect):
    with st.expander("Optional: original cover image (adds MSE / PSNR / SSIM and difference maps)"):
        f = st.file_uploader("Original cover", type=IMG_TYPES, key="orig_cover")
        if f is None:
            return None
        c = sc.load_image(f)
        if c.shape != suspect.shape:
            st.warning("Cover and suspect image must have the same size and colour mode.")
            return None
        return c


def pick_suspect(label, key):
    if "embed_result" in st.session_state and st.checkbox("Use the stego image I just created", key=key + "_use"):
        return st.session_state["embed_result"]["stego"], None
    return pick_image(label, key)


def page_analyze():
    st.header("Detect (Analyze Image)")
    img, meta = pick_suspect("Image to analyze", "an")
    if img is None:
        st.info("Upload an image to analyze.")
        return
    cover = optional_cover(img)
    if cover is not None:
        show_quality(cover, img)
    render_analysis(img, cover)


def page_extract():
    st.header("Extract Hidden Information")
    img, meta = pick_suspect("Image to extract from", "ex")
    if img is None:
        st.info("Upload a stego image.")
        return
    st.image(img, caption="Input image")
    cover = optional_cover(img)
    render_extraction(img, meta, cover)


# ============================================================================ router
{OPS[0]: page_proposed,
 OPS[1]: lambda: page_embed(sc.KIND_TEXT),
 OPS[2]: lambda: page_embed(sc.KIND_IMAGE),
 OPS[3]: page_analyze,
 OPS[4]: page_extract}[op]()
