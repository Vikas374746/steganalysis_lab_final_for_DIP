"""
watermark.py  -  classical DIP tool for removing a VISIBLE watermark / logo from your own image.

Pipeline (all classical image processing, no AI model):
    1. Detect the watermark pixels across a single image, or inside a chosen box:
       background estimate  = large median filter           (order-statistic filter)
       difference           = |image - background|
       threshold            = Otsu inside the box            (automatic thresholding)
       clean-up             = morphological closing + dilation (so soft edges are covered)
  2. Inpaint the masked pixels:
       pull-push fill      : multi-scale normalised Gaussian convolution (low-pass filtering)
       diffusion smoothing : Jacobi iterations of the Laplace equation inside the mask
       texture restore     : add noise matched to the noise level around the hole
Use only on images you own or have permission to edit.
"""
import numpy as np
from scipy import ndimage

import stego_core as sc


def build_mask(img, box, mode="auto", median_size=15, dilate=2):
    """box = (x0, y0, x1, y1) in pixels.  mode 'auto': detect watermark pixels; 'box': use the whole box."""
    x0, y0, x1, y1 = box
    mask = np.zeros(img.shape[:2], bool)
    if x1 <= x0 or y1 <= y0:
        return mask
    if mode == "box":
        mask[y0:y1, x0:x1] = True
        return mask
    g = img.astype(np.float32).mean(axis=2) if img.ndim == 3 else img.astype(np.float32)
    bg = ndimage.median_filter(g, size=median_size)
    reg = np.abs(g - bg)[y0:y1, x0:x1]
    top = float(np.percentile(reg, 99.5))
    if top <= 0:
        return mask
    t = sc._otsu(np.clip(reg / top, 0, 1).ravel()) * top
    sel = reg > t
    mask[y0:y1, x0:x1] = sel
    mask = ndimage.binary_closing(mask, np.ones((3, 3), bool))
    if dilate > 0:
        mask = ndimage.binary_dilation(mask, np.ones((3, 3), bool), iterations=dilate)
    return mask


def detect_single_mask(img, sensitivity=2.5, median_size=21, dilate=1):
    """Estimate a watermark mask over the full image using local median residuals."""
    source = img[..., None] if img.ndim == 2 else img
    source = source.astype(np.float32)
    residual = np.max(np.abs(source - ndimage.median_filter(source, size=(median_size, median_size, 1))), axis=2)
    top = float(np.percentile(residual, 99.5))
    if top <= 0:
        return np.zeros(img.shape[:2], bool)
    clipped = np.clip(residual / top, 0, 1)
    otsu_threshold = sc._otsu(clipped.ravel()) * top
    center = float(np.median(residual))
    spread = 1.4826 * float(np.median(np.abs(residual - center)))
    percentile = float(np.clip(96.5 + (float(sensitivity) - 1.0) * 0.75, 0, 100))
    threshold = max(otsu_threshold, center + float(sensitivity) * max(spread, 1.0),
                    float(np.percentile(residual, percentile)))
    mask = residual > threshold
    mask = ndimage.binary_closing(mask, np.ones((3, 3), bool))
    if dilate > 0:
        mask = ndimage.binary_dilation(mask, np.ones((3, 3), bool), iterations=dilate)
    labels, count = ndimage.label(mask)
    if count:
        sizes = np.bincount(labels.ravel())
        keep = sizes >= 3
        keep[0] = False
        mask = keep[labels]
    return mask


def inpaint(img, mask, smooth_iters=60, add_noise=True, seed=0):
    """Fill masked pixels from their surroundings. Returns uint8 image of the same shape."""
    if not mask.any():
        return img.copy()
    two_d = img.ndim == 2
    cur = (img[..., None] if two_d else img).astype(np.float32).copy()
    filled = ~mask
    for sigma in (1.5, 3, 6, 12, 24, 48, 96, 192):        # pull-push: coarse scales fill larger holes
        todo = ~filled
        if not todo.any():
            break
        kn = filled.astype(np.float32)
        den = ndimage.gaussian_filter(kn, sigma)
        ok = todo & (den > 0.02)
        for c in range(cur.shape[2]):
            num = ndimage.gaussian_filter(cur[..., c] * kn, sigma)
            cur[..., c][ok] = num[ok] / den[ok]
        filled = filled | ok
    k = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float32) / 4.0
    for _ in range(smooth_iters):                          # diffusion (Laplace) inside the hole
        for c in range(cur.shape[2]):
            avg = ndimage.convolve(cur[..., c], k, mode="nearest")
            cur[..., c][mask] = avg[mask]
    if add_noise:                                          # restore natural grain so the patch is not too smooth
        rng = np.random.default_rng(seed)
        ring = ndimage.binary_dilation(mask, np.ones((3, 3), bool), iterations=8) & ~mask
        if ring.any():
            src = img[..., None] if two_d else img
            for c in range(cur.shape[2]):
                x = src[..., c].astype(np.float32)
                hp = x - ndimage.gaussian_filter(x, 1.5)
                sd = float(hp[ring].std())
                cur[..., c][mask] += rng.normal(0, 0.8 * sd, int(mask.sum()))
    out = np.clip(np.rint(cur), 0, 255).astype(np.uint8)
    return out[..., 0] if two_d else out


def naive_blur_fill(img, mask, sigma=6):
    """Reference method for comparison: plain Gaussian blur pasted into the hole."""
    two_d = img.ndim == 2
    x = (img[..., None] if two_d else img).astype(np.float32)
    out = x.copy()
    for c in range(x.shape[2]):
        out[..., c][mask] = ndimage.gaussian_filter(x[..., c], sigma)[mask]
    out = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    return out[..., 0] if two_d else out


def add_demo_watermark(img, text="(c) DEMO", alpha=0.75):
    """Draw a semi-transparent text watermark (for testing). Returns (watermarked, box)."""
    from PIL import Image, ImageDraw, ImageFont
    H, W = img.shape[:2]
    base = Image.fromarray(img if img.ndim == 3 else np.stack([img] * 3, -1)).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    size = max(14, H // 9)
    try:
        font = ImageFont.load_default(size=size)
    except Exception:
        font = ImageFont.load_default()
    d = ImageDraw.Draw(layer)
    l, t, r, b = d.textbbox((0, 0), text, font=font)
    x, y = (W - (r - l)) // 2 - l, int(H * 0.72) - t
    d.text((x, y), text, font=font, fill=(255, 255, 255, int(255 * alpha)))
    out = np.array(Image.alpha_composite(base, layer).convert("RGB"))
    if img.ndim == 2:
        out = np.array(Image.fromarray(out).convert("L"))
    pad = 6
    box = (max(0, x + l - pad), max(0, y + t - pad), min(W, x + r + pad), min(H, y + b + pad))
    return out, box


# =====================================================================================================
# Automatic watermark detection from MANY images that carry the same watermark
# (method of Dekel et al., "On the Effectiveness of Visible Watermarks", CVPR 2017; same idea as the
#  rohitrango/automatic-watermark-detection project, re-implemented here with NumPy/SciPy only.)
#
# Model:   J_k = alpha * W + (1 - alpha) * I_k        J = watermarked image k, I = its clean original
# Idea:    the watermark is identical in every image, the photo content is not.  Therefore the MEDIAN over
#          the images of the image gradients keeps the watermark edges and cancels the photo content.
# Steps:   1 median gradient  ->  2 locate the watermark (Otsu + morphology)  ->  3 Poisson integration
#          of the cleaned gradients gives Q = alpha*(W - mu)  ->  4 alpha from the image statistics
#          ->  5 invert the blending model for every image:  I = mu + (J - mu - Q) / (1 - alpha)
# Use only on images you own or have permission to edit.
# =====================================================================================================
from scipy import fft as _fft


def group_by_shape(images):
    """Keep the largest group of images that have identical size (the watermark must sit at the same pixels)."""
    groups = {}
    for im in images:
        groups.setdefault(im.shape, []).append(im)
    shape, members = max(groups.items(), key=lambda kv: len(kv[1]))
    return np.stack(members), len(images)


def _rows(n, step=48):
    return [(r, min(n, r + step)) for r in range(0, n, step)]


def _median_gradients(stack):
    N, H, W, C = stack.shape
    mx = np.zeros((H, W - 1, C), np.float32)
    my = np.zeros((H - 1, W, C), np.float32)
    for r0, r1 in _rows(H):
        blk = stack[:, r0:r1].astype(np.float32)
        mx[r0:r1] = np.median(blk[:, :, 1:] - blk[:, :, :-1], axis=0)
        blk2 = stack[:, r0:min(H, r1 + 1)].astype(np.float32)
        d = blk2[:, 1:] - blk2[:, :-1]
        my[r0:r0 + d.shape[1]] = np.median(d, axis=0)
    return mx, my


def _per_pixel(stack, fn):
    N, H, W, C = stack.shape
    out = np.zeros((H, W, C), np.float32)
    for r0, r1 in _rows(H):
        out[r0:r1] = fn(stack[:, r0:r1].astype(np.float32))
    return out


def _detect_box(mx, my, pad=8):
    H, W = mx.shape[0], my.shape[1]
    mag = np.zeros((H, W), np.float32)
    mag[:, :-1] += (mx ** 2).sum(2)
    mag[:-1, :] += (my ** 2).sum(2)
    mag = np.sqrt(mag)
    top = float(np.percentile(mag, 99.5))
    t = max(sc._otsu(np.clip(mag / max(top, 1e-6), 0, 1).ravel()) * top, 3.0 * float(np.median(mag)))
    m = mag > t
    m = ndimage.binary_dilation(ndimage.binary_closing(m, np.ones((3, 3), bool)), np.ones((3, 3), bool), iterations=6)
    lab, n = ndimage.label(m)
    if n == 0:
        return None, mag
    mass = ndimage.sum(mag, lab, index=np.arange(1, n + 1))
    keep = [i + 1 for i, v in enumerate(mass) if v >= 0.25 * mass.max()]
    ys, xs = np.where(np.isin(lab, keep))
    return (max(0, xs.min() - pad), max(0, ys.min() - pad), min(W, xs.max() + 1 + pad), min(H, ys.max() + 1 + pad)), mag


def _poisson_dirichlet(gx, gy):
    """Solve  laplace(Q) = div(g)  with Q = 0 outside the box (exact DST-I solver)."""
    h, w = gx.shape
    div = gx.copy()
    div[:, 1:] -= gx[:, :-1]
    div += gy
    div[1:, :] -= gy[:-1, :]
    p = np.arange(1, h + 1)[:, None]
    q = np.arange(1, w + 1)[None, :]
    lam = 2 * np.cos(np.pi * p / (h + 1)) + 2 * np.cos(np.pi * q / (w + 1)) - 4
    D = _fft.dstn(div.astype(np.float64), type=1, norm="ortho")
    return _fft.dstn(D / lam, type=1, norm="ortho").astype(np.float32)


def _fill_outside(arr, box):
    """Replace the box region of a H x W (x C) array by a smooth fill computed from the pixels outside."""
    x0, y0, x1, y1 = box
    mask = np.zeros(arr.shape[:2], bool)
    mask[y0:y1, x0:x1] = True
    two_d = arr.ndim == 2
    a = (arr[..., None] if two_d else arr).astype(np.float32)
    out = a.copy()
    filled = ~mask
    for sigma in (2, 4, 8, 16, 32, 64, 128):
        todo = ~filled
        if not todo.any():
            break
        kn = filled.astype(np.float32)
        den = ndimage.gaussian_filter(kn, sigma)
        ok = todo & (den > 0.02)
        for c in range(a.shape[2]):
            num = ndimage.gaussian_filter(out[..., c] * kn, sigma)
            out[..., c][ok] = num[ok] / den[ok]
        filled = filled | ok
    return out[..., 0] if two_d else out


def estimate_collection(stack, thr_k=3.0, pad=8, box=None):
    """stack: uint8 (N,H,W,3), all showing the same watermark at the same place.  Returns an estimate dict."""
    N, H, W, C = stack.shape
    mx, my = _median_gradients(stack)
    det_box, mag = _detect_box(mx, my, pad)
    box = box or det_box
    if box is None:
        raise ValueError("No consistent watermark found. Use more images, or images that really share one watermark.")
    x0, y0, x1, y1 = box
    inside = np.zeros((H, W), bool)
    inside[y0:y1, x0:x1] = True
    h, w = y1 - y0, x1 - x0

    # noise level of the median gradient (outside the watermark) -> hard-threshold small gradients (sparsity)
    outx, outy = ~inside[:, :-1], ~inside[:-1, :]
    Q = np.zeros((h, w, C), np.float32)
    for c in range(C):
        sg = 1.4826 * float(np.median(np.abs(np.r_[mx[..., c][outx], my[..., c][outy]])))
        tau = thr_k * max(sg, 1e-3)
        gx = np.zeros((h, w), np.float32)
        gy = np.zeros((h, w), np.float32)
        gx[:, :w - 1] = mx[y0:y1, x0:x1 - 1, c]
        gy[:h - 1, :] = my[y0:y1 - 1, x0:x1, c]
        gx[np.abs(gx) < tau] = 0
        gy[np.abs(gy) < tau] = 0
        Q[..., c] = _poisson_dirichlet(gx, gy)

    med = _per_pixel(stack, lambda b: np.median(b, axis=0))
    sd = _per_pixel(stack, lambda b: b.std(axis=0))
    mu = _fill_outside(med, box)[y0:y1, x0:x1]            # local typical background inside the box
    sref = _fill_outside(sd, box)[y0:y1, x0:x1]
    a_raw = np.clip(1.0 - sd[y0:y1, x0:x1] / np.maximum(sref, 1e-3), 0, 1).mean(axis=2)

    # alpha: project Q on its principal colour direction (Q = alpha * (W - mu)), calibrate the scale on the statistics
    Qf = Q.reshape(-1, C)
    u = np.linalg.svd(Qf, full_matrices=False)[2][0]
    p = Qf @ u
    if float((p * a_raw.ravel()).sum()) < 0:
        p, u = -p, -u
    s = float((p * a_raw.ravel()).sum() / max((p * p).sum(), 1e-6))
    alpha = np.clip(s * p, 0, 0.9).reshape(h, w)
    alpha[alpha < 0.03] = 0
    return {"box": box, "Q": Q, "alpha": alpha, "mu": mu, "mag": mag, "n": N, "shape": (H, W, C)}


def remove_with_estimate(img, est):
    """Undo the estimated watermark on one image (must have the same size as the collection)."""
    if img.shape != est["shape"]:
        raise ValueError(f"image size {img.shape} differs from the collection {est['shape']}")
    x0, y0, x1, y1 = est["box"]
    out = img.astype(np.float32).copy()
    J = out[y0:y1, x0:x1]
    a = est["alpha"][..., None]
    out[y0:y1, x0:x1] = est["mu"] + (J - est["mu"] - est["Q"]) / (1.0 - a)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def demo_collection(n=40, size=256, text="SAMPLE", alpha=0.5, seed=0, source=None):
    """Simulated collection for testing: n different pictures, same watermark at the same place.
    Returns (watermarked list, clean list, true box)."""
    from PIL import Image
    rng = np.random.default_rng(seed)
    pics = []
    for i in range(n):
        if source is not None and i % 2 == 0:
            H, W = source.shape[:2]
            s = int(rng.uniform(0.4, 1.0) * min(H, W))
            y, x = rng.integers(0, H - s + 1), rng.integers(0, W - s + 1)
            im = np.array(Image.fromarray(source[y:y + s, x:x + s]).resize((size, size)))
            im = np.rot90(im, rng.integers(0, 4))
            im = im[:, ::-1] if rng.random() < 0.5 else im
            im = np.clip(im.astype(np.float32) * rng.uniform(0.7, 1.25) + rng.uniform(-25, 25), 0, 255).astype(np.uint8)
        else:
            im = sc.synthetic_cover(size, int(rng.integers(0, 10 ** 6)))
        pics.append(np.ascontiguousarray(im))
    marked, box = [], None
    for im in pics:
        m, box = add_demo_watermark(im, text, alpha)
        marked.append(m)
    return marked, pics, box
