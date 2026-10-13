"""
stego_core.py  -  engine for the Image Steganalysis Lab (no GUI code in here).

Embedding techniques : LSB replacement (sequential), LSB matching (+-1), keyed scatter LSB,
                       2-bit LSB, edge-adaptive LSB
Detection            : chi-square (Westfeld-Pfitzmann), RS analysis (Fridrich), LSB statistics,
                       Laplacian / Sobel / Fourier / Haar-wavelet features
Extraction           : tries every technique, a legacy-MATLAB format, and a blind printable-text scan
"""
import hashlib
import io
import os
import struct
import zlib

import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from PIL import Image
from scipy import ndimage
from scipy.stats import chi2

# ----------------------------------------------------------------------------- constants
MAGIC = b"STG2"
HEADER_LEN = 14                       # MAGIC(4) + kind(1) + encrypted(1) + length(4) + crc32(4)
KIND_TEXT, KIND_IMAGE = 0, 1
ENC_NONE, ENC_LEGACY_XOR, ENC_AES_GCM = 0, 1, 2
AEAD_OVERHEAD = 16 + 12 + 16          # salt + nonce + GCM tag
DIP_FEATURES = ("variance", "sobel", "laplacian")

TECHNIQUES = {
    "LSB Replacement (sequential)": "seq",
    "LSB Matching (±1)": "match",
    "Keyed Scatter LSB": "scatter",
    "2-bit LSB (two planes)": "lsb2",
    "Edge-Adaptive LSB": "edge",
    "DIP-Guided Adaptive LSB (proposed)": "dip",
}
TECH_LABEL = {v: k for k, v in TECHNIQUES.items()}
TECH_LABEL["seq"] = "LSB sequential (Replacement or Matching - identical to read back)"


# ----------------------------------------------------------------------------- image I/O
def load_image(src, force_gray=False):
    """Return uint8 array: 2-D for grayscale, (H,W,3) for colour. Alpha is dropped.
    16-bit grayscale is rescaled to 0-255 (not clipped)."""
    im = Image.open(src)
    if im.mode.startswith("I;16"):
        # 16-bit values run 0..65535; dividing by 257 maps 65535 -> 255
        gray = np.array(im, dtype=np.float64) / 257.0
        return np.clip(np.round(gray), 0, 255).astype(np.uint8)
    if force_gray or im.mode in ("L", "1", "I", "F"):
        im = im.convert("L")
    else:
        im = im.convert("RGB")
    return np.array(im, dtype=np.uint8)



def to_png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def to_gray(img):
    if img.ndim == 2:
        return img
    return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).round().astype(np.uint8)


def synthetic_cover(size=384, seed=0):
    """Textured photo-like test image: multi-scale (1/f) noise + mild sensor noise.
    Only for trying the app without files - use real photos for your actual results."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((size, size, 3))
    for sigma, amp in ((2, 10), (4, 16), (8, 26), (16, 40), (32, 60)):
        layer = ndimage.gaussian_filter(rng.normal(size=(size, size, 3)), (sigma, sigma, 0))
        acc += amp * layer / layer.std()
    acc = acc - acc.mean() + 128 + rng.normal(0, 1.5, acc.shape)
    return np.clip(acc.round(), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- password helpers
def _keystream(password, n):
    key = hashlib.sha256(("stego-key:" + password).encode()).digest()
    out, ctr = bytearray(), 0
    while len(out) < n:
        out += hashlib.sha256(key + ctr.to_bytes(8, "big")).digest()
        ctr += 1
    return np.frombuffer(bytes(out[:n]), dtype=np.uint8)


def xor_crypt(data, password):
    """Decode legacy XOR-encrypted payloads. New payloads use authenticated encryption."""
    if not data:
        return data
    return (np.frombuffer(data, dtype=np.uint8) ^ _keystream(password, len(data))).tobytes()


def _payload_key(password, salt):
    return Scrypt(salt=salt, length=32, n=2**14, r=8, p=1).derive(password.encode("utf-8"))


def _payload_aad(kind, encrypted, length, crc):
    return MAGIC + bytes([kind, encrypted]) + struct.pack(">II", length, crc)


def _encrypt_payload(data, password, kind, crc):
    salt, nonce = os.urandom(16), os.urandom(12)
    length = len(data) + AEAD_OVERHEAD
    aad = _payload_aad(kind, ENC_AES_GCM, length, crc)
    ciphertext = AESGCM(_payload_key(password, salt)).encrypt(nonce, data, aad)
    return salt + nonce + ciphertext


def _decrypt_payload(body, password, header):
    if len(body) < AEAD_OVERHEAD:
        raise InvalidTag
    salt, nonce = body[:16], body[16:28]
    aad = _payload_aad(header["kind"], header["enc"], header["length"], header["crc"])
    return AESGCM(_payload_key(password, salt)).decrypt(nonce, body[28:], aad)


def _scatter_seed(password):
    return int.from_bytes(hashlib.sha256(("scatter:" + password).encode()).digest()[:8], "big")


# ----------------------------------------------------------------------------- carrier positions
def edge_strength(img):
    """Sobel magnitude computed on the 7 MSB planes only, so embedding cannot change it."""
    base = (img & 0xFE).astype(np.float32)
    if base.ndim == 2:
        base = base[..., None]
    mag = np.zeros(base.shape, np.float32)
    for c in range(base.shape[2]):
        gx = ndimage.sobel(base[..., c], axis=1)
        gy = ndimage.sobel(base[..., c], axis=0)
        mag[..., c] = np.hypot(gx, gy)
    return mag.reshape(-1)


# ----------------------------------------------------------------------------- PROPOSED METHOD
# DIP-guided adaptive LSB.  Where to hide is decided by an image-processing pipeline that runs on
# the 7 most-significant bit planes only (img & 0xFE).  LSB replacement never changes those planes,
# so the receiver rebuilds exactly the same map from the stego image, with no side information.
#
#   limitation of plain LSB : every pixel is used in raster order, so noise lands in smooth areas
#                             where the eye and statistical detectors notice it most.
#   proposal                : hide in REGIONS of high texture energy (fused local variance + Sobel +
#                             Laplacian, averaged over a 5x5 window), find the region with Otsu
#                             thresholding, clean it with morphological opening/closing, and fill it
#                             strongest-texture-first.  Smooth areas are touched last (or never).
def _otsu(values, bins=256):
    """Otsu threshold of values in [0,1] (maximises between-class variance)."""
    hist, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    p = hist.astype(np.float64) / max(1, hist.sum())
    w0 = np.cumsum(p)
    mu = np.cumsum(p * np.arange(bins))
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mu[-1] * w0 - mu) ** 2 / (w0 * (1.0 - w0))
    return float(edges[int(np.argmax(np.nan_to_num(sb))) + 1])


def _norm01(x):
    top = np.percentile(x, 99)
    return np.clip(x / top, 0.0, 1.0) if top > 0 else np.zeros_like(x)


def dip_stages(img, components=DIP_FEATURES):
    """Every intermediate image of the proposed pipeline, per colour channel (shape H x W x C)."""
    components = tuple(components)
    if not components or any(name not in DIP_FEATURES for name in components):
        raise ValueError(f"components must be a non-empty subset of {DIP_FEATURES}")
    base = (img & 0xFE).astype(np.float32)               # 7 MSB planes: unchanged by embedding
    if base.ndim == 2:
        base = base[..., None]
    names = ("base", "variance", "sobel", "laplacian", "complexity", "energy", "otsu_T", "raw_mask", "mask")
    out = {k: [] for k in names}
    st3 = np.ones((3, 3), bool)
    for ch in range(base.shape[2]):
        b = base[..., ch]
        m1 = ndimage.uniform_filter(b, 3)
        var = np.clip(ndimage.uniform_filter(b * b, 3) - m1 * m1, 0, None)          # 1. local variance (3x3)
        sob = np.hypot(ndimage.sobel(b, axis=1), ndimage.sobel(b, axis=0))          # 2. Sobel gradient magnitude
        lap = np.abs(ndimage.laplace(b))                                            # 3. Laplacian response
        features = {"variance": var, "sobel": sob, "laplacian": lap}
        comp = np.mean([_norm01(features[name]) for name in components], axis=0)  # 4. feature fusion
        energy = ndimage.uniform_filter(comp, 5)                                    # 5. regional texture energy (5x5 mean)
        t = _otsu(energy.ravel())                                                   # 6. Otsu: smooth vs textured
        raw = energy > t
        mask = ndimage.binary_closing(ndimage.binary_opening(raw, st3), st3)        # 7. morphology: clean the region
        for k, v in zip(names, (b, var, sob, lap, comp, energy, np.full(b.shape, t), raw, mask)):
            out[k].append(v)
    res = {k: np.stack(v, axis=-1) for k, v in out.items()}
    res["otsu_T"] = res["otsu_T"][0, 0]
    return res


def dip_order(img, components=DIP_FEATURES):
    """Embedding order: textured region first, strongest texture energy first; smooth samples last."""
    s = dip_stages(img, components)
    energy = s["energy"].reshape(-1)
    mask = s["mask"].reshape(-1)
    return np.lexsort((-energy, ~mask)), int(mask.sum())   # primary key: in-mask first


def secure_capacity_bytes(img, tech, dip_components=DIP_FEATURES):
    """Payload that fits inside the textured (low-risk) region only. Other techniques: full capacity."""
    if tech != "dip":
        return capacity_bytes(img, tech)
    return max(0, (dip_order(img, dip_components)[1] - HEADER_LEN * 8) // 8)


def carrier_order(img, tech, password="", dip_components=DIP_FEATURES):
    """(indices into img.flatten() in embedding order, bits stored per index)."""
    n = img.size
    if tech in ("seq", "match"):
        return np.arange(n), 1
    if tech == "lsb2":
        return np.arange(n), 2
    if tech == "scatter":
        return np.random.default_rng(_scatter_seed(password)).permutation(n), 1
    if tech == "edge":
        return np.argsort(-edge_strength(img), kind="stable"), 1   # strongest edges first
    if tech == "dip":
        return dip_order(img, dip_components)[0], 1
    raise ValueError(f"unknown technique {tech}")


def capacity_bytes(img, tech):
    bpp = 2 if tech == "lsb2" else 1
    return max(0, (img.size * bpp - HEADER_LEN * 8) // 8)


# ----------------------------------------------------------------------------- embedding
def build_payload(data, kind, password=""):
    encrypted = ENC_AES_GCM if (password and kind == KIND_TEXT) else ENC_NONE
    crc = zlib.crc32(data) & 0xFFFFFFFF
    body = _encrypt_payload(data, password, kind, crc) if encrypted else data
    return MAGIC + bytes([kind, encrypted]) + struct.pack(">II", len(body), crc) + body


def embed(img, data, kind, tech, password="", rng_seed=None, dip_components=DIP_FEATURES):
    """Returns (stego_array, info_dict)."""
    payload = build_payload(data, kind, password)
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    flat = img.flatten()
    order, bpp = carrier_order(img, tech, password, dip_components)
    need = -(-len(bits) // bpp)
    if need > len(order):
        raise ValueError(f"Payload needs {len(bits)} bits but this technique only offers {len(order) * bpp}.")
    idx = order[:need]

    if bpp == 2:
        padded = np.zeros(need * 2, np.uint8)
        padded[:len(bits)] = bits
        pairs = padded[0::2] * 2 + padded[1::2]
        flat[idx] = (flat[idx] & 0xFC) | pairs
    elif tech == "match":
        rng = np.random.default_rng(rng_seed)
        v = flat[idx].astype(np.int16)
        step = np.where(rng.random(len(idx)) < 0.5, 1, -1)
        step = np.where(v == 0, 1, np.where(v == 255, -1, step))
        v = np.where((v & 1) != bits, v + step, v)
        flat[idx] = v.astype(np.uint8)
    else:
        flat[idx] = (flat[idx] & 0xFE) | bits

    stego = flat.reshape(img.shape)
    info = {"payload_bytes": len(data), "total_bits": len(bits),
            "capacity_bytes": capacity_bytes(img, tech),
            "changed_samples_pct": 100.0 * float(np.mean(stego != img))}
    return stego, info


def fit_secret_image(secret_pil, max_bytes, gray=False):
    """Shrink a secret image until its PNG encoding fits the available capacity."""
    im = secret_pil.convert("L" if gray else "RGB")
    scale = 1.0
    for _ in range(40):
        w, h = max(1, int(im.width * scale)), max(1, int(im.height * scale))
        buf = io.BytesIO()
        im.resize((w, h), Image.LANCZOS).save(buf, format="PNG", optimize=True)
        if buf.tell() <= max_bytes:
            return buf.getvalue(), (w, h)
        scale *= 0.85
    raise ValueError("Cover image is too small to hold even a tiny secret image.")


# ----------------------------------------------------------------------------- extraction
def read_bits(flat, order, bpp, nbits):
    npix = -(-nbits // bpp)
    vals = flat[order[:npix]]
    if bpp == 1:
        bits = vals & 1
    else:
        bits = np.stack([(vals >> 1) & 1, vals & 1], axis=1).reshape(-1)
    return bits[:nbits]


def _parse_header(hb):
    if len(hb) < HEADER_LEN or hb[:4] != MAGIC:
        return None
    length, crc = struct.unpack(">II", hb[6:14])
    return {"kind": hb[4], "enc": hb[5], "length": length, "crc": crc}


def try_extract(img, tech, password="", dip_components=DIP_FEATURES):
    flat = img.flatten()
    order, bpp = carrier_order(img, tech, password, dip_components)
    if len(order) * bpp < HEADER_LEN * 8:
        return None
    head = _parse_header(np.packbits(read_bits(flat, order, bpp, HEADER_LEN * 8)).tobytes())
    if head is None:
        return None
    total = (HEADER_LEN + head["length"]) * 8
    if total > len(order) * bpp:
        return None
    bits = read_bits(flat, order, bpp, total)
    body = np.packbits(bits[HEADER_LEN * 8:]).tobytes()
    try:
        if head["enc"] == ENC_AES_GCM:
            data = _decrypt_payload(body, password, head)
        elif head["enc"] == ENC_LEGACY_XOR:
            data = xor_crypt(body, password)
        elif head["enc"] == ENC_NONE:
            data = body
        else:
            return None
    except InvalidTag:
        data = b""
    return {"tech": tech, "kind": head["kind"], "encrypted": bool(head["enc"]),
            "length": len(data) if head["enc"] == ENC_AES_GCM else head["length"], "data": data,
            "crc_ok": (zlib.crc32(data) & 0xFFFFFFFF) == head["crc"]}


def _printable_ratio(b):
    return sum(32 <= c < 127 or c in (9, 10, 13) for c in b) / len(b) if b else 0.0


def try_extract_legacy_matlab(img):
    """Format of the original MATLAB prototype: gray image, column-major order,
    32-bit MSB-first length header, then message bytes MSB-first."""
    if img.ndim != 2 or img.size < 40:
        return None
    bits = img.flatten(order="F") & 1
    n = int("".join(map(str, bits[:32])), 2)
    if n == 0 or 32 + n * 8 > bits.size:
        return None
    data = np.packbits(bits[32:32 + n * 8]).tobytes()
    if _printable_ratio(data) < 0.9:
        return None
    return {"tech": "legacy", "kind": KIND_TEXT, "encrypted": False, "length": n,
            "data": data, "crc_ok": True}


def extract_all(img, password="", dip_components=DIP_FEATURES):
    """Try every known technique. Returns a list of successful (header-found) results."""
    results = []
    for tech in ("seq", "scatter", "lsb2", "edge", "dip"):
        r = try_extract(img, tech, password, dip_components)
        if r:
            results.append(r)
    legacy = try_extract_legacy_matlab(img)
    if legacy:
        results.append(legacy)
    return results


def blind_scan(img, nbytes=200):
    """No header known: rank channel / bit-order guesses by fraction of printable ASCII."""
    cands = []
    planes = [("all samples", img.flatten())]
    if img.ndim == 3:
        planes += [(n, img[..., i].flatten()) for i, n in enumerate(("R", "G", "B"))]
    else:
        planes.append(("column-major", img.flatten(order="F")))
    for name, flat in planes:
        bits = (flat[: nbytes * 8] & 1).astype(np.uint8)
        for order in ("big", "little"):
            b = np.packbits(bits, bitorder=order).tobytes()
            cands.append((_printable_ratio(b), name, "MSB-first" if order == "big" else "LSB-first", b))
    cands.sort(key=lambda t: t[0], reverse=True)
    return cands


# ----------------------------------------------------------------------------- quality metrics
def ssim(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    if a.ndim == 3:
        return float(np.mean([ssim(a[..., c], b[..., c]) for c in range(a.shape[2])]))
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = ndimage.uniform_filter(a, 7), ndimage.uniform_filter(b, 7)
    var_a = ndimage.uniform_filter(a * a, 7) - mu_a ** 2
    var_b = ndimage.uniform_filter(b * b, 7) - mu_b ** 2
    cov = ndimage.uniform_filter(a * b, 7) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return float(s.mean())


def quality(cover, stego):
    diff = np.abs(cover.astype(np.float64) - stego.astype(np.float64))
    mse = float(np.mean(diff ** 2))
    psnr = float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)
    return {"MSE": mse, "PSNR (dB)": psnr, "SSIM": ssim(cover, stego)}, diff


# ----------------------------------------------------------------------------- detection
def chi_square(samples):
    """Westfeld-Pfitzmann. Returns (statistic, dof, p). High p => value pairs equalised => stego."""
    h = np.bincount(samples, minlength=256).astype(float)
    even, odd = h[0::2], h[1::2]
    exp = (even + odd) / 2.0
    ok = exp >= 5
    if ok.sum() < 2:
        return 0.0, 0, 0.0
    stat = float(np.sum((even[ok] - exp[ok]) ** 2 / exp[ok]))
    dof = int(ok.sum() - 1)
    return stat, dof, float(chi2.sf(stat, dof))


def progressive_chi(flat, steps=40):
    lengths = np.linspace(flat.size / steps, flat.size, steps).astype(int)
    lengths = lengths[lengths >= 4096]
    return lengths, np.array([chi_square(flat[:L])[2] for L in lengths])


def _disc(G):
    return np.abs(np.diff(G, axis=1)).sum(axis=1)


def _rs_d(chan, flipped):
    x = chan.astype(np.int16).ravel()
    if flipped:
        x = x ^ 1
    x = x[: (x.size // 4) * 4].reshape(-1, 4)
    f0 = _disc(x)
    d = []
    for fn in (lambda v: v ^ 1, lambda v: np.clip(((v + 1) ^ 1) - 1, 0, 255)):
        y = x.copy()
        y[:, 1:3] = fn(x[:, 1:3])
        f1 = _disc(y)
        d.append(float(np.mean(f1 > f0) - np.mean(f1 < f0)))
    return d                                   # [R_M - S_M, R_-M - S_-M]


def rs_analysis(img):
    """Fridrich RS analysis: estimated LSB-replacement rate in [0,1] (bits per sample), or None."""
    chans = [img] if img.ndim == 2 else [img[..., c] for c in range(img.shape[2])]
    est = []
    for ch in chans:
        d0, dm0 = _rs_d(ch, False)
        d1, dm1 = _rs_d(ch, True)
        a, b, c = 2 * (d1 + d0), (dm0 - dm1 - d1 - 3 * d0), (d0 - dm0)
        if abs(a) < 1e-12:
            continue
        disc = b * b - 4 * a * c
        if disc < 0:
            continue
        roots = [(-b + s * np.sqrt(disc)) / (2 * a) for s in (1, -1)]
        x = min(roots, key=abs)
        if abs(x - 0.5) < 1e-9:
            continue
        est.append(x / (x - 0.5))
    if not est:
        return None
    return float(np.clip(np.mean(est), 0.0, 1.0))


def haar2(g):
    g = g[: g.shape[0] // 2 * 2, : g.shape[1] // 2 * 2].astype(np.float64)
    a, b, c, d = g[0::2, 0::2], g[0::2, 1::2], g[1::2, 0::2], g[1::2, 1::2]
    return (a + b + c + d) / 2, (a + b - c - d) / 2, (a - b + c - d) / 2, (a - b - c + d) / 2


def analyze(img):
    flat = img.flatten()
    gray = to_gray(img).astype(np.float64)

    chi_stat, dof, p_all = chi_square(flat)
    first = max(8192, flat.size // 10)
    p_first = chi_square(flat[:first])[2]
    lengths, ps = progressive_chi(flat)
    high = np.where(ps > 0.05)[0]

    lsb = flat & 1
    p1 = float(lsb.mean())
    entropy = 0.0 if p1 in (0.0, 1.0) else float(-(p1 * np.log2(p1) + (1 - p1) * np.log2(1 - p1)))
    transition = float(np.mean(lsb[1:] != lsb[:-1]))

    lap = ndimage.convolve(gray, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float), mode="nearest")
    sob = np.hypot(ndimage.sobel(gray, axis=1), ndimage.sobel(gray, axis=0))
    edge_pct = float(np.mean(sob > sob.mean() + sob.std()))

    M, N = gray.shape
    u, v = np.meshgrid(np.arange(-(N // 2), N - N // 2), np.arange(-(M // 2), M - M // 2))
    D = np.sqrt(u ** 2 + v ** 2)
    Hhp = 1.0 / (1.0 + (30.0 / (D + np.finfo(float).eps)) ** 4)          # Butterworth high-pass, D0=30, n=2
    F = np.fft.fftshift(np.fft.fft2(gray))
    hp = np.real(np.fft.ifft2(np.fft.ifftshift(F * Hhp)))

    LL, LH, HL, HH = haar2(gray)
    rs = rs_analysis(img)

    return {
        "chi_stat": chi_stat, "dof": dof, "p_all": p_all, "p_first": p_first,
        "lengths": lengths, "ps": ps,
        "chi_est_bytes": int(lengths[high[-1]]) // 8 if len(high) else 0,
        "rs_rate": rs,
        "lsb_ones_pct": 100 * p1, "lsb_entropy": entropy, "lsb_transition": transition,
        "hf_spatial": float(np.mean(lap ** 2)), "hf_fft": float(np.mean(hp ** 2)),
        "wav_H": float(np.mean(LH ** 2)), "wav_V": float(np.mean(HL ** 2)), "wav_D": float(np.mean(HH ** 2)),
        "edge_pct": 100 * edge_pct,
        "_lap": lap, "_sobel": sob, "_fft": np.log1p(np.abs(F)), "_hp": hp,
        "_haar": (LL, LH, HL, HH), "_gray": gray,
    }


def verdict(m):
    """Combine the detectors into a suspicion level. Returns (level, headline, reasons)."""
    reasons, score = [], 0
    if m["p_first"] > 0.5:
        score += 1
        reasons.append(f"Chi-square: value pairs are equalised at the start of the pixel stream "
                       f"(p = {m['p_first']:.3f}) - the signature of sequential LSB replacement.")
    else:
        reasons.append(f"Chi-square: no pair equalisation at the start of the stream (p = {m['p_first']:.3f}).")
    if m["rs_rate"] is not None and m["rs_rate"] > 0.08:
        score += 1
        reasons.append(f"RS analysis: estimated embedding rate {m['rs_rate']:.2f} bits/sample "
                       f"(LSB replacement, sequential or scattered).")
    else:
        r = "n/a" if m["rs_rate"] is None else f"{m['rs_rate']:.3f}"
        reasons.append(f"RS analysis: no significant LSB-replacement rate (estimate {r}).")
    level = ["LOW", "MEDIUM", "HIGH"][score]
    head = {"LOW": "No statistical evidence of LSB replacement",
            "MEDIUM": "Some statistical evidence of LSB embedding",
            "HIGH": "Strong statistical evidence of LSB embedding"}[level]
    return level, head, reasons


# ============================================================================= diagnostics
def describe_file(raw):
    """Facts about the uploaded file that decide whether LSB data can possibly have survived."""
    im = Image.open(io.BytesIO(raw))
    fmt, mode = im.format, im.mode
    w, h = im.size
    try:
        rgb = np.array(im.convert("RGB"), dtype=np.uint32)
    except Exception:   # e.g. 16-bit / float modes on some Pillow versions
        rgb = np.array(im.convert("L").convert("RGB"), dtype=np.uint32)
    uniq = int(np.unique((rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]).size)
    warns = []
    if fmt in ("JPEG", "MPO"):
        warns.append("JPEG is lossy: LSB payloads do not survive JPEG compression.")
    if mode in ("RGB", "RGBA") and uniq <= 256 and w * h > 65536:
        warns.append(f"Only {uniq} distinct colours in a {w}x{h} colour image: it has been "
                     "colour-quantised/posterised, which wipes out LSB payloads.")
    if mode in ("RGBA", "LA", "PA"):
        warns.append("The file has an alpha channel; this app ignores it.")
    return {"format": fmt, "mode": mode, "width": w, "height": h, "file_kb": len(raw) / 1024,
            "colours": uniq, "warnings": warns}


def plane_profile(img, planes=4):
    """Per bit plane: share of ones and how often horizontal neighbours agree.
    Noise-like planes (about 50 % / 50 %) may carry data; structured planes do not."""
    ch = img if img.ndim == 2 else img[..., 0]
    rows = []
    for p in range(planes):
        pl = (ch >> p) & 1
        rows.append({"bit plane": f"{p + 1}" + (" (LSB)" if p == 0 else ""),
                     "ones %": round(100 * float(pl.mean()), 2),
                     "neighbours equal %": round(100 * float(np.mean(pl[:, 1:] == pl[:, :-1])), 2)})
    return rows


def probe_headers(img, password=""):
    """What each extractor reads where it expects the 14-byte header."""
    rows = []
    flat = img.flatten()
    for tech in ("seq", "scatter", "lsb2", "edge", "dip"):
        order, bpp = carrier_order(img, tech, password)
        hb = np.packbits(read_bits(flat, order, bpp, HEADER_LEN * 8)).tobytes()
        rows.append({"extractor": TECH_LABEL[tech], "first bytes read": hb.hex(" "),
                     "marker 'STG2' found": "yes" if hb[:4] == MAGIC else "no"})
    return rows


# ============================================================================= universal scan
PROBE_BYTES = 256
_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_PNG_END = b"IEND\xaeB`\x82"


def _streams(img):
    out = [("row-major, all channels", img.flatten())]
    if img.ndim == 3:
        out += [(f"row-major, {n} channel only", img[..., i].flatten()) for i, n in enumerate("RGB")]
        t = np.transpose(img, (1, 0, 2))
        out.append(("column-major, all channels", t.flatten()))
        out += [(f"column-major, {n} channel only", t[..., i].flatten()) for i, n in enumerate("RGB")]
    else:
        out.append(("column-major", img.flatten(order="F")))
    return out


def _bits_from(flat, k, within):
    planes = range(k - 1, -1, -1) if within == "msb" else range(k)
    return np.stack([(flat >> p) & 1 for p in planes], axis=1).reshape(-1).astype(np.uint8)


def _classify(b):
    """Does this probe look like the start of a known payload layout? -> (kind, offset, length)"""
    for off in (0, 4, 8):
        if b[off:off + 8] == _PNG_SIG:
            return "image", off, None
    for width, fmt in ((4, "I"), (2, "H")):
        for endian in (">", "<"):
            n = struct.unpack(endian + fmt, b[:width])[0]
            if n < (4 if width == 4 else 8):
                continue
            body = b[width:width + min(n, 64)]
            if len(body) >= 4 and _printable_ratio(body) >= 0.97:
                return "text", width, n
    run = 0                                   # plain text with no header: long leading printable run
    for c in b:
        if 32 <= c < 127 or c in (9, 10, 13):
            run += 1
        else:
            break
    if run >= 16:
        return "text", 0, None
    return None


def _recover(flat, k, within, byte_order, kind, off, n):
    bits = _bits_from(flat, k, within)
    data = np.packbits(bits[: len(bits) // 8 * 8], bitorder=byte_order).tobytes()
    if kind == "image":
        end = data.find(_PNG_END, off)
        if end < 0:
            return None
        payload = data[off:end + len(_PNG_END)]
        try:
            Image.open(io.BytesIO(payload)).verify()
        except Exception:
            return None
        return payload
    if n is not None:
        return data[off:off + n] if off + n <= len(data) else None
    body = data[off:]
    cut = next((i for i, c in enumerate(body) if not (32 <= c < 127 or c in (9, 10, 13))), len(body))
    return body[:cut] if cut >= 4 else None


def universal_scan(img, max_hits=6):
    """Format-agnostic search for payloads written by OTHER tools: tries every channel choice,
    row/column order, 1-4 low bit planes, and both bit orders, and looks for a PNG signature,
    a length-prefixed text, or plain printable text."""
    hits, seen = [], set()
    variants = [(1, "msb")] + [(k, w) for k in (2, 3, 4) for w in ("msb", "lsb")]
    for k, within in variants:
        for sname, flat in _streams(img):
            if flat.size * k < PROBE_BYTES * 8:
                continue
            probe_bits = _bits_from(flat[: PROBE_BYTES * 8], k, within)[: PROBE_BYTES * 8]
            for bo in ("big", "little"):
                probe = np.packbits(probe_bits, bitorder=bo).tobytes()
                cl = _classify(probe)
                if cl is None:
                    continue
                kind, off, n = cl
                data = _recover(flat, k, within, bo, kind, off, n)
                if not data:
                    continue
                digest = hashlib.md5(data).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                hits.append({"kind": KIND_IMAGE if kind == "image" else KIND_TEXT, "data": data,
                             "where": f"{sname}, lowest {k} bit plane{'s' if k > 1 else ''}"
                                      + (f" ({within.upper()}-first within sample)" if k > 1 else "")
                                      + f", {bo}-endian bit packing",
                             "length": len(data)})
                if len(hits) >= max_hits:
                    return hits
    return hits


# ----------------------------------------------------------------------------- trained steganalyser (for evaluation)
def spam_features(img, T=2):
    """SPAM-style feature vector: co-occurrences of three consecutive clipped neighbour differences
    (horizontal and vertical, every channel).  Noise from embedding changes this distribution."""
    g = img.astype(np.int16)
    if g.ndim == 2:
        g = g[..., None]
    feats = []
    for ch in range(g.shape[2]):
        x = g[..., ch]
        for d in (x[:, 1:] - x[:, :-1], (x[1:, :] - x[:-1, :]).T):
            d = np.clip(d, -T, T) + T
            idx = d[:, :-2] * (2 * T + 1) ** 2 + d[:, 1:-1] * (2 * T + 1) + d[:, 2:]
            feats.append(np.bincount(idx.ravel(), minlength=(2 * T + 1) ** 3) / idx.size)
    return np.mean(feats, axis=0)


def _lda_accuracy(Xc_tr, Xs_tr, Xc_te, Xs_te):
    Xtr = np.vstack([Xc_tr, Xs_tr]); ytr = np.r_[np.zeros(len(Xc_tr)), np.ones(len(Xs_tr))]
    Xte = np.vstack([Xc_te, Xs_te]); yte = np.r_[np.zeros(len(Xc_te)), np.ones(len(Xs_te))]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    m0, m1 = A[ytr == 0].mean(0), A[ytr == 1].mean(0)
    Sw = np.cov(A[ytr == 0].T) + np.cov(A[ytr == 1].T) + 0.1 * np.eye(A.shape[1])
    w = np.linalg.solve(Sw, m1 - m0)
    b = -0.5 * w @ (m0 + m1)
    return float((((B @ w + b) > 0) == (yte > 0)).mean())


def detector_accuracy(cover, techs, fractions, size=96, max_crops=500, seed=0, progress=None,
                      dip_components=DIP_FEATURES):
    """Train a linear (LDA) detector on SPAM features of cover-vs-stego crops of ONE image and test it on
    crops from a different part of the image (top half <-> bottom half, no overlap).
    Returns ({tech: [accuracy per fraction]}, number_of_crops_per_class) or (None, 0) if the image is too small.
    Accuracy 0.5 = detector is guessing; 1.0 = always detects."""
    img = cover if cover.ndim == 3 else cover[..., None]
    H, W = img.shape[:2]
    if H < 2 * size or W < size:
        return None, 0
    rng = np.random.default_rng(seed)
    stride = max(8, size // 4)
    pos = [(y, x) for y in range(0, H - size + 1, stride) for x in range(0, W - size + 1, stride)]
    if len(pos) > max_crops:
        pos = [pos[i] for i in rng.choice(len(pos), max_crops, replace=False)]
    top = [p for p in pos if p[0] + size <= H // 2]
    bot = [p for p in pos if p[0] >= H // 2]
    if len(top) < 20 or len(bot) < 20:
        return None, 0
    crops = {p: img[p[0]:p[0] + size, p[1]:p[1] + size].copy() for p in top + bot}
    cover_f = {p: spam_features(crops[p]) for p in crops}
    variants = []
    for item in techs:
        if isinstance(item, str):
            variants.append((item, item, dip_components))
        else:
            name, technique, components = item
            variants.append((name, technique, tuple(components)))
    res = {name: [] for name, _, _ in variants}
    total, done = len(fractions) * len(variants), 0
    for fr in fractions:
        data_rng = np.random.default_rng(seed + int(fr * 1000))
        stego_f = {name: {} for name, _, _ in variants}
        for p, c in crops.items():
            n = max(8, int(capacity_bytes(c, "seq") * fr))
            data = data_rng.integers(0, 256, n, dtype=np.uint8).tobytes()
            for index, (name, technique, components) in enumerate(variants):
                match_seed = seed + int(fr * 1000) * 1_000_003 + p[0] * W + p[1]
                stego_f[name][p] = spam_features(embed(
                    c, data, KIND_IMAGE, technique, "", rng_seed=match_seed + index,
                    dip_components=components
                )[0])
        for name, _, _ in variants:
            accs = []
            for tr, te in ((top, bot), (bot, top)):
                accs.append(_lda_accuracy(
                    np.array([cover_f[p] for p in tr]), np.array([stego_f[name][p] for p in tr]),
                    np.array([cover_f[p] for p in te]), np.array([stego_f[name][p] for p in te])
                ))
            res[name].append(float(np.mean(accs)))
            done += 1
            if progress:
                progress(done / total, f"detector: {name} @ {int(fr * 100)}%")
    return res, min(len(top), len(bot))
