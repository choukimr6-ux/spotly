"""Visual analysis of a packshot frame (runs inside the OCR batch process, never in the monitoring process).

Measured layout of El-Heddaf frames (2026-10-13 captures):
  * channel bug top-right, a QR code bottom-left on every frame (ads included) - both masked;
  * programme content carries a news/score ticker with a clock ("19:43 GMT+1") in the bottom ~22%;
    paid spots run full-screen without it. A ticker clock on the packshot is therefore strong evidence
    that a recurring "asset" is programme material (replays, idents), not a commercial.

Everything here is derived from ONE frame (1.5 s before the spot ends) and is labelled as such in the UI.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BUG = (0.82, 0.0, 1.0, 0.22)          # x0, y0, x1, y1 as fractions of width/height
QR = (0.0, 0.62, 0.13, 0.86)
TICKER_TOP = 0.78
_CLOCK = re.compile(r"\b\d{1,2}[:.]\d{2}\b|gmt", re.I)
_ARABIC = re.compile(r"[؀-ۿ]")
_RTL_RUNS = re.compile(r"[A-Za-z0-9][A-Za-z0-9.,:'/&+-]*")
LOGO_DIR_NAME = "logos"


def visual_to_logical(text: str) -> str:
    """PaddleOCR emits Arabic in visual (left-to-right) order: reverse it, keeping Latin/digit runs readable."""
    if not _ARABIC.search(text):
        return text
    rev = text[::-1]
    return _RTL_RUNS.sub(lambda m: m.group(0)[::-1], rev)


# Specific programme titles only: a bare "sport" / "الرياضة" also appears in genuine gym or sportswear spots.
_PROGRAMME_TITLES = re.compile(r"أخبار\s*الرياضة|نشرة\s*الأخبار|المباراة|الدوري|مباشر\s*من", re.I)
_SCORE = re.compile(r"(?<!\d)\d{1,2}\s*[-\u2013:]\s*\d{1,2}(?!\d)")
_MATCH_MINUTE = re.compile(r"\b\d{1,3}'|\b\d{1,3}\s*\+\s*\d{1,2}'?|\b\d{1,3}\u2019")


def programme_text(boxes, width: int, height: int) -> str | None:
    """Why this frame is programme content, from raw OCR boxes (box, logical text, conf); None if it looks like a spot.

    Sports ticker bar / match score / match clock in the lower band, or a programme title ('أخبار الرياضة') anywhere.
    The ticker clock itself is programme_overlay(); this adds the text cues.
    """
    for box, text, conf in boxes:
        text = (text or "").strip()
        if not text or conf < 0.5:
            continue
        cy = sum(p[1] for p in box) / 4 / height
        if cy > TICKER_TOP and (_SCORE.search(text) or _MATCH_MINUTE.search(text)):
            return f"ticker score/match clock: {text[:40]}"
        if _PROGRAMME_TITLES.search(text) and not re.search(r"\.(?:dz|com)\b", text, re.I):
            return f"programme title: {text[:40]}"
    return None


def programme_overlay(boxes, width: int, height: int) -> bool:
    """True if the frame shows the channel's ticker clock (programme, not a paid spot)."""
    for box, text, _ in boxes:
        cy = sum(p[1] for p in box) / 4 / height
        if cy > TICKER_TOP and _CLOCK.search(text or ""):
            return True
    return False


def _mask(h: int, w: int, ticker: bool) -> np.ndarray:
    m = np.ones((h, w), bool)
    for x0, y0, x1, y1 in (BUG, QR):
        m[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = False
    if ticker:
        m[int(TICKER_TOP * h):, :] = False
    return m


def sharpness(gray: np.ndarray) -> float:
    import cv2
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def dhash(gray: np.ndarray) -> int:
    """64-bit difference hash of the central region (overlays excluded)."""
    import cv2
    h, w = gray.shape
    core = gray[int(0.1 * h):int(0.76 * h), int(0.13 * w):int(0.82 * w)]
    small = cv2.resize(core, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def salient_regions(img: np.ndarray, ticker: bool, max_regions: int = 3) -> list[tuple]:
    """Candidate logo/product regions: high-contrast, saturated blobs outside the overlays.
    -> [(x0, y0, x1, y1, score)] in pixels of `img`, best first."""
    import cv2
    h, w = img.shape[:2]
    sw, sh = 320, max(1, int(320 * h / w))
    small = cv2.GaussianBlur(cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA), (5, 5), 0)  # grain is not a logo
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    sat = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
    sal = grad / (grad.max() + 1e-6) + 0.5 * sat / 255.0
    sal[~_mask(sh, sw, ticker)] = 0
    thr = np.percentile(sal[sal > 0], 85) if np.any(sal > 0) else 1.0
    binm = (sal >= thr).astype(np.uint8)
    binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binm)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        frac = bw * bh / (sw * sh)
        if not 0.004 <= frac <= 0.35 or bw < 8 or bh < 6:
            continue
        score = float(sal[labels == i].mean()) * np.sqrt(area)
        fx, fy = w / sw, h / sh
        out.append((int(x * fx), int(y * fy), int((x + bw) * fx), int((y + bh) * fy), round(score, 2)))
    return sorted(out, key=lambda r: -r[4])[:max_regions]


@dataclass
class LogoRef:
    brand: str
    source: str                    # file it came from
    kp_desc: tuple = field(repr=False, default=None)


class LogoLibrary:
    """Reference logos: spotly/data/logos/<Brand>/*.png|jpg (supplied by the user) plus crops learned from
    packshots of assets a human classified. Matching: ORB keypoints + ratio test + RANSAC homography, so it
    tolerates scale/position changes but needs the same artwork. Nothing is fetched from the web."""

    MIN_INLIERS = 14

    def __init__(self, root: Path):
        import cv2
        self.root = Path(root)
        self.orb = cv2.ORB_create(nfeatures=600)                    # references: a logo / wordmark crop
        self.orb_frame = cv2.ORB_create(nfeatures=1500)             # packshots: features spread over the whole frame
        self.refs: list[LogoRef] = []
        if self.root.is_dir():
            dirs = []
            for d in sorted(p for p in self.root.iterdir() if p.is_dir()):
                # "_learned" (and any "_" folder) holds <Brand>/ folders one level down
                dirs += sorted(p for p in d.iterdir() if p.is_dir()) if d.name.startswith("_") else [d]
            for d in dirs:
                for f in sorted(d.glob("*")):
                    if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                        self.add(d.name, f)

    def _features(self, img, frame: bool = False):
        import cv2
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        limit = 1280 if frame else 640                          # keep packshots at capture resolution
        if max(gray.shape) > limit:
            s = limit / max(gray.shape)
            gray = cv2.resize(gray, (int(gray.shape[1] * s), int(gray.shape[0] * s)), interpolation=cv2.INTER_AREA)
        return (self.orb_frame if frame else self.orb).detectAndCompute(gray, None)

    def add(self, brand: str, path: Path) -> bool:
        import cv2
        img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return False
        kp, desc = self._features(img)
        if desc is None or len(kp) < 10:
            return False
        self.refs.append(LogoRef(brand, path.name, (kp, desc)))
        return True

    def match(self, img: np.ndarray, ticker: bool = False) -> tuple[str, float, int] | None:
        """Best (brand, confidence 0..0.95, inliers) for a packshot, or None."""
        import cv2
        if not self.refs:
            return None
        frame = img.copy()
        h, w = frame.shape[:2]
        frame[~_mask(h, w, ticker)] = 0                         # overlays never count as logo evidence
        kp, desc = self._features(frame, frame=True)
        if desc is None or len(kp) < 10:
            return None
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        best = None
        for ref in self.refs:
            rkp, rdesc = ref.kp_desc
            pairs = bf.knnMatch(rdesc, desc, k=2)
            good = [m for m, *rest in pairs if rest and m.distance < 0.75 * rest[0].distance]
            if len(good) < self.MIN_INLIERS:
                continue
            src = np.float32([rkp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            _, inl = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            n = int(inl.sum()) if inl is not None else 0
            if n >= self.MIN_INLIERS and (best is None or n > best[2]):
                best = (ref.brand, round(min(0.95, n / 40), 3), n)
        return best


def analyse_frame(img: np.ndarray, boxes, brand_boxes, library: LogoLibrary | None = None) -> dict:
    """Visual features of a packshot. `boxes` = raw OCR [(box, text, conf)], `brand_boxes` = boxes whose text
    matched a brand alias (the wordmark). -> JSON-serialisable dict."""
    import cv2
    h, w = img.shape[:2]
    ticker = programme_overlay(boxes, w, h)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    regions = salient_regions(img, ticker)
    if brand_boxes:
        # the largest single wordmark (a union of every mention on packs + wordmark would overstate it)
        rects = [(int(min(p[0] for p in b)), int(min(p[1] for p in b)), int(max(p[0] for p in b)), int(max(p[1] for p in b)))
                 for b in brand_boxes]
        logo = max(rects, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
        basis = "brand text"
    elif regions:
        logo, basis = regions[0][:4], "salient region"
    else:
        logo, basis = None, None
    area = ((logo[2] - logo[0]) * (logo[3] - logo[1]) / (w * h)) if logo else 0.0
    lib = library.match(img, ticker) if library is not None else None
    return dict(programme_overlay=ticker, sharpness=round(sharpness(gray), 1), logo_box=logo, logo_basis=basis,
                logo_area=round(area, 4), dhash=f"{dhash(gray):016x}",
                logo_match=dict(brand=lib[0], confidence=lib[1], inliers=lib[2]) if lib else None)
