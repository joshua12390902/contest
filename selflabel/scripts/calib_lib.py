"""Shared lib: floor-tape square corner extraction for camera self-calibration."""
import cv2, numpy as np
from sklearn.cluster import KMeans


def tape_mask(img, roi_y=0.45):
    H, W = img.shape[:2]; y0 = int(H * roi_y)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(int)
    floor = gray[y0:, :]
    blur = cv2.GaussianBlur(floor.astype(np.uint8), (31, 31), 0).astype(int)
    dark = ((blur - floor) > 30).astype(np.uint8) * 255
    m = np.zeros((H, W), np.uint8); m[y0:, :] = dark
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def _line_from_segs(segs):
    pts = np.vstack([[(s[0], s[1]), (s[2], s[3])] for s in segs]).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    return np.array([vy, -vx, vx * y0 - vy * x0])  # a,b,c : ax+by+c=0


def _inter(l1, l2):
    a1, b1, c1 = l1; a2, b2, c2 = l2
    d = a1 * b2 - a2 * b1
    if abs(d) < 1e-6: return None
    return np.array([(b1 * c2 - b2 * c1) / d, (a2 * c1 - a1 * c2) / d])


def extract_quad(img, roi_y=0.45, thr=70, minlen=110, gap=30):
    m = tape_mask(img, roi_y)
    segs = cv2.HoughLinesP(m, 1, np.pi / 180, threshold=thr, minLineLength=minlen, maxLineGap=gap)
    if segs is None: return None, m, []
    segs = [s[0] for s in segs]
    ang = np.array([np.degrees(np.arctan2(s[3] - s[1], s[2] - s[0])) % 180 for s in segs])
    L = np.array([np.hypot(s[2] - s[0], s[3] - s[1]) for s in segs])
    feat = np.c_[np.cos(2 * np.radians(ang)), np.sin(2 * np.radians(ang))]
    ka = KMeans(2, n_init=5, random_state=0).fit(feat, sample_weight=L)
    lines = []
    for g in range(2):
        gi = np.where(ka.labels_ == g)[0]
        if len(gi) == 0: return None, m, []
        a = np.radians(ang[gi].mean())
        mid = np.array([((segs[i][0] + segs[i][2]) / 2, (segs[i][1] + segs[i][3]) / 2) for i in gi])
        proj = mid @ np.array([-np.sin(a), np.cos(a)])
        if len(gi) == 1:
            lines.append(_line_from_segs([segs[gi[0]]])); lines.append(_line_from_segs([segs[gi[0]]]))
            continue
        kr = KMeans(2, n_init=5, random_state=0).fit(proj.reshape(-1, 1), sample_weight=L[gi])
        for h in range(2):
            sel = gi[kr.labels_ == h]
            if len(sel) == 0: return None, m, []
            lines.append(_line_from_segs([segs[i] for i in sel]))
    if len(lines) != 4: return None, m, []
    corners = [_inter(lines[i], lines[j]) for i in (0, 1) for j in (2, 3)]
    corners = [c for c in corners if c is not None]
    if len(corners) != 4: return None, m, []
    c = np.array(corners); cen = c.mean(0)
    order = np.argsort(np.arctan2(c[:, 1] - cen[1], c[:, 0] - cen[0]))
    return c[order], m, lines
