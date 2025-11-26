import os
import glob
import cv2
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
IMG_DIR = "/home/amir-admin/projects/CNN_hand_control/datasets/images/train"    # change to your images folder
LABEL_DIR = "/home/amir-admin/projects/CNN_hand_control/datasets/labels/train"  # change to your labels folder
NUM_KEYPOINTS = 21            # 21 for hands (x, y, v)
DRAW_SKELETON = True          # set False if you only want dots
SHOW_NAMES = True            # set True to label each point name
# -----------------------------------------------------------------------------

KPT_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
]

# Simple hand skeleton connections (by indices in KPT_NAMES)
SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9,10), (10,11), (11,12),        # middle
    (0,13), (13,14), (14,15), (15,16),       # ring
    (0,17), (17,18), (18,19), (19,20)        # pinky
]

def parse_yolo_kpt_line(line, num_kpts=NUM_KEYPOINTS):
    """
    YOLO keypoint label format:
    class cx cy w h  x1 y1 v1  x2 y2 v2 ... xK yK vK
    All coords are normalized in [0,1].
    """
    parts = line.strip().split()
    if len(parts) < 5 + 3*num_kpts:
        raise ValueError(f"Label line too short for {num_kpts} keypoints:\n{line}")
    cls = int(float(parts[0]))
    cx, cy, w, h = map(float, parts[1:5])
    kpts = np.array(list(map(float, parts[5:5+3*num_kpts])), dtype=np.float32).reshape(num_kpts, 3)
    return cls, (cx, cy, w, h), kpts

def draw_overlay(img, bbox, kpts, draw_skeleton=True, show_names=False):
    H, W = img.shape[:2]

    # draw bbox
    cx, cy, bw, bh = bbox
    x1 = int((cx - bw/2) * W)
    y1 = int((cy - bh/2) * H)
    x2 = int((cx + bw/2) * W)
    y2 = int((cy + bh/2) * H)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # draw skeleton lines (only if both endpoints have v>0)
    if draw_skeleton:
        for a, b in SKELETON:
            if a < len(kpts) and b < len(kpts):
                xa, ya, va = kpts[a]
                xb, yb, vb = kpts[b]
                if va > 0 and vb > 0:
                    pa = (int(xa * W), int(ya * H))
                    pb = (int(xb * W), int(yb * H))
                    cv2.line(img, pa, pb, (255, 255, 255), 1)

    # draw points
    for i, (kx, ky, v) in enumerate(kpts):
        if v <= 0:
            continue
        px, py = int(kx * W), int(ky * H)
        color = (0, 255, 0) if int(v) == 2 else (0, 165, 255)  # visible=green, occluded=orange
        cv2.circle(img, (px, py), 3, color, -1)
        if show_names and i < len(KPT_NAMES):
            cv2.putText(img, KPT_NAMES[i], (px+4, py-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1, cv2.LINE_AA)

def find_matching_image(label_file):
    base = os.path.splitext(os.path.basename(label_file))[0]
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        p = os.path.join(IMG_DIR, base + ext)
        if os.path.isfile(p):
            return p
    # fallback: search recursively
    for ext in ["**/*.jpg", "**/*.jpeg", "**/*.png", "**/*.bmp", "**/*.webp"]:
        for p in glob.glob(os.path.join(IMG_DIR, ext), recursive=True):
            if os.path.splitext(os.path.basename(p))[0] == base:
                return p
    return None

def load_pairs():
    pairs = []
    for txt in sorted(glob.glob(os.path.join(LABEL_DIR, "*.txt"))):
        img = find_matching_image(txt)
        if img is not None:
            pairs.append((img, txt))
    return pairs

def show_image_with_labels(image_path, label_path, index, total):
    img = cv2.imread(image_path)
    if img is None:
        return None

    # overlay all objects in the label (in case there are multiple hands)
    with open(label_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    for line in lines:
        _, bbox, kpts = parse_yolo_kpt_line(line, NUM_KEYPOINTS)
        draw_overlay(img, bbox, kpts, draw_skeleton=DRAW_SKELETON, show_names=SHOW_NAMES)

    # small on-image instructions
    msg = f"[{index+1}/{total}]  ←/A: prev   →/D: next   Q/ESC: quit"
    cv2.putText(img, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
    return img

def is_left(key):
    # cover multiple platforms: arrow-left, and A/a
    return key in (81, 2424832, ord('a'), ord('A'))  # 81 (Linux), 2424832 (Windows)

def is_right(key):
    # cover multiple platforms: arrow-right, and D/d
    return key in (83, 2555904, ord('d'), ord('D'))  # 83 (Linux), 2555904 (Windows)

def main():
    pairs = load_pairs()
    if not pairs:
        print("No (image, label) pairs found. Check IMG_DIR and LABEL_DIR.")
        return

    i = 0
    win = "YOLO Keypoint Viewer"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        img = show_image_with_labels(pairs[i][0], pairs[i][1], i, len(pairs))
        if img is None:
            print(f"Failed to read image: {pairs[i][0]}")
            break

        cv2.imshow(win, img)
        key = cv2.waitKeyEx(0)  # wait indefinitely, capture special keys

        if key in (27, ord('q'), ord('Q')):  # ESC or Q
            break
        elif is_right(key):
            i = (i + 1) % len(pairs)
        elif is_left(key):
            i = (i - 1) % len(pairs)
        # else: ignore other keys and keep image

    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Controls: Right/Left arrows (or D/A) to navigate, Q/ESC to quit.")
    print("Note: This viewer DOES NOT save any files.")
    main()