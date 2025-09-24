import os
from pathlib import Path
import cv2
import numpy as np

# =======================
# CONFIG — edit if needed
# =======================
VIDEO_PATH   = r"C:\Users\facti\OneDrive\Desktop\groundschool\cir.mp4"
FRAMES_DIR   = "frames_out"              # folder where extracted frames are saved
RESULT_IMAGE = "stitched_result.png"     # final stitched image
STEP         = 5                         # take every Nth frame

# ORB / matching knobs
ORB_N_FEATURES   = 8000                  # increase for more matches
RATIO_TEST_THRES = 0.75                  # Lowe's ratio test
RANSAC_REPROJ    = 3.0                   # pixels; smaller = stricter
MIN_GOOD_MATCHES = 40                    # skip if below this after ratio test
MIN_INLIERS      = 25                    # skip if fewer affine inliers

# =======================
# Utilities
# =======================
def extract_every_nth_frame(video_path: str, out_dir: str, step: int = 5) -> list:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
    i = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % step == 0:
            save_path = os.path.join(out_dir, f"frame_{i:06d}.png")
            cv2.imwrite(save_path, frame)
            frames.append(frame)
            saved += 1
        i += 1
    cap.release()
    print(f"[INFO] Extracted {saved} frames into '{out_dir}'.")
    return frames

def detect_and_match(imgA, imgB, orb, ratio=0.75):
    """ORB detect + KNN match + ratio test. Returns matched point arrays."""
    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    kpsA, desA = orb.detectAndCompute(grayA, None)
    kpsB, desB = orb.detectAndCompute(grayB, None)
    if desA is None or desB is None or len(kpsA) < 2 or len(kpsB) < 2:
        return None, None, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = bf.knnMatch(desA, desB, k=2)

    good = []
    for m, n in raw:
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < MIN_GOOD_MATCHES:
        return None, None, len(good)

    src_pts = np.float32([kpsA[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpsB[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    return src_pts, dst_pts, len(good)

def estimate_affine(src_pts, dst_pts):
    """
    Estimate A (2x3) that maps dst -> src using RANSAC (affine partial: shift/rot/scale, no perspective).
    Returns matrix and inlier count.
    """
    M, inliers = cv2.estimateAffinePartial2D(
        dst_pts, src_pts, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_REPROJ, maxIters=5000, confidence=0.995
    )
    inlier_count = int(inliers.sum()) if inliers is not None else 0
    return M, inlier_count

def warp_size_for_affine(base_shape, add_shape, M):
    """Compute the bounding canvas size and translation to fit base and warped add."""
    h1, w1 = base_shape[:2]
    h2, w2 = add_shape[:2]

    base_corners = np.float32([[0,0], [0,h1], [w1,h1], [w1,0]]).reshape(-1,1,2)
    add_corners  = np.float32([[0,0], [0,h2], [w2,h2], [w2,0]]).reshape(-1,1,2)

    warped_add = cv2.transform(add_corners, M)  # affine (2x3)
    all_corners = np.concatenate([base_corners, warped_add], axis=0)

    [xmin, ymin] = np.floor(all_corners.min(axis=0).ravel()).astype(np.int32)
    [xmax, ymax] = np.ceil(all_corners.max(axis=0).ravel()).astype(np.int32)

    tx = -xmin if xmin < 0 else 0
    ty = -ymin if ymin < 0 else 0
    width  = xmax - xmin
    height = ymax - ymin
    return (width, height), (tx, ty)

def feather_blend(canvasA, maskA, canvasB, maskB):
    """
    Feather-blend two same-size images using distance transform weights.
    canvasA/B: uint8 HxWx3
    maskA/B:   uint8 HxW (0/255)
    """
    # Where either contributes
    union = (maskA > 0) | (maskB > 0)
    if not np.any(union):
        return canvasA

    # Distance transforms on inverse masks give higher weights far from borders
    # OpenCV expects 8-bit single channel where >0 is foreground
    invA = (maskA == 0).astype(np.uint8) * 255
    invB = (maskB == 0).astype(np.uint8) * 255

    # Distances
    dA = cv2.distanceTransform(invA, cv2.DIST_L2, 5).astype(np.float32)
    dB = cv2.distanceTransform(invB, cv2.DIST_L2, 5).astype(np.float32)

    # Avoid divide-by-zero
    eps = 1e-5
    wA = dA / (dA + dB + eps)
    wB = 1.0 - wA

    # Where a mask is zero, force weight 0; where only one is present, force 1 for that side
    wA[maskA == 0] = 0.0
    wB[maskB == 0] = 0.0
    onlyA = (maskA > 0) & (maskB == 0)
    onlyB = (maskB > 0) & (maskA == 0)
    wA[onlyA] = 1.0
    wB[onlyA] = 0.0
    wA[onlyB] = 0.0
    wB[onlyB] = 1.0

    # Blend
    A = canvasA.astype(np.float32)
    B = canvasB.astype(np.float32)
    # expand weights to 3 channels
    wA3 = np.stack([wA, wA, wA], axis=-1)
    wB3 = np.stack([wB, wB, wB], axis=-1)
    out = (A * wA3 + B * wB3)

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def compose_on_canvas(base_img, add_img, M):
    """
    Warp add_img by affine M into a canvas big enough for both images,
    then feather-blend the overlap. Returns the new canvas image.
    """
    # 1) compute canvas size + translation so nothing gets cropped
    (W, H), (tx, ty) = warp_size_for_affine(base_img.shape, add_img.shape, M)

    # Promote both to 3x3 homogeneous matrices
    M_h = np.vstack([M, [0,0,1]])  # (3x3)
    T_h = np.array([[1,0,tx],
                    [0,1,ty],
                    [0,0,1]], dtype=np.float32)

    M_total = T_h @ M_h   # combine translation + affine
    M_total = M_total[:2, :]  # back to 2x3

    # Warp add_img and mask
    warped_add = cv2.warpAffine(add_img, M_total, (W, H))
    mask_add = cv2.warpAffine(np.ones(add_img.shape[:2], dtype=np.uint8) * 255, M_total, (W, H))


    # 4) place base_img into the canvas
    canvas_base = np.zeros_like(warped_add)
    mask_base = np.zeros_like(mask_add)
    h1, w1 = base_img.shape[:2]
    canvas_base[ty:ty+h1, tx:tx+w1] = base_img
    mask_base[ty:ty+h1, tx:tx+w1] = 255

    # 5) feather blend the two canvases
    blended = feather_blend(canvas_base, mask_base, warped_add, mask_add)
    return blended

# =======================
# Main stitching pipeline
# =======================
def stitch_frames(frames):
    orb = cv2.ORB_create(ORB_N_FEATURES)
    stitched = frames[0].copy()

    for i in range(1, len(frames)):
        img_prev = stitched
        img_curr = frames[i]

        # 1) feature detect + match
        src_pts, dst_pts, n_good = detect_and_match(img_prev, img_curr, orb, RATIO_TEST_THRES)
        if src_pts is None or dst_pts is None:
            print(f"[WARN] Frame {i}: insufficient good matches ({n_good}). Skipping.")
            continue

        # 2) affine (no perspective)
        M, inliers = estimate_affine(src_pts, dst_pts)
        if M is None or inliers < MIN_INLIERS:
            print(f"[WARN] Frame {i}: affine failed or few inliers ({inliers}). Skipping.")
            continue

        print(f"[INFO] Frame {i}: good_matches={n_good}, inliers={inliers}")

        # 3) compose on canvas with feather blend
        stitched = compose_on_canvas(img_prev, img_curr, M)

    return stitched

# =======================
# Run
# =======================
if __name__ == "__main__":
    frames = extract_every_nth_frame(VIDEO_PATH, FRAMES_DIR, STEP)
    if len(frames) < 2:
        print("[ERROR] Need at least 2 frames to stitch.")
    else:
        result = stitch_frames(frames)
        cv2.imwrite(RESULT_IMAGE, result)
        print(f"[DONE] Stitched image saved as '{RESULT_IMAGE}'")
