import os
from pathlib import Path
import cv2
import numpy as np

VIDEO_PATH   = r"C:\Users\facti\OneDrive\Desktop\groundschool\cir.mp4"
FRAMES_DIR   = "frames_out"
RESULT_IMAGE = "stitched_result.png"
STEP         = 5

ORB_N_FEATURES   = 8000
RATIO_TEST_THRES = 0.75 #FILTERS OUT BAD MATCHES
RANSAC_REPROJ    = 3.0
MIN_GOOD_MATCHES = 40
MIN_INLIERS      = 25

def extract_every_nth_frame(video_path: str, out_dir: str, step: int = 5) -> list:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

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
    print(f"output: {saved} '{out_dir}'.")
    return frames

def detect_and_match(imgA, imgB, orb, ratio=0.75):
    #convert to gary 
    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
    
    #detect keypoints
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
    M, inliers = cv2.estimateAffinePartial2D(
        dst_pts, src_pts, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_REPROJ, maxIters=5000, confidence=0.995
    )
    inlier_count = int(inliers.sum()) if inliers is not None else 0
    return M, inlier_count

def warp_size_for_affine(base_shape, add_shape, M):
    h1, w1 = base_shape[:2]
    h2, w2 = add_shape[:2]
    base_corners = np.float32([[0,0], [0,h1], [w1,h1], [w1,0]]).reshape(-1,1,2)
    add_corners  = np.float32([[0,0], [0,h2], [w2,h2], [w2,0]]).reshape(-1,1,2)
    warped_add = cv2.transform(add_corners, M)
    all_corners = np.concatenate([base_corners, warped_add], axis=0)
    [xmin, ymin] = np.floor(all_corners.min(axis=0).ravel()).astype(np.int32)
    [xmax, ymax] = np.ceil(all_corners.max(axis=0).ravel()).astype(np.int32)
    tx = -xmin if xmin < 0 else 0
    ty = -ymin if ymin < 0 else 0
    width  = xmax - xmin
    height = ymax - ymin
    return (width, height), (tx, ty)

def feather_blend(canvasA, maskA, canvasB, maskB):
    union = (maskA > 0) | (maskB > 0)
    if not np.any(union):
        return canvasA
    invA = (maskA == 0).astype(np.uint8) * 255
    invB = (maskB == 0).astype(np.uint8) * 255
    dA = cv2.distanceTransform(invA, cv2.DIST_L2, 5).astype(np.float32)
    dB = cv2.distanceTransform(invB, cv2.DIST_L2, 5).astype(np.float32)
    eps = 1e-5
    wA = dA / (dA + dB + eps)
    wB = 1.0 - wA
    wA[maskA == 0] = 0.0
    wB[maskB == 0] = 0.0
    onlyA = (maskA > 0) & (maskB == 0)
    onlyB = (maskB > 0) & (maskA == 0)
    wA[onlyA] = 1.0
    wB[onlyA] = 0.0
    wA[onlyB] = 0.0
    wB[onlyB] = 1.0
    A = canvasA.astype(np.float32)
    B = canvasB.astype(np.float32)
    wA3 = np.stack([wA, wA, wA], axis=-1)
    wB3 = np.stack([wB, wB, wB], axis=-1)
    out = (A * wA3 + B * wB3)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def compose_on_canvas(base_img, add_img, M):
    (W, H), (tx, ty) = warp_size_for_affine(base_img.shape, add_img.shape, M)
    M_h = np.vstack([M, [0,0,1]])
    T_h = np.array([[1,0,tx],
                    [0,1,ty],
                    [0,0,1]], dtype=np.float32)
    M_total = T_h @ M_h
    M_total = M_total[:2, :]
    warped_add = cv2.warpAffine(add_img, M_total, (W, H))
    mask_add = cv2.warpAffine(np.ones(add_img.shape[:2], dtype=np.uint8) * 255, M_total, (W, H))
    canvas_base = np.zeros_like(warped_add)
    mask_base = np.zeros_like(mask_add)
    h1, w1 = base_img.shape[:2]
    canvas_base[ty:ty+h1, tx:tx+w1] = base_img
    mask_base[ty:ty+h1, tx:tx+w1] = 255
    blended = feather_blend(canvas_base, mask_base, warped_add, mask_add)
    return blended

def stitch_frames(frames):
    orb = cv2.ORB_create(ORB_N_FEATURES)
    stitched = frames[0].copy()
    for i in range(1, len(frames)):
        img_prev = stitched
        img_curr = frames[i]
        src_pts, dst_pts, n_good = detect_and_match(img_prev, img_curr, orb, RATIO_TEST_THRES)
        if src_pts is None or dst_pts is None:
            print(f"insufficient good matches ({n_good})continue loop")
            continue
        M, inliers = estimate_affine(src_pts, dst_pts)
        if M is None or inliers < MIN_INLIERS:
            print(f"affine failed({inliers}). continue loop.")
            continue
        print(f"good_matches={n_good}, inliers={inliers}")
        stitched = compose_on_canvas(img_prev, img_curr, M)
    return stitched

if __name__ == "__main__":
    frames = extract_every_nth_frame(VIDEO_PATH, FRAMES_DIR, STEP)
    if len(frames) < 2:
        print("[ERROR] Need at least 2 frames to stitch.")
    else:
        result = stitch_frames(frames)
        cv2.imwrite(RESULT_IMAGE, result)
        print(f"[DONE] Stitched image saved as '{RESULT_IMAGE}'")
