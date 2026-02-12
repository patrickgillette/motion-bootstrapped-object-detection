import os
import time
import cv2
import numpy as np

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def ensure_dirs(root):
    flat_dir = os.path.join(root, "flat")
    labels_dir = os.path.join(root, "labels")
    boxed_dir = os.path.join(root, "boxed")
    os.makedirs(flat_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(boxed_dir, exist_ok=True)
    return flat_dir, labels_dir, boxed_dir

def draw_box_overlay(frame_bgr, box_xyxy, text=None):
    out = frame_bgr.copy()
    if box_xyxy is not None:
        x1, y1, x2, y2 = box_xyxy
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if text:
            cv2.putText(out, text, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out

def save_sample(flat_dir, labels_dir, boxed_dir, raw_bgr, box_xyxy, is_negative=False, tag=None):
    sample_id = str(int(time.time() * 1000))

    flat_path = os.path.join(flat_dir, f"{sample_id}.jpg")
    lbl_path = os.path.join(labels_dir, f"{sample_id}.txt")
    boxed_path = os.path.join(boxed_dir, f"{sample_id}.jpg")

    cv2.imwrite(flat_path, raw_bgr)

    with open(lbl_path, "w", encoding="utf-8") as f:
        if (not is_negative) and (box_xyxy is not None):
            x1, y1, x2, y2 = box_xyxy
            f.write(f"{x1} {y1} {x2} {y2}\n")

    overlay = draw_box_overlay(raw_bgr, box_xyxy if not is_negative else None, text=tag)
    if is_negative:
        cv2.putText(overlay, "negative", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(boxed_path, overlay)

    return sample_id

def open_best_camera(preferred_index=None, max_index=5):
    """
    If preferred_index is set (e.g. 1), we try that first.
    Otherwise we scan 0..max_index and pick the first camera that returns frames.
    """
    indices = []
    if preferred_index is not None:
        indices.append(preferred_index)
    indices.extend([i for i in range(0, max_index + 1) if i != preferred_index])

    for idx in indices:
        cap = cv2.VideoCapture(idx)  # optionally add cv2.CAP_DSHOW on Windows
        if not cap.isOpened():
            cap.release()
            continue

        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            # Basic info to help confirm it's the USB cam
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"Using camera index {idx} ({w}x{h}, fps={fps})")
            return cap, idx

        cap.release()

    raise RuntimeError("Could not open any camera (tried indices 0..{max_index})")

def main():
    # ---------- Config ----------
    dataset_root = "dataset"

    # If you *know* your USB cam index, set it here (common: 1)
    preferred_cam_index = 1  # change to None to auto-scan without preference

    # Motion detection (classic)
    history = 500
    var_threshold = 32
    detect_shadows = True

    min_area = 900
    blur_ksize = 5
    morph_ksize = 5
    open_iters = 1
    dilate_iters = 2

    pad = 8
    save_cooldown_sec = 0.25

    # Ignore black (soft)
    black_v_max = 35
    black_s_max = 90
    ignore_black_strength = 0.60  # 0..1
    # ----------------------------

    flat_dir, labels_dir, boxed_dir = ensure_dirs(dataset_root)

    cap, cam_index = open_best_camera(preferred_index=preferred_cam_index, max_index=5)

    backsub = cv2.createBackgroundSubtractorMOG2(
        history=history, varThreshold=var_threshold, detectShadows=detect_shadows
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))

    last_save_t = 0.0
    last_sample_id = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        raw = frame.copy()
        H, W = frame.shape[:2]

        blurred = cv2.GaussianBlur(frame, (blur_ksize, blur_ksize), 0)
        fgmask = backsub.apply(blurred)
        _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        black_mask = cv2.inRange(hsv, (0, 0, 0), (179, black_s_max, black_v_max))

        if ignore_black_strength > 0.0:
            black_thin = cv2.erode(black_mask, kernel, iterations=1)
            m = (black_thin > 0).astype(np.float32) * ignore_black_strength
            fg_f = fgmask.astype(np.float32)
            fg_f *= (1.0 - m)
            fgmask = fg_f.astype(np.uint8)
            _, fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)

        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=open_iters)
        fgmask = cv2.dilate(fgmask, kernel, iterations=dilate_iters)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_cnt = None
        best_area = 0.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > best_area:
                best_area = area
                best_cnt = cnt

        box = None
        overlay_live = raw.copy()

        if best_cnt is not None and best_area >= min_area:
            x, y, w, h = cv2.boundingRect(best_cnt)
            x1 = clamp(x - pad, 0, W - 1)
            y1 = clamp(y - pad, 0, H - 1)
            x2 = clamp(x + w + pad, 0, W - 1)
            y2 = clamp(y + h + pad, 0, H - 1)
            box = (x1, y1, x2, y2)

            overlay_live = draw_box_overlay(raw, box, text=f"area={int(best_area)}")
            if last_sample_id:
                cv2.putText(overlay_live, f"last saved: {last_sample_id}", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            now = time.time()
            if now - last_save_t >= save_cooldown_sec:
                last_sample_id = save_sample(flat_dir, labels_dir, boxed_dir, raw, box,
                                             is_negative=False, tag="auto")
                last_save_t = now

        cv2.putText(overlay_live, f"cam index: {cam_index}", (10, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Live (overlay)", overlay_live)
        cv2.imshow("Motion mask", fgmask)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

        if key == ord("n"):
            last_sample_id = save_sample(flat_dir, labels_dir, boxed_dir, raw, None,
                                         is_negative=True, tag=None)
            print("Saved NEGATIVE:", last_sample_id)

        if key == ord("s") and box is not None:
            last_sample_id = save_sample(flat_dir, labels_dir, boxed_dir, raw, box,
                                         is_negative=False, tag="manual")
            print("Saved POSITIVE:", last_sample_id)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()