import os
import time
import cv2
import torch
import torchvision
import numpy as np
from torchvision.transforms import functional as F

def build_model(num_classes: int = 2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    return model

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def ensure_dirs(dataset_root):
    flat_dir = os.path.join(dataset_root, "flat")
    labels_dir = os.path.join(dataset_root, "labels")
    boxed_dir = os.path.join(dataset_root, "boxed")
    os.makedirs(flat_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(boxed_dir, exist_ok=True)
    return flat_dir, labels_dir, boxed_dir

def draw_overlay(raw_bgr, box_xyxy, label_text=None):
    out = raw_bgr.copy()
    if box_xyxy is not None:
        x1, y1, x2, y2 = box_xyxy
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if label_text:
            cv2.putText(out, label_text, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        if label_text:
            cv2.putText(out, label_text, (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return out

def save_sample(flat_dir, labels_dir, boxed_dir, raw_bgr, box_xyxy, is_negative=False, tag="auto"):
    sample_id = str(int(time.time() * 1000))

    flat_path = os.path.join(flat_dir, f"{sample_id}.jpg")
    lbl_path = os.path.join(labels_dir, f"{sample_id}.txt")
    boxed_path = os.path.join(boxed_dir, f"{sample_id}.jpg")

    cv2.imwrite(flat_path, raw_bgr)

    with open(lbl_path, "w", encoding="utf-8") as f:
        if (not is_negative) and (box_xyxy is not None):
            x1, y1, x2, y2 = box_xyxy
            f.write(f"{x1} {y1} {x2} {y2}\n")

    if is_negative:
        overlay = draw_overlay(raw_bgr, None, label_text=f"{tag} negative")
    else:
        overlay = draw_overlay(raw_bgr, box_xyxy, label_text=tag)

    cv2.imwrite(boxed_path, overlay)

    return sample_id

def main():
 
    dataset_root = "dataset"         
    weights_path = "models/frcnn_best.pth"
    source = 0

 
    score_thresh = 0.20

    
    autosave_thresh = 0.85               
    autosave_cooldown_sec = 0.35         

    draw_only_best = True                
    smoothing = 0.35                     


    flat_dir, labels_dir, boxed_dir = ensure_dirs(dataset_root)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    model = build_model(num_classes=2)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Could not open video source")

    prev_box = None
    autosave_enabled = False
    last_auto_t = 0.0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        raw = frame_bgr.copy()
        H, W = raw.shape[:2]

        frame_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        img = F.to_tensor(frame_rgb).to(device)

        with torch.no_grad():
            out = model([img])[0]

        boxes = out["boxes"].detach().cpu().numpy()
        scores = out["scores"].detach().cpu().numpy()
        labels = out["labels"].detach().cpu().numpy()


        best_i = None
        best_s = 0.0

        for i, (s, lab) in enumerate(zip(scores, labels)):
            if int(lab) != 1:
                continue
            s = float(s)
            if s > best_s:
                best_s = s
                best_i = i

        box_xyxy = None
        if best_i is not None and best_s >= score_thresh:
            x1, y1, x2, y2 = boxes[best_i]
            x1 = clamp(int(x1), 0, W - 1)
            y1 = clamp(int(y1), 0, H - 1)
            x2 = clamp(int(x2), 0, W - 1)
            y2 = clamp(int(y2), 0, H - 1)

            if smoothing > 0.0 and prev_box is not None:
                px1, py1, px2, py2 = prev_box
                x1 = int((1 - smoothing) * x1 + smoothing * px1)
                y1 = int((1 - smoothing) * y1 + smoothing * py1)
                x2 = int((1 - smoothing) * x2 + smoothing * px2)
                y2 = int((1 - smoothing) * y2 + smoothing * py2)

            prev_box = (x1, y1, x2, y2)
            box_xyxy = (x1, y1, x2, y2)
        else:
            prev_box = None

        overlay = raw.copy()
        status = f"autosave={'ON' if autosave_enabled else 'OFF'}  score={best_s:.2f}"
        cv2.putText(overlay, status, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if box_xyxy is not None:
            x1, y1, x2, y2 = box_xyxy
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(overlay, f"Object {best_s:.2f}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        now = time.time()
        if autosave_enabled and box_xyxy is not None and best_s >= autosave_thresh:
            if now - last_auto_t >= autosave_cooldown_sec:
                sid = save_sample(flat_dir, labels_dir, boxed_dir, raw, box_xyxy, is_negative=False,
                                  tag=f"auto {best_s:.2f}")
                print("Auto-saved:", sid, "score:", f"{best_s:.2f}")
                last_auto_t = now

        cv2.imshow("Object detector (autosave)", overlay)

        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break

        if k == ord("a"):
            autosave_enabled = not autosave_enabled
            print("autosave_enabled =", autosave_enabled)

 
        if k == ord("s") and box_xyxy is not None:
            sid = save_sample(flat_dir, labels_dir, boxed_dir, raw, box_xyxy, is_negative=False,
                              tag=f"manual {best_s:.2f}")
            print("Manual saved:", sid, "score:", f"{best_s:.2f}")


        if k == ord("n"):
            sid = save_sample(flat_dir, labels_dir, boxed_dir, raw, None, is_negative=True, tag="manual")
            print("Saved NEGATIVE:", sid)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
