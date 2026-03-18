import os
import glob
import cv2


DATASET_DIR = "dataset"
BOXED_DIR = os.path.join(DATASET_DIR, "boxed")
FLAT_DIR = os.path.join(DATASET_DIR, "flat")
LBL_DIR = os.path.join(DATASET_DIR, "labels")

WINDOW = "Rebox Annotator (boxed-only)"
BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 2


def img_id_from_path(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


def find_flat_image(img_id: str):
    """Try common extensions for the corresponding flat/original image."""
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        cand = os.path.join(FLAT_DIR, img_id + ext)
        if os.path.exists(cand):
            return cand
    return None


def label_path(img_id: str) -> str:
    return os.path.join(LBL_DIR, img_id + ".txt")


def boxed_path(img_id: str) -> str:
    return os.path.join(BOXED_DIR, img_id + ".jpg")  


def load_box(lbl_path: str):
    """Return (x1,y1,x2,y2) or None if missing/empty/invalid."""
    if not os.path.exists(lbl_path):
        return None
    try:
        s = open(lbl_path, "r").read().strip()
        if not s:
            return None
        parts = s.split()
        if len(parts) < 4:
            return None
        x1, y1, x2, y2 = map(int, parts[:4])
        return (x1, y1, x2, y2)
    except Exception:
        return None


def save_box(lbl_path: str, box):
    """If box is None, write empty file; else write 'x1 y1 x2 y2'."""
    os.makedirs(os.path.dirname(lbl_path), exist_ok=True)
    with open(lbl_path, "w") as f:
        if box is None:
            f.write("")
        else:
            x1, y1, x2, y2 = box
            f.write(f"{x1} {y1} {x2} {y2}\n")


def clamp_and_normalize(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(0, min(int(y2), h - 1))
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def draw(img, box, temp_box=None, text=None):
    out = img.copy()
    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
    if temp_box is not None:
        x1, y1, x2, y2 = temp_box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 1)
    if text:
        cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def write_boxed_preview(img_id: str, flat_img, box):
    os.makedirs(BOXED_DIR, exist_ok=True)
    preview = flat_img.copy()
    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(preview, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
    cv2.imwrite(boxed_path(img_id), preview)


drawing = False
start_pt = None
temp_box = None


def main():
    global drawing, start_pt, temp_box


    boxed_files = sorted(glob.glob(os.path.join(BOXED_DIR, "*.jpg")) +
                         glob.glob(os.path.join(BOXED_DIR, "*.png")) +
                         glob.glob(os.path.join(BOXED_DIR, "*.jpeg")))
    if not boxed_files:
        raise SystemExit(f"No images found in {BOXED_DIR}")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    idx = 0
    saves = 0

    while 0 <= idx < len(boxed_files):
        img_id = img_id_from_path(boxed_files[idx])
        flat_path = find_flat_image(img_id)

        if flat_path is None:
            print(f"Warning: no matching flat image for id={img_id}. Skipping.")
            idx += 1
            continue

        img = cv2.imread(flat_path)
        if img is None:
            print(f"Warning: couldn't read {flat_path}. Skipping.")
            idx += 1
            continue

        h, w = img.shape[:2]
        lblp = label_path(img_id)
        cur_box = load_box(lblp)
        new_box = cur_box
        temp_box = None
        drawing = False
        start_pt = None

        def on_mouse(event, x, y, flags, param):
            nonlocal new_box
            global drawing, start_pt, temp_box

            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                start_pt = (x, y)
                temp_box = (x, y, x, y)

            elif event == cv2.EVENT_MOUSEMOVE and drawing and start_pt is not None:
                x1, y1 = start_pt
                temp_box = (x1, y1, x, y)

            elif event == cv2.EVENT_LBUTTONUP and drawing and start_pt is not None:
                drawing = False
                x1, y1 = start_pt
                temp_box = (x1, y1, x, y)
                start_pt = None

                candidate = clamp_and_normalize(temp_box, w, h)
                if candidate is not None:
                    new_box = candidate

        cv2.setMouseCallback(WINDOW, on_mouse)

        while True:
            overlay = draw(
                img,
                new_box,
                temp_box=temp_box if drawing else None,
                text=f"[{idx+1}/{len(boxed_files)}] drag=redraw | s=save | x=clear | n/p next/prev | q=quit"
            )
            cv2.imshow(WINDOW, overlay)
            key = cv2.waitKey(15) & 0xFF

            if key in (ord('q'), 27):
                cv2.destroyAllWindows()
                print(f"Saved: {saves}")
                return

            elif key == ord('n'):
                idx += 1
                break

            elif key == ord('p'):
                idx -= 1
                break

            elif key == ord('x'):
                new_box = None
                temp_box = None

            elif key == ord('s'):
                save_box(lblp, new_box)
                write_boxed_preview(img_id, img, new_box)
                saves += 1
                idx += 1
                break

    cv2.destroyAllWindows()
    print(f"Saved: {saves}")


if __name__ == "__main__":
    main()
