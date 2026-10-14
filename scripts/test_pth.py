#!/usr/bin/env python3
"""Test a trained ConvNeXt checkpoint (.pth) on CPU.

Install (once):   pip install torch timm pillow numpy
Usage:            1) edit the paths in the "EDIT THESE" block below
                  2) run:  python test_pth.py

INPUT_PATHS accepts image files and/or folders (searched recursively). If images sit in folders
named like the classes (e.g. val/Class_1/*.jpg, val/Class_2/*.jpg), accuracy and a confusion
matrix are reported too. Every image is resized to 640x640 in memory before inference; the files
on disk are never changed.
"""
import csv, os, sys, time
import numpy as np
from PIL import Image
import torch
import timm

# ============================ EDIT THESE ============================
WEIGHTS_PATH = r"F:\Diagopreutic Pvt Ltd\Cattle_lumpy_skin\convnext_cattle_best.pth"       # trained checkpoint

#INPUT_PATHS = [                                  # image files and/or folders
#    r"Cow_lumpy_skin/val",
    # r"C:\Users\you\Pictures\cow1.jpg",]
INPUT_PATHS = [r"F:\Diagopreutic Pvt Ltd\Cattle_lumpy_skin\Cows.v1i.folder\test\Normal Skin"]

CSV_PATH    = None    # e.g. r"results_pth.csv" to save per-image results, or None
IMG_SIZE    = None    # None = value stored in the checkpoint (640)
CLASS_NAMES = None    # None = names stored in the checkpoint (Class_1, Class_2)
MODEL_NAME  = None    # None = timm model name stored in the checkpoint
NUM_THREADS = None    # None = PyTorch default, or an int such as 4
# ====================================================================

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)   # same as training
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
RESAMPLE = getattr(Image, 'Resampling', Image).BILINEAR     # same as torchvision Resize on PIL images


def collect_images(inputs):
    """Accept image files and/or folders (searched recursively)."""
    found = []
    for p in inputs:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                found += [os.path.join(root, f) for f in files if f.lower().endswith(IMG_EXTS)]
        elif os.path.isfile(p):
            found.append(p)
        else:
            print(f"[warning] not found: {p}")
    return sorted(found)


def preprocess(path, size):
    """Load an image and resize it to size x size IN MEMORY ONLY.
    The file on disk is never modified. Returns (original_size, float32 array [1,3,size,size])."""
    with Image.open(path) as im:
        orig_size = im.size                                   # (width, height) of the real file
        im = im.convert('RGB').resize((size, size), RESAMPLE) # only the in-memory pixels change
    arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return orig_size, np.ascontiguousarray(arr.transpose(2, 0, 1)[None], dtype=np.float32)


def softmax(logits):
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def truth_from_folder(path, class_names):
    """If the image sits in a folder named like a class (e.g. .../Class_2/img.jpg), use it as ground truth."""
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    return class_names.index(parent) if parent in class_names else None


def report(rows, class_names, csv_path=None):
    n = len(class_names)
    labelled = [r for r in rows if r['truth'] is not None]
    if labelled:
        correct = sum(r['pred'] == r['truth'] for r in labelled)
        print(f"\nAccuracy on {len(labelled)} images with a known class folder: {correct / len(labelled):.4f} ({correct}/{len(labelled)})")
        cm = np.zeros((n, n), dtype=int)
        for r in labelled:
            cm[r['truth'], r['pred']] += 1
        w = max(len(c) for c in class_names) + 2
        print("Confusion matrix (rows = true, columns = predicted):")
        print(" " * w + "".join(f"{c:>{w}}" for c in class_names))
        for i, c in enumerate(class_names):
            print(f"{c:<{w}}" + "".join(f"{cm[i, j]:>{w}}" for j in range(n)))
    else:
        counts = {c: sum(r['pred'] == i for r in rows) for i, c in enumerate(class_names)}
        print(f"\nPrediction counts: {counts}")
    if csv_path:
        with open(csv_path, 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(['path', 'orig_width', 'orig_height', 'predicted', 'confidence']
                        + [f'prob_{c}' for c in class_names] + ['true_class'])
            for r in rows:
                wr.writerow([r['path'], r['orig'][0], r['orig'][1], class_names[r['pred']], f"{r['probs'][r['pred']]:.6f}"]
                            + [f"{p:.6f}" for p in r['probs']]
                            + [class_names[r['truth']] if r['truth'] is not None else ''])
        print(f"Saved results to {csv_path}")


def print_row(r, class_names):
    probs = "  ".join(f"{c}={p * 100:5.1f}%" for c, p in zip(class_names, r['probs']))
    mark = ""
    if r['truth'] is not None:
        mark = "  [correct]" if r['pred'] == r['truth'] else f"  [WRONG, true: {class_names[r['truth']]}]"
    print(f"{os.path.basename(r['path'])} | original {r['orig'][0]}x{r['orig'][1]} | "
          f"-> {class_names[r['pred']]} ({r['probs'][r['pred']] * 100:.1f}%) | {probs}{mark}")


def load_checkpoint(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except Exception:
        # Falls back to full unpickling. Only do this for checkpoints you created yourself.
        return torch.load(path, map_location='cpu', weights_only=False)


def main():
    if NUM_THREADS:
        torch.set_num_threads(NUM_THREADS)

    ckpt = load_checkpoint(WEIGHTS_PATH)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    class_names = CLASS_NAMES or ckpt.get('class_names') or ['Class_1', 'Class_2']
    model_name = MODEL_NAME or ckpt.get('model_name') or 'convnext_base.fb_in22k_ft_in1k'
    size = IMG_SIZE or ckpt.get('img_size') or 640

    model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names))
    model.load_state_dict(state)
    model.eval()
    print(f"Class_1 -> Lumpy_skin;      Class_2 -> Normal_skin\n")
    print(f"Loaded {model_name} | classes: {class_names} | input size: {size}x{size} | device: cpu")

    images = collect_images(INPUT_PATHS)
    if not images:
        sys.exit("No images found. Check INPUT_PATHS at the top of this file.")
    print(f"Found {len(images)} image(s)\n")

    rows = []
    t_start = time.time()
    for path in images:
        try:
            orig, x = preprocess(path, size)
        except Exception as e:
            print(f"[skipped] {path}: {e}")
            continue
        with torch.inference_mode():
            logits = model(torch.from_numpy(x)).numpy()[0]
        probs = softmax(logits)
        row = dict(path=path, orig=orig, pred=int(probs.argmax()), probs=probs,
                   truth=truth_from_folder(path, class_names))
        rows.append(row)
        print_row(row, class_names)

    if rows:
        print(f"\nDone: {len(rows)} image(s) in {time.time() - t_start:.1f}s")
        report(rows, class_names, CSV_PATH)


if __name__ == '__main__':
    main()
