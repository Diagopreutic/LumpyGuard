#!/usr/bin/env python3
"""Convert a trained ConvNeXt checkpoint (.pth) to ONNX (runs on CPU).

Install (once):   pip install torch timm onnx onnxruntime pillow numpy
Usage:            1) edit the paths in the "EDIT THESE" block below
                  2) run:  python export_onnx.py

The exported model takes a float32 tensor [batch, 3, 640, 640] (already resized and normalized with
ImageNet mean/std; test_onnx.py does this for you) and returns logits [batch, num_classes].
Class names, input size and normalization values are stored inside the .onnx file as metadata.
The batch size is dynamic; the image size is fixed at export time.
"""
import json, os, sys
import numpy as np
import torch
import timm

# ============================ EDIT THESE ============================
WEIGHTS_PATH = r"F:\Diagopreutic Pvt Ltd\Cattle_lumpy_skin\convnext_cattle_best.pth"     # trained checkpoint to convert
ONNX_PATH    = r"F:\Diagopreutic Pvt Ltd\Cattle_lumpy_skin\convnext_cattle_best.onnx"    # where to save the ONNX file (None = next to the .pth)
IMG_SIZE     = None    # None = value stored in the checkpoint (640)
CLASS_NAMES  = None    # None = names stored in the checkpoint (Class_1, Class_2)
MODEL_NAME   = None    # None = timm model name stored in the checkpoint
OPSET        = 17      # ONNX opset version
VERIFY       = True    # compare PyTorch and ONNX outputs after export (needs onnxruntime)
# ====================================================================


def load_checkpoint(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except Exception:
        # Falls back to full unpickling. Only do this for checkpoints you created yourself.
        return torch.load(path, map_location='cpu', weights_only=False)


def main():
    out_path = ONNX_PATH or os.path.splitext(WEIGHTS_PATH)[0] + '.onnx'

    ckpt = load_checkpoint(WEIGHTS_PATH)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    class_names = CLASS_NAMES or ckpt.get('class_names') or ['Class_1', 'Class_2']
    model_name = MODEL_NAME or ckpt.get('model_name') or 'convnext_base.fb_in22k_ft_in1k'
    size = IMG_SIZE or ckpt.get('img_size') or 640

    model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names))
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, size, size)
    kwargs = dict(input_names=['input'], output_names=['logits'],
                  dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}},
                  opset_version=OPSET, do_constant_folding=True)
    print(f"Exporting {model_name} ({size}x{size}, classes {class_names}) to {out_path} ... (can take a minute on CPU)")
    with torch.no_grad():
        try:
            torch.onnx.export(model, dummy, out_path, dynamo=False, **kwargs)   # classic exporter (torch >= 2.5)
        except TypeError:
            torch.onnx.export(model, dummy, out_path, **kwargs)                 # older torch versions

    # Store class names etc. inside the ONNX file so test_onnx.py needs no extra arguments
    try:
        import onnx
        m = onnx.load(out_path)
        for k, v in {'class_names': json.dumps(class_names), 'img_size': str(size),
                     'mean': '0.485,0.456,0.406', 'std': '0.229,0.224,0.225',
                     'model_name': model_name}.items():
            entry = m.metadata_props.add(); entry.key, entry.value = k, v
        onnx.checker.check_model(m)
        onnx.save(m, out_path)
    except ImportError:
        print("[note] 'onnx' package not installed: metadata not embedded (pip install onnx). "
              "test_onnx.py will fall back to Class_1/Class_2 and 640.")

    print(f"Saved: {out_path} ({os.path.getsize(out_path) / 1024**2:.0f} MB)")

    if VERIFY:
        try:
            import onnxruntime as ort
        except ImportError:
            print("[note] onnxruntime not installed: skipping verification (pip install onnxruntime).")
            return
        x = np.random.RandomState(0).randn(1, 3, size, size).astype(np.float32)
        with torch.no_grad():
            ref = model(torch.from_numpy(x)).numpy()
        sess = ort.InferenceSession(out_path, providers=['CPUExecutionProvider'])
        got = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        diff = float(np.abs(ref - got).max())
        print(f"Verification: max |PyTorch - ONNX| logit difference = {diff:.2e} "
              f"({'OK' if diff < 1e-3 else 'LARGER THAN EXPECTED, check the export'})")


if __name__ == '__main__':
    main()
