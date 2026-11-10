"""
livestock_cam.py -- REAL-TIME cow + goat disease detection on the PYNQ-Z2 with a USB camera.

Every camera frame goes through TWO models and the results are fused:
  GreenCNN (int8)   -> runs in the FPGA fabric  (your HLS bitstream green_cnn.bit)
  MobileNetV2       -> runs on the ARM CPU      (mobilenetv2.onnx with OpenCV DNN)
  final = w * p_cnn + (1 - w) * p_mnv2          (w from fusion.json, chosen on validation data)

Every detection is written to logs/detections.csv on the SD card (offline-first).

On the PYNQ board (Jupyter cell, in this folder):
    %run livestock_cam.py                       # live camera, picture shown in the notebook
    %run livestock_cam.py --frames 100          # stop after 100 frames
    %run livestock_cam.py --image cow.jpg       # one photo, no camera needed
In a board terminal (no picture, text only):
    sudo python3 livestock_cam.py --display none
On your Windows PC, without the board (FPGA simulated bit-exactly in numpy):
    python livestock_cam.py --sim --display window
Stop the live loop with the Jupyter stop button (Kernel -> Interrupt) or Ctrl+C.
"""
import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


def path(name):
    return os.path.join(HERE, name)


def load_json(name):
    with open(path(name)) as f:
        return json.load(f)


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


# ----------------------------------------------------------------------------- preprocessing
# Must match training exactly: square centre-crop -> resize (PIL bilinear) -> centre-crop.
def square_rgb(frame_bgr):
    h, w = frame_bgr.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return Image.fromarray(cv2.cvtColor(frame_bgr[y0:y0 + s, x0:x0 + s], cv2.COLOR_BGR2RGB))


def resize_crop(sq_img, cache_size, out_size):
    a = np.asarray(sq_img.resize((cache_size, cache_size), Image.BILINEAR), dtype=np.uint8)
    o = (cache_size - out_size) // 2
    return a[o:o + out_size, o:o + out_size]                 # H x W x 3 uint8 RGB


# ----------------------------------------------------------------------------- GreenCNN on FPGA
class FpgaCNN:
    """Runs green_cnn.bit through PYNQ: image -> DDR buffer -> HLS IP -> logits."""
    name = "FPGA"

    def __init__(self, bitfile, n_classes, img_size):
        from pynq import Overlay, allocate
        self.ol = Overlay(bitfile)
        ip_name = next(k for k in self.ol.ip_dict if k.startswith("green_cnn"))
        self.ip = getattr(self.ol, ip_name)
        self.nc = n_classes
        self.inbuf = allocate(shape=(3 * img_size * img_size,), dtype=np.uint8)
        self.outbuf = allocate(shape=(n_classes + 1,), dtype=np.int32)
        regs = self.ol.ip_dict[ip_name]["registers"]
        self._set_addr(regs, "img", self.inbuf.physical_address)
        self._set_addr(regs, "out", self.outbuf.physical_address)
        print(f"FPGA: overlay {os.path.basename(bitfile)} loaded, IP '{ip_name}'")

    def _set_addr(self, regs, prefix, addr):
        """HLS gives each pointer an address register (img_1/img_2 or out_r_1/out_r_2)."""
        names = sorted((k for k in regs if k.lower().startswith(prefix)),
                       key=lambda k: regs[k]["address_offset"])
        if not names:
            raise RuntimeError(f"no '{prefix}' address register in the IP -- wrong .hwh?")
        self.ip.write(regs[names[0]]["address_offset"], addr & 0xFFFFFFFF)
        if len(names) > 1:
            self.ip.write(regs[names[1]]["address_offset"], (addr >> 32) & 0xFFFFFFFF)

    def run(self, chw_u8):
        self.inbuf[:] = chw_u8.ravel()
        self.inbuf.flush()
        self.ip.write(0x00, 1)                                # ap_start
        t0 = time.time()
        while not (self.ip.read(0x00) & 0x2):                 # ap_done
            if time.time() - t0 > 2:
                raise TimeoutError("green_cnn did not finish -- check clocks/HP0 in the block design")
        self.outbuf.invalidate()
        return np.array(self.outbuf[:self.nc], dtype=np.int64), int(self.outbuf[self.nc])


class SimCNN:
    """Bit-exact numpy model of the FPGA (for testing on a PC without the board)."""
    name = "SIM"

    def __init__(self, npz, n_classes):
        p = np.load(npz)
        self.layers = [(p[f"W{i}"].astype(np.int64), p[f"B{i}"].astype(np.int64),
                        p[f"M{i}"].astype(np.int64)) for i in range(1, 5)]
        self.fc_w, self.fc_b = p["FC_W"].astype(np.int64), p["FC_B"].astype(np.int64)
        self.shift = int(p["SHIFT"])
        self.nc = n_classes
        print("SIM: FPGA replaced by bit-exact numpy model")

    def run(self, chw_u8):
        a = chw_u8.astype(np.int64)
        for w, b, m in self.layers:
            ap = np.pad(a, ((0, 0), (1, 1), (1, 1)))
            cols = np.lib.stride_tricks.sliding_window_view(ap, (3, 3), axis=(1, 2))
            acc = np.maximum(np.einsum("chwij,ocij->ohw", cols, w, optimize=True) + b[:, None, None], 0)
            a = np.clip((acc * m[:, None, None] + (1 << (self.shift - 1))) >> self.shift, 0, 255)
            c, h, wd = a.shape
            a = a.reshape(c, h // 2, 2, wd // 2, 2).max(axis=(2, 4))
        logits = self.fc_b + self.fc_w @ a.sum(axis=(1, 2))
        return logits, int(np.argmax(logits))


# ----------------------------------------------------------------------------- MobileNetV2 on ARM
class MobileNetARM:
    def __init__(self, onnx_file, meta):
        self.net = cv2.dnn.readNetFromONNX(onnx_file)
        self.size = meta["img_size"]
        self.mean = np.array(meta["mean"], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(meta["std"], dtype=np.float32).reshape(3, 1, 1)
        print(f"ARM: MobileNetV2 loaded ({self.size}x{self.size}, OpenCV {cv2.__version__})")

    def run(self, hwc_u8):
        x = (hwc_u8.transpose(2, 0, 1).astype(np.float32) / 255.0 - self.mean) / self.std
        self.net.setInput(x[None])
        return self.net.forward()[0]


# ----------------------------------------------------------------------------- the detector
class LivestockDetector:
    MNV2_CACHE = 128          # must match training config (MNV2_CACHE_SIZE)
    CNN_CACHE = 72            # must match training config (CACHE_SIZE)

    def __init__(self, sim=False, bitfile="green_cnn.bit"):
        self.cnn_meta = load_json("cnn_meta.json")
        self.mnv2_meta = load_json("mnv2_meta.json")
        self.fusion = load_json("fusion.json")
        self.classes = self.fusion["classes"]
        nc = len(self.classes)
        if sim:
            self.cnn = SimCNN(path("cnn_int8_params.npz"), nc)
        else:
            self.cnn = FpgaCNN(path(bitfile), nc, self.cnn_meta["img_size"])
        self.mnv2 = MobileNetARM(path("mobilenetv2.onnx"), self.mnv2_meta)
        self.w = float(self.fusion["w_cnn"])
        self.thr = float(self.fusion.get("confidence_threshold", 0.6))
        print(f"Fusion: {self.w:.2f} x CNN + {1 - self.w:.2f} x MobileNetV2, "
              f"confidence threshold {self.thr:.2f}")
        print("Classes:", ", ".join(self.classes))

    def predict(self, frame_bgr):
        sq = square_rgb(frame_bgr)
        cnn_in = resize_crop(sq, self.CNN_CACHE, self.cnn_meta["img_size"])
        mn_in = resize_crop(sq, self.MNV2_CACHE, self.mnv2_meta["img_size"])

        t0 = time.time()
        logits, cls_hw = self.cnn.run(np.ascontiguousarray(cnn_in.transpose(2, 0, 1)))
        t1 = time.time()
        mn_logits = self.mnv2.run(mn_in)
        t2 = time.time()

        p_cnn = softmax(logits * self.cnn_meta["logit_scale"])
        p_mn = softmax(mn_logits)
        p = self.w * p_cnn + (1 - self.w) * p_mn
        k = int(np.argmax(p))
        label = self.classes[k] if p[k] >= self.thr else "Uncertain"
        return {"label": label, "conf": float(p[k]), "probs": p,
                "cnn": self.classes[cls_hw], "cnn_conf": float(p_cnn[cls_hw]),
                "mnv2": self.classes[int(np.argmax(p_mn))], "mnv2_conf": float(p_mn.max()),
                "cnn_ms": (t1 - t0) * 1000, "arm_ms": (t2 - t1) * 1000}


# ----------------------------------------------------------------------------- drawing / display
def color_for(label):
    if label == "Uncertain":
        return (0, 200, 255)                       # orange
    return (0, 200, 0) if "Healthy" in label and "Unhealthy" not in label else (0, 0, 255)


def annotate(frame, r, fps):
    img = frame.copy()
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    cv2.rectangle(img, (x0, y0), (x0 + s - 1, y0 + s - 1), (255, 255, 255), 1)   # area the models see
    col = color_for(r["label"])
    cv2.rectangle(img, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.putText(img, f"{r['label']}  {r['conf'] * 100:.0f}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
    cv2.putText(img, f"FPGA-CNN: {r['cnn']} {r['cnn_conf'] * 100:.0f}% ({r['cnn_ms']:.0f} ms)   "
                     f"ARM-MobileNetV2: {r['mnv2']} {r['mnv2_conf'] * 100:.0f}% ({r['arm_ms']:.0f} ms)",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(img, f"{fps:.1f} FPS", (w - 90, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
    return img


class Display:
    def __init__(self, mode):
        self.mode = mode
        if mode == "jupyter":
            import ipywidgets as widgets
            from IPython.display import display
            self.img = widgets.Image(format="jpeg", width=640)
            self.txt = widgets.HTML()
            display(widgets.VBox([self.img, self.txt]))

    def show(self, img, r):
        if self.mode == "jupyter":
            self.img.value = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
            self.txt.value = (f"<b>{r['label']}</b> ({r['conf'] * 100:.1f}%) &nbsp; "
                              f"FPGA {r['cnn_ms']:.0f} ms, ARM {r['arm_ms']:.0f} ms")
        elif self.mode == "window":
            cv2.imshow("Livestock disease detection (q = quit)", img)
            return (cv2.waitKey(1) & 0xFF) == ord("q")
        return False


# ----------------------------------------------------------------------------- main loop
def open_camera(src):
    src = int(src) if str(src).isdigit() else src
    if isinstance(src, int):
        if sys.platform.startswith("linux"):
            backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
        elif sys.platform.startswith("win"):
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_ANY]
        cap = None
        for be in backends:                     # try each camera driver until one works
            cap = cv2.VideoCapture(src, be)
            if cap.isOpened():
                break
            cap.release()
    else:
        cap = cv2.VideoCapture(src)
    if isinstance(src, int) and cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit(f"ERROR: cannot open camera/video '{src}'. Run test_camera.py first.")
    return cap


def main(argv=None):
    ap = argparse.ArgumentParser(description="Real-time cow/goat disease detection")
    ap.add_argument("--sim", action="store_true", help="no FPGA: use the numpy model (PC testing)")
    ap.add_argument("--bit", default="green_cnn.bit")
    ap.add_argument("--camera", default="0", help="camera number or a video file")
    ap.add_argument("--image", help="classify one image file instead of the camera")
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = run until stopped)")
    ap.add_argument("--display", default=None, choices=["jupyter", "window", "none"])
    ap.add_argument("--log", default=path(os.path.join("logs", "detections.csv")))
    args = ap.parse_args(argv)

    if args.display is None:
        in_jupyter = "ipykernel" in sys.modules
        args.display = "jupyter" if in_jupyter else ("window" if args.sim else "none")

    det = LivestockDetector(sim=args.sim, bitfile=args.bit)

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"ERROR: cannot read {args.image}")
        r = det.predict(frame)
        print(f"{os.path.basename(args.image)}: {r['label']} ({r['conf'] * 100:.1f}%)  "
              f"| FPGA-CNN {r['cnn']} {r['cnn_ms']:.1f} ms | MobileNetV2 {r['mnv2']} {r['arm_ms']:.0f} ms")
        for c, p in zip(det.classes, r["probs"]):
            print(f"   {c:15s} {p * 100:5.1f}%")
        out = os.path.splitext(args.image)[0] + "_result.jpg"
        cv2.imwrite(out, annotate(frame, r, 0))
        print(f"annotated image -> {out}")
        return r

    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    new = not os.path.isfile(args.log)
    logf = open(args.log, "a", newline="")
    log = csv.writer(logf)
    if new:
        log.writerow(["time", "frame", "label", "confidence", "fpga_cnn", "arm_mobilenetv2",
                      "fpga_ms", "arm_ms"])

    cap = open_camera(args.camera)
    disp = Display(args.display)
    print(f"Running... (display: {args.display}, log: {args.log})  stop: Kernel->Interrupt / Ctrl+C")
    n, t_last, fps = 0, time.time(), 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera/video ended")
                break
            r = det.predict(frame)
            n += 1
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_last, 1e-6)) if n > 1 else 0.0
            t_last = now
            log.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), n, r["label"], f"{r['conf']:.3f}",
                          r["cnn"], r["mnv2"], f"{r['cnn_ms']:.1f}", f"{r['arm_ms']:.1f}"])
            if disp.show(annotate(frame, r, fps), r):
                break
            if args.display == "none" or n % 10 == 0:
                print(f"frame {n:5d}  {r['label']:15s} {r['conf'] * 100:5.1f}%  "
                      f"FPGA {r['cnn_ms']:5.1f} ms  ARM {r['arm_ms']:5.0f} ms  {fps:4.1f} FPS")
            if args.frames and n >= args.frames:
                break
    except KeyboardInterrupt:
        print("stopped")
    finally:
        cap.release()
        logf.close()
        if args.display == "window":
            cv2.destroyAllWindows()
    print(f"{n} frames processed, detections saved in {args.log}")


if __name__ == "__main__":
    main()
