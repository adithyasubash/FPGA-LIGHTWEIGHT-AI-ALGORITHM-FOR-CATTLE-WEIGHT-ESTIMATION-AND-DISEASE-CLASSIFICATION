"""
test_fpga_cnn.py -- run on the PYNQ-Z2 BEFORE the camera app.
Loads green_cnn.bit, sends the 16 test images exported by step6 through the FPGA and checks
every logit is BIT-EXACT with the Python integer model. Also measures FPGA speed.

Jupyter cell (in this folder):   %run test_fpga_cnn.py
"""
import os
import time

import numpy as np

from livestock_cam import FpgaCNN, load_json, path

meta = load_json("cnn_meta.json")
classes = meta["classes"]
tv = np.load(path("cnn_test_vectors.npz"))
cnn = FpgaCNN(path("green_cnn.bit"), len(classes), meta["img_size"])
print("IP blocks:", list(cnn.ol.ip_dict.keys()))

fails, correct, times = 0, 0, []
for i, (img, exp, lab) in enumerate(zip(tv["img"], tv["logits"], tv["label"])):
    t0 = time.time()
    logits, cls = cnn.run(img)
    times.append((time.time() - t0) * 1000)
    ok = np.array_equal(logits, exp) and cls == int(np.argmax(exp))
    fails += not ok
    correct += cls == lab
    print(f"image {i:2d}  FPGA: {classes[cls]:15s} true: {classes[lab]:15s} "
          f"{'BIT-EXACT' if ok else 'MISMATCH  got ' + str(logits) + ' expected ' + str(exp)}")

n = len(tv["label"])
print(f"\n{n - fails}/{n} bit-exact, {correct}/{n} correct class")
print(f"FPGA inference time: {np.median(times):.1f} ms per image (median, incl. Python overhead)")
print("*** ALL TESTS PASSED ***" if fails == 0 else "*** TEST FAILED ***")
