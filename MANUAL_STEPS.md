# GREEN_VLSI – Real-time Cow + Goat Disease Detection on PYNQ-Z2
### Custom CNN on the FPGA (HLS) + MobileNetV2 on the ARM CPU + USB camera — every step done by hand

```
USB camera ─► PYNQ-Z2 ARM (Python)
                 ├─► 64×64 image ─► FPGA: GreenCNN int8 (your HLS IP, ~20 ms) ─┐
                 └─► 112×112 image ─► ARM: MobileNetV2 (OpenCV DNN)           ─┴─► fused result
                                                       Cow_Healthy / Cow_FMD / Cow_Lumpy / Goat_Healthy / Goat_Unhealthy
                                                       shown in Jupyter + saved to logs\detections.csv
```

Folders in this package — unzip to `C:\LivestockAI` so you get:

| Folder | What | Where it runs |
|---|---|---|
| `C:\LivestockAI\training` | dataset, ML + DL training, exports | your Windows PC |
| `C:\LivestockAI\hls_cnn` | HLS C++ of the FPGA CNN | Vitis HLS 2023.2 |
| `C:\LivestockAI\pynq_app` | camera app + tests | PYNQ-Z2 board |

---

## PHASE 1 – Python setup on the PC (once)
Command Prompt:
```
cd C:\LivestockAI
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r training\requirements.txt
```
(If `.venv` does not exist: `py -3.11 -m venv .venv` first.)

## PHASE 2 – Build the combined cow + goat dataset (once)
```
cd C:\LivestockAI\training
python step0_build_dataset.py
```
It finds your old datasets automatically and COPIES them (originals untouched):
- cows: `C:\AI-Environment_OLD\dataset` (5,204 images)
- goats: `C:\Users\VLSI\Downloads\archive (4)\healthy_goat` + `unhealthy_goat`

✅ Prints 5 classes with counts. If it says goat "NOT FOUND", run:
`python step0_build_dataset.py --goat-healthy "FOLDER" --goat-unhealthy "FOLDER"`

## PHASE 3 – Train everything (PC, ~1.5–2.5 h on CPU)
Run one by one (or all: `run_all.bat`):

| Step | Command | You get |
|---|---|---|
| 1 | `python step1_check_dataset.py` | counts, `outputs\sample_grid.png` |
| 2 | `python step2_prepare_data.py` | 70/15/15 split + 2 image caches |
| 3 | `python step3_train_ml.py` | ML: LogReg / SVM / RandomForest / KNN |
| 4 | `python step4_train_dl.py` | DL: GreenCNN from scratch |
| 6 | `python step6_export_hls.py` | int8 FPGA weights `hls_export\cnn_weights.h` (+ test vectors) |
| 7 | `python step7_train_mobilenetv2.py` | MobileNetV2 + custom head → `deploy\mobilenetv2.onnx` |
| 8 | `python step8_fusion.py` | best CNN/MobileNetV2 mix → `deploy\fusion.json` |
| 5 | `python step5_compare.py` | `outputs\comparison.png` (all models, for the report) |

**Test on the PC before touching the board** (laptop webcam, FPGA simulated bit-exactly):
```
cd C:\LivestockAI\pynq_app
copy ..\training\deploy\* .
python livestock_cam.py --sim --display window      (q = quit)
python livestock_cam.py --sim --image "C:\path\to\cow.jpg"
```

---

## PHASE 4 – Vitis HLS: CNN C++ → IP (manual, GUI)
Copy `training\hls_export\cnn_weights.h` and `cnn_test_vectors.h` into `C:\LivestockAI\hls_cnn`.
Open **Vitis HLS 2023.2** (click once, wait; answer **No** to the deprecation notice).

1. **File → New Project**: name `green_cnn_prj`, location `C:\gv\cnn_work` → Next
2. **Design files → Add Files**: `green_cnn.cpp`, `green_cnn.h`, `cnn_weights.h`.
   **Top Function → Browse → green_cnn** (the one in green_cnn.cpp) → Next
3. **Testbench → Add Files**: `tb_green_cnn.cpp`, `cnn_test_vectors.h` → Next
4. **Solution**: period `10`; Part **…** → **Parts** → search `xc7z020clg400` → click the row → OK;
   Flow Target **Vivado IP Flow Target** → Finish
5. **Check**: Project → Project Settings → Synthesis → Top Function = `green_cnn` (type it if empty)
6. **Run C Simulation → OK** ✅ 16 lines `BIT-EXACT`, `*** ALL TESTS PASSED ***`
   (the C++ gives exactly the same numbers as your trained Python model)
7. **Run C Synthesis → Active Solution → OK** (2–5 min). In the report note:
   - Estimated Fmax ≥ 100 MHz
   - Latency ≈ 1.5–2 million cycles (≈ 15–20 ms per image at 100 MHz)
   - BRAM / DSP / FF / LUT — all must be < 100 %
   - Interface: `s_axi_CTRL` (control) + `m_axi_gmem` (reads image from DDR)
8. *(optional, slow)* Cosimulation: Project Settings → Simulation → tb file CFLAGS `-DTB_IMAGES=2` → Run Cosimulation
9. **Export RTL → Vivado IP (.zip) → OK** ✅ `Finished Export RTL`
   IP folder: `C:\gv\cnn_work\green_cnn_prj\solution1\impl\ip`

## PHASE 5 – Vivado: IP + Zynq → bitstream (manual, GUI)
1. **Create Project** → name `green_cnn`, location `C:/gv/cnn_vivado` → RTL Project, ☑ Do not specify sources →
   **Boards** → `pynq-z2` → Finish
2. **Settings → IP → Repository → +** → the `impl\ip` folder above → "1 IP" → OK
3. **Create Block Design** → name `green_cnn`
4. **+ → ZYNQ7 Processing System** → **Run Block Automation** (Apply Board Preset ☑) → OK
5. **NEW vs the vitals test – enable the DDR port for the CNN:**
   double-click the Zynq block → **PS-PL Configuration → HP Slave AXI Interface → ☑ S AXI HP0 interface** → OK
   (the CNN reads camera images from DDR memory through HP0)
6. **+ → Green_cnn** (instance must start with `green_cnn`, e.g. `green_cnn_0`)
7. **Run Connection Automation → ☑ All Automation → OK**. You now see:
   - `M_AXI_GP0 → AXI Interconnect → green_cnn_0/s_axi_CTRL` (CPU controls the IP)
   - `green_cnn_0/m_axi_gmem → AXI SmartConnect/Interconnect → S_AXI_HP0` (IP reads DDR)
   - clocks + Processor System Reset
8. **F6 Validate** ✅ no errors → Ctrl+S
   **Address Editor**: green_cnn_0 has `s_axi_CTRL` (≈0x4000_0000) and a `Data_m_axi_gmem` space mapped to
   `HP0_DDR_LOWOCM` (0x0000_0000, 512 M)
9. **Sources → right-click green_cnn.bd → Create HDL Wrapper → Let Vivado manage → OK**
10. **Generate Bitstream → Yes → OK** (10–25 min)
11. **Open Implemented Design**: Design Runs → impl_1 **WNS ≥ 0**; note LUT / FF / BRAM / DSP for the report

## PHASE 6 – Collect the files (PC)
Make `C:\gv\cnn_overlay` and put in it:

| File | From |
|---|---|
| `green_cnn.bit` | `C:\gv\cnn_vivado\green_cnn\green_cnn.runs\impl_1\green_cnn_wrapper.bit` (**rename**) |
| `green_cnn.hwh` | `C:\gv\cnn_vivado\green_cnn\green_cnn.gen\sources_1\bd\green_cnn\hw_handoff\green_cnn.hwh` |
| `mobilenetv2.onnx`, `mnv2_meta.json`, `cnn_meta.json`, `fusion.json`, `cnn_test_vectors.npz`, `cnn_int8_params.npz` | `C:\LivestockAI\training\deploy\` |
| `livestock_cam.py`, `test_fpga_cnn.py`, `test_camera.py` | `C:\LivestockAI\pynq_app\` |

## PHASE 7 – PYNQ-Z2 board (manual)
1. Board: boot jumper **SD**, power jumper **USB**, Ethernet to PC, PC IP `192.168.2.1`, power ON, `ping 192.168.2.99`
2. **Plug the USB camera** into the board's **USB host** port (the full-size USB-A socket)
3. Chrome → `http://192.168.2.99:9090` → password `xilinx`
4. New folder `livestock` → open it → **Upload** all files from `C:\gv\cnn_overlay`
5. New Notebook (in that folder). Run each in its own cell:

```python
%run test_camera.py        # ✅ CAMERA OK + cam_test.jpg
```
```python
%run test_fpga_cnn.py      # ✅ 16/16 bit-exact, FPGA time per image
```
```python
%run livestock_cam.py --frames 200      # live video with label in the notebook
```
Stop early: **Kernel → Interrupt** (■ button). Results: `logs/detections.csv`.
Single photo, no camera: `%run livestock_cam.py --image cow.jpg`

---

## Troubleshooting
| Problem | Fix |
|---|---|
| HLS C sim `MISMATCH` | `cnn_weights.h` and `cnn_test_vectors.h` come from different step6 runs — copy both again |
| HLS "Top function is empty" | Project Settings → Synthesis → type `green_cnn` |
| Vivado Connection Automation doesn't connect `m_axi_gmem` | Phase 5 step 5 (HP0) was skipped — enable it, run automation again |
| `test_fpga_cnn` TimeoutError | HP0 not connected or wrong .hwh; .bit and .hwh must come from the same build and have the same name |
| Camera not found | `%run test_camera.py 1`; try another USB port; use a UVC webcam |
| Only "Uncertain" | Confidence < `confidence_threshold` in `fusion.json` (0.60) — lower it, or train with more images |
| Slow FPS | MobileNetV2 on the ARM is the slow part; the FPGA CNN alone is ~20 ms |

**Goat note:** the goat dataset only says healthy / unhealthy, so the system reports `Goat_Unhealthy`
without naming the disease. LSD does not occur in goats. To name goat diseases, collect labelled goat
photos, add them to `labels.csv` + `classes.json` with new class numbers, and re-run Phases 3–7 —
no code changes.
