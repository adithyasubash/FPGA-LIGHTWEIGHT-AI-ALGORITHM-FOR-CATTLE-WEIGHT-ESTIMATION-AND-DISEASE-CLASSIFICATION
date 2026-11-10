"""
test_overlay.py -- run on the PYNQ-Z2 (Jupyter cell or: sudo python3 test_overlay.py)
Needs green_vlsi.bit and green_vlsi.hwh in the same folder (same base name!).
"""
import time
from pynq import Overlay

ol = Overlay("green_vlsi.bit")          # programs the FPGA (blue DONE LED blinks)
print("IP blocks in the bitstream:", list(ol.ip_dict.keys()))

ip = ol.vitals_mlp_0                    # name of the block in your Vivado block design
print(ip.register_map)                  # shows every register and its offset


def score_animal(temp_x10, humidity, heart_rate, activity, gait_asym,
                 resp_rate, feed_min=0, threshold=40):
    rm = ip.register_map
    rm.temp_x10 = temp_x10
    rm.humidity = humidity
    rm.heart_rate = heart_rate
    rm.activity = activity
    rm.gait_asym = gait_asym
    rm.resp_rate = resp_rate
    rm.feed_min = feed_min
    rm.spare_in = 0
    rm.threshold = threshold
    ip.write(0x00, 1)                   # AP_START
    t = time.time()
    while not (ip.read(0x00) & 0x2):    # wait for AP_DONE
        if time.time() - t > 1:
            raise TimeoutError("IP did not finish -- check clock/reset in block design")
    return int(rm.score), int(rm.alert)


tests = [
    ("healthy goat",     (390, 60,  80, 400, 100, 25), (0,   0)),
    ("fever + fast HR",  (412, 60, 120, 400, 100, 25), (135, 1)),
    ("lame + lethargic", (390, 60,  80,  50, 450, 25), (250, 1)),
    ("mild heat stress", (395, 90,  85, 300, 100, 30), (10,  0)),
]
for name, args, expect in tests:
    got = score_animal(*args)
    print(f"{name:18s} score={got[0]:4d} alert={got[1]}  expected {expect}  "
          f"{'PASS' if got == expect else 'FAIL'}")
