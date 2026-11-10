// =============================================================================
// vitals_mlp.cpp  --  HLS kernel. All ports are AXI4-Lite registers (bundle
// CTRL), so the PYNQ Python code just writes inputs, sets START, reads score.
// =============================================================================
#include "vitals_mlp.h"

void vitals_mlp(int temp_x10, int humidity, int heart_rate, int activity,
                int gait_asym, int resp_rate, int feed_min, int spare_in,
                int threshold, int *score, int *alert)
{
#pragma HLS INTERFACE mode=s_axilite port=temp_x10   bundle=CTRL
#pragma HLS INTERFACE mode=s_axilite port=humidity   bundle=CTRL
#pragma HLS INTERFACE mode=s_axilite port=heart_rate bundle=CTRL
#pragma HLS INTERFACE mode=s_axilite port=activity   bundle=CTRL
#pragma HLS INTERFACE mode=s_axilite port=gait_asym  bundle=CTRL
#pragma HLS INTERFACE mode=s_axilite port=resp_rate  bundle=CTRL
#pragma HLS INTERFACE mode=s_axilite port=feed_min   bundle=CTRL
#pragma HLS INTERFACE mode=s_axilite port=spare_in   bundle=CTRL
#pragma HLS INTERFACE mode=s_axilite port=threshold  bundle=CTRL
#pragma HLS INTERFACE mode=s_axilite port=score      bundle=CTRL
#pragma HLS INTERFACE mode=s_axilite port=alert      bundle=CTRL
#pragma HLS INTERFACE mode=s_axilite port=return     bundle=CTRL

    int x[NIN] = { temp_x10, humidity, heart_rate, activity,
                   gait_asym, resp_rate, feed_min, spare_in };
#pragma HLS ARRAY_PARTITION variable=x complete

    int out = B2;

HIDDEN:
    for (int j = 0; j < HID; j++) {
#pragma HLS UNROLL
        int acc = B1[j];
    MAC:
        for (int i = 0; i < NIN; i++) {
#pragma HLS UNROLL
            acc += (int)W1[j][i] * x[i];
        }
        if (acc < 0) acc = 0;              // ReLU
        out += (int)W2[j] * acc;
    }

    *score = out;
    *alert = (out > threshold) ? 1 : 0;
}
