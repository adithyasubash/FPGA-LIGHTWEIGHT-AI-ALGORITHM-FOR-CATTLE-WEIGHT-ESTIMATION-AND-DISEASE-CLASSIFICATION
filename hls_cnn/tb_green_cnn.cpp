// =============================================================================
// tb_green_cnn.cpp -- C simulation / co-simulation testbench.
// Runs the real test images exported by step6 and checks that every logit is
// BIT-EXACT with the Python integer model. Returns 0 = PASS, 1 = FAIL.
// cnn_test_vectors.h must be next to this file (copy from training\hls_export).
// =============================================================================
#include <cstdio>
#include "green_cnn.h"
#include "cnn_test_vectors.h"

// co-simulation is slow (~1.5 M cycles per image): set to 2 for cosim, N_VECTORS for csim
#ifndef TB_IMAGES
#define TB_IMAGES N_VECTORS
#endif

int main()
{
    static unsigned char img[IMG_BYTES];
    static int out[OUT_WORDS];
    int fails = 0, correct = 0;
    const int n = TB_IMAGES < N_VECTORS ? TB_IMAGES : N_VECTORS;

    for (int t = 0; t < n; t++) {
        const unsigned char *src = &TV_IMG[t][0][0][0];
        for (int i = 0; i < IMG_BYTES; i++) img[i] = src[i];

        green_cnn(img, out);

        bool ok = true;
        int exp_cls = 0;
        for (int k = 0; k < NUM_CLASSES; k++) {
            if (out[k] != (int)TV_LOGITS[t][k]) ok = false;
            if (TV_LOGITS[t][k] > TV_LOGITS[t][exp_cls]) exp_cls = k;
        }
        if (out[NUM_CLASSES] != exp_cls) ok = false;
        if (out[NUM_CLASSES] == TV_LABEL[t]) correct++;
        if (!ok) fails++;

        printf("image %2d  class=%d (%-14s) true=%d (%-14s) logits:", t, out[NUM_CLASSES],
               CLASS_NAMES[out[NUM_CLASSES]], TV_LABEL[t], CLASS_NAMES[TV_LABEL[t]]);
        for (int k = 0; k < NUM_CLASSES; k++) printf(" %d", out[k]);
        printf("  %s\n", ok ? "BIT-EXACT" : "MISMATCH");
    }
    printf("\n%d/%d images bit-exact with Python, %d/%d correct class\n",
           n - fails, n, correct, n);
    printf(fails ? "*** TEST FAILED ***\n" : "*** ALL TESTS PASSED ***\n");
    return fails ? 1 : 0;
}
