// =============================================================================
// green_cnn.cpp -- GreenCNN int8 inference engine (from-scratch HLS accelerator)
//
//  DDR image (3x64x64 uint8)
//    -> conv3x3+ReLU+requant+maxpool  3->16   64x64 -> 32x32
//    -> conv3x3+ReLU+requant+maxpool 16->32   32x32 -> 16x16
//    -> conv3x3+ReLU+requant+maxpool 32->48   16x16 ->  8x8
//    -> conv3x3+ReLU+requant+maxpool 48->64    8x8  ->  4x4
//    -> global sum pool -> fully connected -> logits + argmax -> DDR
//
// Integer maths is bit-exact with int_forward() in training\step6_export_hls.py.
// PAR (=8) output channels are computed in parallel: 8 MACs per clock.
// Activations live in on-chip BRAM with a 1-pixel zero border (= padding 1).
// =============================================================================
#include "green_cnn.h"

// One layer: 3x3 conv (pad 1) + bias + ReLU + requantise to uint8 + 2x2 max-pool.
//   in  : CIN  x (H+2)   x (H+2)     (zero border)
//   out : COUT x (H/2+2) x (H/2+2)   (zero border, ready for the next layer)
template <int CIN, int COUT, int H>
static void conv_block(act_t in[CIN][H + 2][H + 2],
                       act_t out[COUT][H / 2 + 2][H / 2 + 2],
                       const signed char W[COUT][CIN][3][3],
                       const int B[COUT], const int M[COUT])
{
    const int HO = H / 2 + 2;

    // clear the whole output: border = padding for the next layer, inside = start of max-pool
CLR_G:
    for (int g = 0; g < COUT / PAR; g++)
    CLR_Y:
        for (int y = 0; y < HO; y++)
        CLR_X:
            for (int x = 0; x < HO; x++) {
#pragma HLS PIPELINE II=1
            CLR_P:
                for (int p = 0; p < PAR; p++) {
#pragma HLS UNROLL
                    out[g * PAR + p][y][x] = 0;
                }
            }

GROUP:
    for (int g = 0; g < COUT / PAR; g++) {
    ROW:
        for (int y = 0; y < H; y++) {
        COL:
            for (int x = 0; x < H; x++) {
                int acc[PAR];
#pragma HLS ARRAY_PARTITION variable=acc complete
            INIT:
                for (int p = 0; p < PAR; p++) {
#pragma HLS UNROLL
                    acc[p] = B[g * PAR + p];
                }
            CI:
                for (int c = 0; c < CIN; c++)
                KY:
                    for (int ky = 0; ky < 3; ky++)
                    KX:
                        for (int kx = 0; kx < 3; kx++) {
#pragma HLS PIPELINE II=1
                            int v = in[c][y + ky][x + kx];
                        MAC:
                            for (int p = 0; p < PAR; p++) {
#pragma HLS UNROLL
                                acc[p] += (int)W[g * PAR + p][c][ky][kx] * v;
                            }
                        }
            STORE:
                for (int p = 0; p < PAR; p++) {
#pragma HLS UNROLL
                    int a = acc[p] < 0 ? 0 : acc[p];                                   // ReLU
                    long long r = ((long long)a * M[g * PAR + p] + (1LL << (REQ_SHIFT - 1)))
                                  >> REQ_SHIFT;                                         // requant
                    act_t q = r > 255 ? (act_t)255 : (act_t)r;                          // clip
                    int py = (y >> 1) + 1, px = (x >> 1) + 1;
                    if (q > out[g * PAR + p][py][px]) out[g * PAR + p][py][px] = q;     // max-pool
                }
            }
        }
    }
}

void green_cnn(const unsigned char *img, int *out)
{
#pragma HLS INTERFACE mode=m_axi port=img offset=slave bundle=gmem depth=12288 max_read_burst_length=256
#pragma HLS INTERFACE mode=m_axi port=out offset=slave bundle=gmem depth=16
#pragma HLS INTERFACE mode=s_axilite port=img bundle=CTRL
#pragma HLS INTERFACE mode=s_axilite port=out bundle=CTRL
#pragma HLS INTERFACE mode=s_axilite port=return bundle=CTRL

    // weights: PAR banks so 8 output channels are read every clock
#pragma HLS ARRAY_PARTITION variable=W1 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=W2 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=W3 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=W4 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=B1 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=B2 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=B3 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=B4 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=M1 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=M2 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=M3 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=M4 cyclic factor=8 dim=1

    // on-chip activation buffers (BRAM), each with a 1-pixel zero border
    unsigned char raw[IMG_BYTES];
    act_t b0[L1_CIN][IMG_SIZE + 2][IMG_SIZE + 2];
    act_t b1[L1_COUT][IMG_SIZE / 2 + 2][IMG_SIZE / 2 + 2];
    act_t b2[L2_COUT][IMG_SIZE / 4 + 2][IMG_SIZE / 4 + 2];
    act_t b3[L3_COUT][IMG_SIZE / 8 + 2][IMG_SIZE / 8 + 2];
    act_t b4[L4_COUT][IMG_SIZE / 16 + 2][IMG_SIZE / 16 + 2];
#pragma HLS ARRAY_PARTITION variable=b1 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=b2 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=b3 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=b4 cyclic factor=8 dim=1

    // 1) burst-read the image from DDR
READ:
    for (int i = 0; i < IMG_BYTES; i++) {
#pragma HLS PIPELINE II=1
        raw[i] = img[i];
    }
    // 2) place it inside a zero border
PAD_C:
    for (int c = 0; c < L1_CIN; c++)
    PAD_Y:
        for (int y = 0; y < IMG_SIZE + 2; y++)
        PAD_X:
            for (int x = 0; x < IMG_SIZE + 2; x++) {
#pragma HLS PIPELINE II=1
                bool inside = y >= 1 && y <= IMG_SIZE && x >= 1 && x <= IMG_SIZE;
                b0[c][y][x] = inside ? raw[(c * IMG_SIZE + (y - 1)) * IMG_SIZE + (x - 1)] : (act_t)0;
            }

    // 3) the four conv blocks
    conv_block<L1_CIN, L1_COUT, IMG_SIZE>(b0, b1, W1, B1, M1);
    conv_block<L2_CIN, L2_COUT, IMG_SIZE / 2>(b1, b2, W2, B2, M2);
    conv_block<L3_CIN, L3_COUT, IMG_SIZE / 4>(b2, b3, W3, B3, M3);
    conv_block<L4_CIN, L4_COUT, IMG_SIZE / 8>(b3, b4, W4, B4, M4);

    // 4) global sum-pool over 4x4 (no divide: the 1/16 is folded into FC_B)
    int gap[L4_COUT];
GAP:
    for (int c = 0; c < L4_COUT; c++) {
        int s = 0;
    GAP_Y:
        for (int y = 1; y <= IMG_SIZE / 16; y++)
        GAP_X:
            for (int x = 1; x <= IMG_SIZE / 16; x++) {
#pragma HLS PIPELINE II=1
                s += b4[c][y][x];
            }
        gap[c] = s;
    }

    // 5) fully connected layer + argmax, results written back to DDR
    int best = 0;
    long long best_v = 0;
FC:
    for (int k = 0; k < NUM_CLASSES; k++) {
        long long s = FC_B[k];
    FC_C:
        for (int c = 0; c < L4_COUT; c++) {
#pragma HLS PIPELINE II=1
            s += (long long)FC_W[k][c] * gap[c];
        }
        out[k] = (int)s;
        if (k == 0 || s > best_v) {
            best_v = s;
            best = k;
        }
    }
    out[NUM_CLASSES] = best;
}
