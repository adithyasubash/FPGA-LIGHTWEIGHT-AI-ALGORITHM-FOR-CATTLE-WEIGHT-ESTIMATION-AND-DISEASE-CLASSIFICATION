// =============================================================================
// vitals_mlp.h  --  GREEN_VLSI livestock vitals anomaly scorer (Vitis HLS)
//
// A tiny 2-layer integer MLP:  8 inputs -> 8 hidden (ReLU) -> 1 score.
// Inputs are plain integers sent by the ARM (PS) from each ESP32 node:
//   temp_x10   body temperature x10       (392 = 39.2 C)
//   humidity   ambient humidity %          (from DHT11)
//   heart_rate beats per minute
//   activity   activity index 0..1000      (from IMU, computed on PS/ESP32)
//   gait_asym  gait asymmetry 0..1000      (from IMU gyro, lameness signal)
//   resp_rate  breaths per minute
//   feed_min   minutes spent feeding/ruminating (0 if not measured)
//   spare_in   free input for later use
//
// The weights below are HAND-SET PLACEHOLDERS that encode simple vet rules
// (fever, low temp, fast/slow heart, heat stress, lethargy, lameness, fast
// breathing). Replace W1/B1/W2/B2 with your trained INT8 weights later --
// the hardware does not change, only these numbers.
// =============================================================================
#ifndef VITALS_MLP_H
#define VITALS_MLP_H

#define NIN 8
#define HID 8

// Hidden layer: h[j] = ReLU( sum_i W1[j][i]*x[i] + B1[j] )
static const signed char W1[HID][NIN] = {
//   temp hum  hr  act gait resp feed spare
    {  1,   0,  0,   0,   0,   0,   0,   0 },  // h0 fever        temp > 39.7 C
    { -1,   0,  0,   0,   0,   0,   0,   0 },  // h1 hypothermia  temp < 37.5 C
    {  0,   0,  1,   0,   0,   0,   0,   0 },  // h2 tachycardia  HR > 95
    {  0,   0, -1,   0,   0,   0,   0,   0 },  // h3 bradycardia  HR < 60
    {  0,   1,  0,   0,   0,   0,   0,   0 },  // h4 heat stress  humidity > 80 %
    {  0,   0,  0,  -1,   0,   0,   0,   0 },  // h5 lethargy     activity < 150
    {  0,   0,  0,   0,   1,   0,   0,   0 },  // h6 lameness     gait asym > 300
    {  0,   0,  0,   0,   0,   1,   0,   0 },  // h7 tachypnea    resp > 40
};
static const int B1[HID] = { -397, 375, -95, 60, -80, 150, -300, -40 };

// Output: score = sum_j W2[j]*h[j] + B2
static const signed char W2[HID] = { 4, 3, 3, 2, 1, 1, 1, 2 };
static const int B2 = 0;

// Top function (this becomes the IP "vitals_mlp" in Vivado)
void vitals_mlp(int temp_x10, int humidity, int heart_rate, int activity,
                int gait_asym, int resp_rate, int feed_min, int spare_in,
                int threshold, int *score, int *alert);

#endif
