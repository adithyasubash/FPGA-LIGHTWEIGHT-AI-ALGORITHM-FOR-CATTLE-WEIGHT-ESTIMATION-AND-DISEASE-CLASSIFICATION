// =============================================================================
// tb_vitals_mlp.cpp  --  C simulation / C-RTL cosim testbench.
// Returns 0 on pass, 1 on fail (Vitis HLS uses this to decide PASS/FAIL).
// =============================================================================
#include <cstdio>
#include "vitals_mlp.h"

struct Case { const char *name; int v[NIN]; int thr; int exp_score; int exp_alert; };

int main()
{
    // temp_x10, hum, hr, act, gait, resp, feed, spare
    Case cases[] = {
        { "healthy goat",      {390, 60,  80, 400, 100, 25, 0, 0}, 40,   0, 0 },
        { "fever + fast HR",   {412, 60, 120, 400, 100, 25, 0, 0}, 40, 135, 1 },
        { "lame + lethargic",  {390, 60,  80,  50, 450, 25, 0, 0}, 40, 250, 1 },
        { "mild heat stress",  {395, 90,  85, 300, 100, 30, 0, 0}, 40,  10, 0 },
    };
    const int N = sizeof(cases) / sizeof(cases[0]);

    int fails = 0;
    for (int k = 0; k < N; k++) {
        int score = -1, alert = -1;
        const int *v = cases[k].v;
        vitals_mlp(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
                   cases[k].thr, &score, &alert);
        bool ok = (score == cases[k].exp_score) && (alert == cases[k].exp_alert);
        printf("%-18s score=%4d alert=%d  (expected %4d/%d)  %s\n",
               cases[k].name, score, alert, cases[k].exp_score,
               cases[k].exp_alert, ok ? "PASS" : "FAIL");
        if (!ok) fails++;
    }
    printf(fails ? "\n*** TEST FAILED (%d) ***\n" : "\n*** ALL TESTS PASSED ***\n", fails);
    return fails ? 1 : 0;
}
