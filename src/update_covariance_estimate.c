#include "prediction.h"
#include <math.h>

/* ПРГ:Тс-5: P' = P */
int32_t update_covariance_estimate(uint32_t n, float *P_est, const float *P_pred) {
    if (n == 0) return PREDICT_ERR_INV_N;
    if (!P_est || !P_pred) return PREDICT_ERR_NULL;

    if (P_est != P_pred) {
        for (uint32_t i = 0; i < n * n; i++)
            P_est[i] = P_pred[i];
    }

    for (uint32_t i = 0; i < n * n; i++)
        if (isnan(P_est[i])) return PREDICT_ERR_NAN;
    return PREDICT_OK;
}
