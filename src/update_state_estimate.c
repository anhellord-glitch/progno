#include "prediction.h"
#include <math.h>

/* ПРГ:Тс-4: X' = X */
int32_t update_state_estimate(uint32_t n, float *X_est, const float *X_pred) {
    if (n == 0) return PREDICT_ERR_INV_N;
    if (!X_est || !X_pred) return PREDICT_ERR_NULL;

    if (X_est != X_pred) {
        for (uint32_t i = 0; i < n; i++)
            X_est[i] = X_pred[i];
    }

    for (uint32_t i = 0; i < n; i++)
        if (isnan(X_est[i])) return PREDICT_ERR_NAN;
    return PREDICT_OK;
}
