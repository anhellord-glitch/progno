#include "prediction.h"
#include <math.h>

/* ПРГ:Тс-2: X = F * X' */
int32_t predict_state_vector(uint32_t n, const float *X_est, float *X_pred, const float *F) {
    if (n == 0) return PREDICT_ERR_INV_N;
    if (!X_est || !X_pred || !F) return PREDICT_ERR_NULL;

    for (uint32_t i = 0; i < n; i++) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < n; j++)
            sum += F[i * n + j] * X_est[j];
        X_pred[i] = sum;
    }

    for (uint32_t i = 0; i < n; i++)
        if (isnan(X_pred[i])) return PREDICT_ERR_NAN;
    return PREDICT_OK;
}
