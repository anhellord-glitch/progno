#include "prediction.h"

/* ПРГ:Тс-1, ПРГ:Тс-6, ПРГ:Тс-7 */
PredictionResult predict_full(uint32_t n, uint32_t p,
    float *X_est, float *P_est,
    float *X_pred, float *P_pred,
    const float *F, const float *G, const float *Q)
{
    PredictionResult res = {false, false, PREDICT_OK, 0};

    if (n == 0) {
        res.error_code = PREDICT_ERR_INV_N;
        return res;
    }
    if (!X_est || !P_est || !X_pred || !P_pred || !F) {
        res.error_code = PREDICT_ERR_NULL;
        return res;
    }
    if (p > 0 && (!G || !Q)) {
        res.error_code = PREDICT_ERR_NULL;
        return res;
    }

    /* Шаг 1: X = F * X' (ПРГ:Тс-2) */
    int32_t ret = predict_state_vector(n, X_est, X_pred, F);
    if (ret != PREDICT_OK) {
        res.error_code = ret;
        res.failed_step = 1;
        return res;
    }

    /* Шаг 2: X' = X (ПРГ:Тс-4) */
    ret = update_state_estimate(n, X_est, X_pred);
    if (ret != PREDICT_OK) {
        res.error_code = ret;
        res.failed_step = 2;
        return res;
    }

    /* Шаг 3: P = F*P'*F^T + G*Q*G^T (ПРГ:Тс-3) */
    ret = predict_covariance_matrix(n, p, P_est, P_pred, F, G, Q);
    if (ret != PREDICT_OK) {
        res.error_code = ret;
        res.failed_step = 3;
        res.x_updated = true;  /* X уже обновлен */
        return res;
    }

    /* Шаг 4: P' = P (ПРГ:Тс-5) */
    ret = update_covariance_estimate(n, P_est, P_pred);
    if (ret != PREDICT_OK) {
        res.error_code = ret;
        res.failed_step = 4;
        res.x_updated = true;
        return res;
    }

    /* Шаг 5: Признак X (ПРГ:Тс-7) */
    res.x_updated = true;
    /* Шаг 6: Признак P (ПРГ:Тс-6) */
    res.p_updated = true;

    return res;
}
