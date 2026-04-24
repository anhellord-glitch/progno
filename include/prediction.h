#ifndef PREDICTION_H
#define PREDICTION_H

#include <stdint.h>
#include <stdbool.h>

/* ==================== КОДЫ ВОЗВРАТА ==================== */
#define PREDICT_OK           0
#define PREDICT_ERR_NULL     1
#define PREDICT_ERR_INV_N    2
#define PREDICT_ERR_INV_P    3
#define PREDICT_ERR_MEM      4
#define PREDICT_ERR_STEP1    5
#define PREDICT_ERR_STEP2    6
#define PREDICT_ERR_STEP3    7
#define PREDICT_ERR_STEP4    8
#define PREDICT_ERR_NAN      9
#define PREDICT_ERR_MAT_SIZE 10

/* ==================== СТРУКТУРЫ ==================== */
/* ПРГ:Тр-1, ПРГ:Тс-1, ПРГ:Тс-6, ПРГ:Тс-7 */
typedef struct {
    bool x_updated;      /* Обновлен прогноз вектора X */
    bool p_updated;      /* Обновлен прогноз матрицы P */
    int32_t error_code;
    uint32_t failed_step;
} PredictionResult;

/* ==================== ПРГ:Ту-1, ПРГ:Тр-1 ==================== */
int32_t predict_state(
    uint32_t n, uint32_t p,
    const float *X_est, const float *P_est,
    float *X_pred, float *P_pred,
    const float *F, const float *G, const float *Q
);

PredictionResult predict_state_ex(
    uint32_t n, uint32_t p,
    const float *X_est, const float *P_est,
    float *X_pred, float *P_pred,
    const float *F, const float *G, const float *Q
);

/* ==================== ПРГ:Тс-2 ==================== */
int32_t predict_state_vector(
    uint32_t n,
    const float *X_est,
    float *X_pred,
    const float *F
);

/* ==================== ПРГ:Тс-3 ==================== */
int32_t predict_covariance_matrix(
    uint32_t n, uint32_t p,
    const float *P_est,
    float *P_pred,
    const float *F,
    const float *G,
    const float *Q
);

/* ==================== ПРГ:Тс-4 ==================== */
int32_t update_state_estimate(
    uint32_t n,
    float *X_est,
    const float *X_pred
);

/* ==================== ПРГ:Тс-5 ==================== */
int32_t update_covariance_estimate(
    uint32_t n,
    float *P_est,
    const float *P_pred
);

/* ==================== ПРГ:Тс-1, ПРГ:Тс-6, ПРГ:Тс-7 ==================== */
PredictionResult predict_full(
    uint32_t n, uint32_t p,
    float *X_est, float *P_est,
    float *X_pred, float *P_pred,
    const float *F, const float *G, const float *Q
);

#endif /* PREDICTION_H */
