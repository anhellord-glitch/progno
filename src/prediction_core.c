#include "prediction.h"
#include <string.h>
#include <stdlib.h>

/* Вспомогательные функции */
static void matrix_multiply(const float *A, uint32_t ra, uint32_t ca,
                            const float *B, uint32_t cb, float *C) {
    for (uint32_t i = 0; i < ra; i++)
        for (uint32_t j = 0; j < cb; j++) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < ca; k++)
                sum += A[i * ca + k] * B[k * cb + j];
            C[i * cb + j] = sum;
        }
}

static void matrix_transpose(const float *A, uint32_t r, uint32_t c, float *AT) {
    for (uint32_t i = 0; i < r; i++)
        for (uint32_t j = 0; j < c; j++)
            AT[j * r + i] = A[i * c + j];
}

static void matrix_add(const float *A, const float *B, uint32_t r, uint32_t c, float *C) {
    for (uint32_t i = 0; i < r * c; i++) C[i] = A[i] + B[i];
}

static void matrix_copy(const float *src, float *dst, uint32_t r, uint32_t c) {
    for (uint32_t i = 0; i < r * c; i++) dst[i] = src[i];
}

static void vector_copy(const float *src, float *dst, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) dst[i] = src[i];
}

/* ПРГ:Ту-1, ПРГ:Тр-1 */
int32_t predict_state(uint32_t n, uint32_t p,
    const float *X_est, const float *P_est,
    float *X_pred, float *P_pred,
    const float *F, const float *G, const float *Q)
{
    if (n == 0) return PREDICT_ERR_INV_N;
    if (!X_est || !P_est || !X_pred || !P_pred || !F) return PREDICT_ERR_NULL;
    if (p > 0 && (!G || !Q)) return PREDICT_ERR_NULL;

    for (uint32_t i = 0; i < n; i++) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < n; j++)
            sum += F[i * n + j] * X_est[j];
        X_pred[i] = sum;
    }

    float *T1 = malloc(n * n * sizeof(float));
    float *FT = malloc(n * n * sizeof(float));
    float *T2 = malloc(n * n * sizeof(float));
    if (!T1 || !FT || !T2) {
        free(T1); free(FT); free(T2);
        return PREDICT_ERR_MEM;
    }

    matrix_multiply(F, n, n, P_est, n, T1);
    matrix_transpose(F, n, n, FT);
    matrix_multiply(T1, n, n, FT, n, T2);

    if (p == 0) {
        matrix_copy(T2, P_pred, n, n);
    } else {
        float *GQ = malloc(n * p * sizeof(float));
        float *GT = malloc(p * n * sizeof(float));
        float *NC = malloc(n * n * sizeof(float));
        if (!GQ || !GT || !NC) {
            free(T1); free(FT); free(T2); free(GQ); free(GT); free(NC);
            return PREDICT_ERR_MEM;
        }
        matrix_multiply(G, n, p, Q, p, GQ);
        matrix_transpose(G, n, p, GT);
        matrix_multiply(GQ, n, p, GT, n, NC);
        matrix_add(T2, NC, n, n, P_pred);
        free(GQ); free(GT); free(NC);
    }

    free(T1); free(FT); free(T2);
    return PREDICT_OK;
}

PredictionResult predict_state_ex(uint32_t n, uint32_t p,
    const float *X_est, const float *P_est,
    float *X_pred, float *P_pred,
    const float *F, const float *G, const float *Q)
{
    PredictionResult res = {false, false, PREDICT_OK, 0};
    float *Xb = NULL, *Pb = NULL;
    if (X_pred && n) { Xb = malloc(n * sizeof(float)); if (Xb) vector_copy(X_pred, Xb, n); }
    if (P_pred && n) { Pb = malloc(n * n * sizeof(float)); if (Pb) matrix_copy(P_pred, Pb, n, n); }

    int32_t ret = predict_state(n, p, X_est, P_est, X_pred, P_pred, F, G, Q);
    if (ret == PREDICT_OK) {
        res.x_updated = res.p_updated = true;
        res.error_code = PREDICT_OK;
    } else {
        res.error_code = ret;
        if (Xb && X_pred) vector_copy(Xb, X_pred, n);
        if (Pb && P_pred) matrix_copy(Pb, P_pred, n, n);
    }
    free(Xb); free(Pb);
    return res;
}
