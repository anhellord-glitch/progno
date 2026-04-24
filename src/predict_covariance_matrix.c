#include "prediction.h"
#include <stdlib.h>
#include <math.h>

static void mat_mul(const float *A, uint32_t ra, uint32_t ca,
                    const float *B, uint32_t cb, float *C) {
    for (uint32_t i = 0; i < ra; i++)
        for (uint32_t j = 0; j < cb; j++) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < ca; k++)
                sum += A[i * ca + k] * B[k * cb + j];
            C[i * cb + j] = sum;
        }
}

static void mat_trans(const float *A, uint32_t r, uint32_t c, float *AT) {
    for (uint32_t i = 0; i < r; i++)
        for (uint32_t j = 0; j < c; j++)
            AT[j * r + i] = A[i * c + j];
}

static void mat_add(const float *A, const float *B, uint32_t r, uint32_t c, float *C) {
    for (uint32_t i = 0; i < r * c; i++) C[i] = A[i] + B[i];
}

static void mat_cpy(const float *src, float *dst, uint32_t n) {
    for (uint32_t i = 0; i < n * n; i++) dst[i] = src[i];
}

static bool has_nan(const float *mat, uint32_t n) {
    for (uint32_t i = 0; i < n * n; i++)
        if (isnan(mat[i])) return true;
    return false;
}

/* ПРГ:Тс-3: P = F * P' * F^T + G * Q * G^T */
int32_t predict_covariance_matrix(uint32_t n, uint32_t p,
    const float *P_est, float *P_pred,
    const float *F, const float *G, const float *Q)
{
    if (n == 0) return PREDICT_ERR_INV_N;
    if (!P_est || !P_pred || !F) return PREDICT_ERR_NULL;
    if (p > 0 && (!G || !Q)) return PREDICT_ERR_NULL;

    float *T1 = malloc(n * n * sizeof(float));
    float *FT = malloc(n * n * sizeof(float));
    float *T2 = malloc(n * n * sizeof(float));
    if (!T1 || !FT || !T2) { free(T1); free(FT); free(T2); return PREDICT_ERR_MEM; }

    mat_mul(F, n, n, P_est, n, T1);
    mat_trans(F, n, n, FT);
    mat_mul(T1, n, n, FT, n, T2);

    if (has_nan(T2, n)) { free(T1); free(FT); free(T2); return PREDICT_ERR_NAN; }

    if (p == 0) {
        mat_cpy(T2, P_pred, n);
    } else {
        float *GQ = malloc(n * p * sizeof(float));
        float *GT = malloc(p * n * sizeof(float));
        float *NC = malloc(n * n * sizeof(float));
        if (!GQ || !GT || !NC) {
            free(T1); free(FT); free(T2); free(GQ); free(GT); free(NC);
            return PREDICT_ERR_MEM;
        }
        mat_mul(G, n, p, Q, p, GQ);
        mat_trans(G, n, p, GT);
        mat_mul(GQ, n, p, GT, n, NC);
        if (has_nan(NC, n)) {
            free(T1); free(FT); free(T2); free(GQ); free(GT); free(NC);
            return PREDICT_ERR_NAN;
        }
        mat_add(T2, NC, n, n, P_pred);
        free(GQ); free(GT); free(NC);
    }

    free(T1); free(FT); free(T2);
    if (has_nan(P_pred, n)) return PREDICT_ERR_NAN;
    return PREDICT_OK;
}
