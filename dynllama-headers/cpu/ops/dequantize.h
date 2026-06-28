//
// Dequantization op: convert GGUF block-quantized tensor data to float32.
//
// Supports:
//   F32, F16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q6_K
// Type ids match enum ggml_type. Q4_K/Q6_K validated bit-exact against the
// gguf-py reference dequantizer. Other K-quants / IQ-quants return false.
//
// Each quantized format packs values into fixed-size blocks of QK elements.
// Layout follows ggml-common.h exactly so the same .gguf data decodes
// identically.
//

#pragma once

#include <cstdint>
#include <cstring>

// subset of enum ggml_type (see dynllama-headers/.../ggml.h)
enum dynllama_qtype {
    DQ_F32  = 0,
    DQ_F16  = 1,
    DQ_Q4_0 = 2,
    DQ_Q4_1 = 3,
    DQ_Q5_0 = 6,
    DQ_Q5_1 = 7,
    DQ_Q8_0 = 8,
    DQ_Q4_K = 12,
    DQ_Q6_K = 14,
};

#define DQ_QK   32   // block size for the legacy quant formats
#define DQ_QK_K 256  // super-block size for the K-quant formats

// Unpack a 6-bit scale and 6-bit min for sub-block j from the packed 12-byte
// q4_K scales array (matches ggml get_scale_min_k4).
static inline void dq_get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>   4) | ((q[j - 0] >> 6) << 4);
    }
}

// alias-safe little-endian uint16 read from a byte buffer
static inline uint16_t dq_rd_u16(const uint8_t * p) {
    uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
}

// half -> float (IEEE 754 binary16 -> binary32)
static inline float dq_fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    const uint32_t exp  = (h >> 10) & 0x1F;
    const uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;                       // +/- zero
        } else {
            // subnormal: normalize
            int e = -1;
            uint32_t m = mant;
            do { m <<= 1; e++; } while ((m & 0x400) == 0);
            m &= 0x3FF;
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | (m << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000 | (mant << 13); // inf / nan
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// Dequantize `nelements` values of the given type from `src` into `dst`.
// `nelements` must be a multiple of the format's block size for quantized types.
// Returns false for unsupported types.
static inline bool dequantize_row(int type, const void * src, float * dst, int64_t nelements) {
    const uint8_t * p = (const uint8_t *) src;

    switch (type) {
        case DQ_F32: {
            std::memcpy(dst, src, (size_t) nelements * sizeof(float));
            return true;
        }
        case DQ_F16: {
            const uint8_t * h = (const uint8_t *) src;
            for (int64_t i = 0; i < nelements; i++) dst[i] = dq_fp16_to_fp32(dq_rd_u16(h + 2 * i));
            return true;
        }
        case DQ_Q8_0: {
            // block: fp16 d, int8 qs[32]
            const int64_t nb = nelements / DQ_QK;
            for (int64_t b = 0; b < nb; b++) {
                const float d = dq_fp16_to_fp32(dq_rd_u16(p)); p += 2;
                const int8_t * qs = (const int8_t *) p;                 p += DQ_QK;
                for (int j = 0; j < DQ_QK; j++) dst[b * DQ_QK + j] = qs[j] * d;
            }
            return true;
        }
        case DQ_Q4_0: {
            // block: fp16 d, uint8 qs[16] (two 4-bit nibbles each), values in [-8,7]
            const int64_t nb = nelements / DQ_QK;
            for (int64_t b = 0; b < nb; b++) {
                const float d = dq_fp16_to_fp32(dq_rd_u16(p)); p += 2;
                const uint8_t * qs = p;                                 p += DQ_QK / 2;
                for (int j = 0; j < DQ_QK / 2; j++) {
                    const int x0 = (qs[j] & 0x0F) - 8;
                    const int x1 = (qs[j] >> 4)   - 8;
                    dst[b * DQ_QK + j]              = x0 * d;
                    dst[b * DQ_QK + j + DQ_QK / 2]  = x1 * d;
                }
            }
            return true;
        }
        case DQ_Q4_1: {
            // block: fp16 d, fp16 m, uint8 qs[16]; value = x*d + m
            const int64_t nb = nelements / DQ_QK;
            for (int64_t b = 0; b < nb; b++) {
                const float d = dq_fp16_to_fp32(dq_rd_u16(p)); p += 2;
                const float m = dq_fp16_to_fp32(dq_rd_u16(p)); p += 2;
                const uint8_t * qs = p;                                 p += DQ_QK / 2;
                for (int j = 0; j < DQ_QK / 2; j++) {
                    const int x0 = qs[j] & 0x0F;
                    const int x1 = qs[j] >> 4;
                    dst[b * DQ_QK + j]              = x0 * d + m;
                    dst[b * DQ_QK + j + DQ_QK / 2]  = x1 * d + m;
                }
            }
            return true;
        }
        case DQ_Q5_0: {
            // block: fp16 d, uint8 qh[4] (high bits), uint8 qs[16]; values in [-16,15]
            const int64_t nb = nelements / DQ_QK;
            for (int64_t b = 0; b < nb; b++) {
                const float d = dq_fp16_to_fp32(dq_rd_u16(p)); p += 2;
                uint32_t qh; std::memcpy(&qh, p, 4);                    p += 4;
                const uint8_t * qs = p;                                 p += DQ_QK / 2;
                for (int j = 0; j < DQ_QK / 2; j++) {
                    const int xh0 = ((qh >> (j))        << 4) & 0x10;
                    const int xh1 = ((qh >> (j + 16))   << 4) & 0x10;
                    const int x0 = ((qs[j] & 0x0F) | xh0) - 16;
                    const int x1 = ((qs[j] >> 4)   | xh1) - 16;
                    dst[b * DQ_QK + j]              = x0 * d;
                    dst[b * DQ_QK + j + DQ_QK / 2]  = x1 * d;
                }
            }
            return true;
        }
        case DQ_Q5_1: {
            // block: fp16 d, fp16 m, uint8 qh[4], uint8 qs[16]; value = x*d + m
            const int64_t nb = nelements / DQ_QK;
            for (int64_t b = 0; b < nb; b++) {
                const float d = dq_fp16_to_fp32(dq_rd_u16(p)); p += 2;
                const float m = dq_fp16_to_fp32(dq_rd_u16(p)); p += 2;
                uint32_t qh; std::memcpy(&qh, p, 4);                    p += 4;
                const uint8_t * qs = p;                                 p += DQ_QK / 2;
                for (int j = 0; j < DQ_QK / 2; j++) {
                    const int xh0 = ((qh >> (j))      << 4) & 0x10;
                    const int xh1 = ((qh >> (j + 16)) << 4) & 0x10;
                    const int x0 = (qs[j] & 0x0F) | xh0;
                    const int x1 = (qs[j] >> 4)   | xh1;
                    dst[b * DQ_QK + j]              = x0 * d + m;
                    dst[b * DQ_QK + j + DQ_QK / 2]  = x1 * d + m;
                }
            }
            return true;
        }
        case DQ_Q4_K: {
            // super-block (256): fp16 d, fp16 dmin, uint8 scales[12], uint8 qs[128]
            const int64_t nb = nelements / DQ_QK_K;
            for (int64_t b = 0; b < nb; b++) {
                const float d    = dq_fp16_to_fp32(dq_rd_u16(p)); p += 2;
                const float dmin = dq_fp16_to_fp32(dq_rd_u16(p)); p += 2;
                const uint8_t * scales = p;                                p += 12;
                const uint8_t * qs     = p;                                p += 128;

                float * y = dst + b * DQ_QK_K;
                int is = 0;
                uint8_t sc, m;
                for (int j = 0; j < DQ_QK_K; j += 64) {
                    dq_get_scale_min_k4(is + 0, scales, &sc, &m);
                    const float d1 = d * sc, m1 = dmin * m;
                    dq_get_scale_min_k4(is + 1, scales, &sc, &m);
                    const float d2 = d * sc, m2 = dmin * m;
                    for (int l = 0; l < 32; l++) *y++ = d1 * (qs[l] & 0x0F) - m1;
                    for (int l = 0; l < 32; l++) *y++ = d2 * (qs[l] >>   4) - m2;
                    qs += 32;
                    is += 2;
                }
            }
            return true;
        }
        case DQ_Q6_K: {
            // super-block (256): uint8 ql[128], uint8 qh[64], int8 scales[16], fp16 d
            const int64_t nb = nelements / DQ_QK_K;
            for (int64_t b = 0; b < nb; b++) {
                const uint8_t * ql = p;                                    p += 128;
                const uint8_t * qh = p;                                    p += 64;
                const int8_t  * sc = (const int8_t *) p;                   p += 16;
                const float d = dq_fp16_to_fp32(dq_rd_u16(p));    p += 2;

                float * y = dst + b * DQ_QK_K;
                for (int n = 0; n < DQ_QK_K; n += 128) {
                    for (int l = 0; l < 32; l++) {
                        const int is = l / 16;
                        const int8_t q1 = (int8_t)((ql[l +  0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        const int8_t q2 = (int8_t)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        const int8_t q3 = (int8_t)((ql[l +  0] >>   4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        const int8_t q4 = (int8_t)((ql[l + 32] >>   4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                        y[l +  0] = d * sc[is + 0] * q1;
                        y[l + 32] = d * sc[is + 2] * q2;
                        y[l + 64] = d * sc[is + 4] * q3;
                        y[l + 96] = d * sc[is + 6] * q4;
                    }
                    y  += 128;
                    ql += 64;
                    qh += 32;
                    sc += 8;
                }
            }
            return true;
        }
        default:
            return false; // other K-quants / IQ-quants not yet supported
    }
}

// Bytes occupied by `nelements` of the given quant type (block-aligned).
// Returns 0 for unsupported types.
static inline int64_t dequantize_type_size(int type, int64_t nelements) {
    switch (type) {
        case DQ_F32:  return nelements * 4;
        case DQ_F16:  return nelements * 2;
        case DQ_Q8_0: return (nelements / DQ_QK)   * (2 + DQ_QK);
        case DQ_Q4_0: return (nelements / DQ_QK)   * (2 + DQ_QK / 2);
        case DQ_Q4_1: return (nelements / DQ_QK)   * (2 + 2 + DQ_QK / 2);
        case DQ_Q5_0: return (nelements / DQ_QK)   * (2 + 4 + DQ_QK / 2);
        case DQ_Q5_1: return (nelements / DQ_QK)   * (2 + 2 + 4 + DQ_QK / 2);
        case DQ_Q4_K: return (nelements / DQ_QK_K) * (2 + 2 + 12 + 128);
        case DQ_Q6_K: return (nelements / DQ_QK_K) * (128 + 64 + 16 + 2);
        default:      return 0;
    }
}
