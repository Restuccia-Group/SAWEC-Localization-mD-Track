// Include Files
#include "CSIPreprocessor.h"
#include "InterpolateCSIAndRemoveCSDAndAutoUnpermutation_types.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include "crl_permlogic.h"
#include "rt_defines.h"
#include <cfloat>
#include <cmath>
#include <cstring>

// Function Declarations
static void InterpolateCSIAndRemoveCSDAndAutoUnpermutation_init(CSIPreprocessor *aInstancePtr);

static void binary_expand_op(coder::array<boolean_T, 1U> &in1, const coder::array<double, 1U> &in2);

namespace coder {
static void polyfit(const ::coder::array<double, 1U> &x, const ::coder::array<double, 1U> &y, double p[2]);

}
static int div_s32(int numerator, int denominator);

static double rt_atan2d_snf(double u0, double u1);

static double rt_hypotd_snf(double u0, double u1);

static double rt_remd_snf(double u0, double u1);

// Function Definitions
//
// Arguments    : void
// Return Type  : InterpolateCSIAndRemoveCSDAndAutoUnpermutationStackData *
//
InterpolateCSIAndRemoveCSDAndAutoUnpermutationStackData *CSIPreprocessor::getStackData()
{
    return &SD_;
}

//
// Arguments    : CSIPreprocessor *aInstancePtr
// Return Type  : void
//
static void InterpolateCSIAndRemoveCSDAndAutoUnpermutation_init(CSIPreprocessor *aInstancePtr)
{
    InterpolateCSIAndRemoveCSDAndAutoUnpermutationStackData *localSD;
    localSD = aInstancePtr->getStackData();
    localSD->pd->lmag_not_empty = false;
}

//
// Arguments    : coder::array<boolean_T, 1U> &in1
//                const coder::array<double, 1U> &in2
// Return Type  : void
//
static void binary_expand_op(coder::array<boolean_T, 1U> &in1, const coder::array<double, 1U> &in2)
{
    coder::array<boolean_T, 1U> b_in2;
    int i;
    int loop_ub;
    int stride_0_0;
    int stride_1_0;
    if (in1.size(0) == 1) {
        i = in2.size(0);
    } else {
        i = in1.size(0);
    }
    b_in2.set_size(i);
    stride_0_0 = (in2.size(0) != 1);
    stride_1_0 = (in1.size(0) != 1);
    if (in1.size(0) == 1) {
        loop_ub = in2.size(0);
    } else {
        loop_ub = in1.size(0);
    }
    for (i = 0; i < loop_ub; i++) {
        b_in2[i] = ((in2[i * stride_0_0] == 0.0) && in1[i * stride_1_0]);
    }
    in1.set_size(b_in2.size(0));
    loop_ub = b_in2.size(0);
    for (i = 0; i < loop_ub; i++) {
        in1[i] = b_in2[i];
    }
}

//
// Arguments    : const ::coder::array<double, 1U> &x
//                const ::coder::array<double, 1U> &y
//                double p[2]
// Return Type  : void
//
namespace coder {
static void polyfit(const ::coder::array<double, 1U> &x, const ::coder::array<double, 1U> &y, double p[2])
{
    array<double, 2U> V;
    array<double, 1U> B;
    double tau_data[2];
    double temp;
    int i;
    int kend;
    int lastv;
    int m;
    int minmana;
    int pvt;
    int u0;
    signed char jpvt[2];
    V.set_size(x.size(0), 2);
    if (x.size(0) != 0) {
        i = x.size(0);
        for (lastv = 0; lastv < i; lastv++) {
            V[lastv + V.size(0)] = 1.0;
        }
        i = x.size(0);
        for (lastv = 0; lastv < i; lastv++) {
            V[lastv] = x[lastv];
        }
    }
    m = V.size(0);
    minmana = V.size(0);
    if (minmana > 2) {
        minmana = 2;
    }
    if (minmana - 1 >= 0) {
        std::memset(&tau_data[0], 0, static_cast<unsigned int>(minmana) * sizeof(double));
    }
    if (V.size(0) == 0) {
        jpvt[0] = 1;
        jpvt[1] = 2;
    } else {
        double vn1[2];
        double vn2[2];
        double work[2];
        double absxk;
        double scale;
        double t;
        int ix0;
        int ma;
        ma = V.size(0);
        u0 = V.size(0);
        if (u0 > 2) {
            u0 = 2;
        }
        for (lastv = 0; lastv < 2; lastv++) {
            jpvt[lastv] = static_cast<signed char>(lastv + 1);
            work[lastv] = 0.0;
            ix0 = lastv * ma;
            temp = 0.0;
            if (m >= 1) {
                if (m == 1) {
                    temp = std::abs(V[ix0]);
                } else {
                    scale = 3.3121686421112381E-170;
                    kend = ix0 + m;
                    for (minmana = ix0 + 1; minmana <= kend; minmana++) {
                        absxk = std::abs(V[minmana - 1]);
                        if (absxk > scale) {
                            t = scale / absxk;
                            temp = temp * t * t + 1.0;
                            scale = absxk;
                        } else {
                            t = absxk / scale;
                            temp += t * t;
                        }
                    }
                    temp = scale * std::sqrt(temp);
                }
            }
            vn1[lastv] = temp;
            vn2[lastv] = temp;
        }
        for (int b_i{0}; b_i < u0; b_i++) {
            double atmp;
            int ii;
            int ip1;
            int mmi;
            ip1 = b_i + 2;
            ix0 = b_i * ma;
            ii = ix0 + b_i;
            mmi = (m - b_i) - 1;
            minmana = 0;
            if ((2 - b_i > 1) && (vn1[1] > vn1[b_i])) {
                minmana = 1;
            }
            pvt = b_i + minmana;
            if (pvt != b_i) {
                minmana = pvt * ma;
                for (lastv = 0; lastv < m; lastv++) {
                    kend = minmana + lastv;
                    temp = V[kend];
                    i = ix0 + lastv;
                    V[kend] = V[i];
                    V[i] = temp;
                }
                minmana = jpvt[pvt];
                jpvt[pvt] = jpvt[b_i];
                jpvt[b_i] = static_cast<signed char>(minmana);
                vn1[pvt] = vn1[b_i];
                vn2[pvt] = vn2[b_i];
            }
            if (b_i + 1 < m) {
                atmp = V[ii];
                ix0 = ii + 2;
                tau_data[b_i] = 0.0;
                if (mmi + 1 > 0) {
                    temp = 0.0;
                    if (mmi >= 1) {
                        if (mmi == 1) {
                            temp = std::abs(V[ii + 1]);
                        } else {
                            scale = 3.3121686421112381E-170;
                            kend = (ii + mmi) + 1;
                            for (lastv = ix0; lastv <= kend; lastv++) {
                                absxk = std::abs(V[lastv - 1]);
                                if (absxk > scale) {
                                    t = scale / absxk;
                                    temp = temp * t * t + 1.0;
                                    scale = absxk;
                                } else {
                                    t = absxk / scale;
                                    temp += t * t;
                                }
                            }
                            temp = scale * std::sqrt(temp);
                        }
                    }
                    if (temp != 0.0) {
                        scale = rt_hypotd_snf(V[ii], temp);
                        if (V[ii] >= 0.0) {
                            scale = -scale;
                        }
                        if (std::abs(scale) < 1.0020841800044864E-292) {
                            minmana = 0;
                            i = (ii + mmi) + 1;
                            do {
                                minmana++;
                                for (lastv = ix0; lastv <= i; lastv++) {
                                    V[lastv - 1] = 9.9792015476736E+291 * V[lastv - 1];
                                }
                                scale *= 9.9792015476736E+291;
                                atmp *= 9.9792015476736E+291;
                            } while ((std::abs(scale) < 1.0020841800044864E-292) && (minmana < 20));
                            temp = 0.0;
                            if (mmi >= 1) {
                                if (mmi == 1) {
                                    temp = std::abs(V[ii + 1]);
                                } else {
                                    scale = 3.3121686421112381E-170;
                                    for (lastv = ix0; lastv <= i; lastv++) {
                                        absxk = std::abs(V[lastv - 1]);
                                        if (absxk > scale) {
                                            t = scale / absxk;
                                            temp = temp * t * t + 1.0;
                                            scale = absxk;
                                        } else {
                                            t = absxk / scale;
                                            temp += t * t;
                                        }
                                    }
                                    temp = scale * std::sqrt(temp);
                                }
                            }
                            scale = rt_hypotd_snf(atmp, temp);
                            if (atmp >= 0.0) {
                                scale = -scale;
                            }
                            tau_data[b_i] = (scale - atmp) / scale;
                            temp = 1.0 / (atmp - scale);
                            for (lastv = ix0; lastv <= i; lastv++) {
                                V[lastv - 1] = temp * V[lastv - 1];
                            }
                            for (lastv = 0; lastv < minmana; lastv++) {
                                scale *= 1.0020841800044864E-292;
                            }
                            atmp = scale;
                        } else {
                            tau_data[b_i] = (scale - V[ii]) / scale;
                            temp = 1.0 / (V[ii] - scale);
                            i = (ii + mmi) + 1;
                            for (lastv = ix0; lastv <= i; lastv++) {
                                V[lastv - 1] = temp * V[lastv - 1];
                            }
                            atmp = scale;
                        }
                    }
                }
                V[ii] = atmp;
            } else {
                tau_data[b_i] = 0.0;
            }
            if (b_i + 1 < 2) {
                int jA;
                atmp = V[ii];
                V[ii] = 1.0;
                jA = (ii + ma) + 1;
                if (tau_data[0] != 0.0) {
                    lastv = mmi;
                    minmana = ii + mmi;
                    while ((lastv + 1 > 0) && (V[minmana] == 0.0)) {
                        lastv--;
                        minmana--;
                    }
                    kend = 1;
                    pvt = jA;
                    int exitg1;
                    do {
                        exitg1 = 0;
                        if (pvt <= jA + lastv) {
                            if (V[pvt - 1] != 0.0) {
                                exitg1 = 1;
                            } else {
                                pvt++;
                            }
                        } else {
                            kend = 0;
                            exitg1 = 1;
                        }
                    } while (exitg1 == 0);
                } else {
                    lastv = -1;
                    kend = 0;
                }
                if (lastv + 1 > 0) {
                    if (kend != 0) {
                        work[0] = 0.0;
                        minmana = 0;
                        for (ix0 = jA; ma < 0 ? ix0 >= jA : ix0 <= jA; ix0 += ma) {
                            temp = 0.0;
                            i = ix0 + lastv;
                            for (pvt = ix0; pvt <= i; pvt++) {
                                temp += V[pvt - 1] * V[(ii + pvt) - ix0];
                            }
                            work[minmana] += temp;
                            minmana++;
                        }
                    }
                    if (!(-tau_data[0] == 0.0)) {
                        for (pvt = 0; pvt < kend; pvt++) {
                            if (work[0] != 0.0) {
                                temp = work[0] * -tau_data[0];
                                i = lastv + jA;
                                for (minmana = jA; minmana <= i; minmana++) {
                                    V[minmana - 1] = V[minmana - 1] + V[(ii + minmana) - jA] * temp;
                                }
                            }
                            jA += ma;
                        }
                    }
                }
                V[ii] = atmp;
            }
            for (pvt = ip1; pvt < 3; pvt++) {
                minmana = (b_i + ma) + 1;
                if (vn1[1] != 0.0) {
                    temp = std::abs(V[minmana - 1]) / vn1[1];
                    temp = 1.0 - temp * temp;
                    if (temp < 0.0) {
                        temp = 0.0;
                    }
                    scale = vn1[1] / vn2[1];
                    scale = temp * (scale * scale);
                    if (scale <= 1.4901161193847656E-8) {
                        if (b_i + 1 < m) {
                            ix0 = minmana + 1;
                            temp = 0.0;
                            if (mmi >= 1) {
                                if (mmi == 1) {
                                    temp = std::abs(V[minmana]);
                                } else {
                                    scale = 3.3121686421112381E-170;
                                    kend = minmana + mmi;
                                    for (lastv = ix0; lastv <= kend; lastv++) {
                                        absxk = std::abs(V[lastv - 1]);
                                        if (absxk > scale) {
                                            t = scale / absxk;
                                            temp = temp * t * t + 1.0;
                                            scale = absxk;
                                        } else {
                                            t = absxk / scale;
                                            temp += t * t;
                                        }
                                    }
                                    temp = scale * std::sqrt(temp);
                                }
                            }
                            vn1[1] = temp;
                            vn2[1] = temp;
                        } else {
                            vn1[1] = 0.0;
                            vn2[1] = 0.0;
                        }
                    } else {
                        vn1[1] *= std::sqrt(temp);
                    }
                }
            }
        }
    }
    kend = 0;
    minmana = V.size(0);
    if (minmana > 2) {
        minmana = 2;
    }
    if (minmana > 0) {
        for (lastv = 0; lastv < minmana; lastv++) {
            if (V[lastv + V.size(0) * lastv] != 0.0) {
                kend++;
            }
        }
    }
    B.set_size(y.size(0));
    minmana = y.size(0);
    for (i = 0; i < minmana; i++) {
        B[i] = y[i];
    }
    p[0] = 0.0;
    p[1] = 0.0;
    m = V.size(0);
    u0 = V.size(0);
    if (u0 > 2) {
        u0 = 2;
    }
    for (pvt = 0; pvt < u0; pvt++) {
        if (tau_data[pvt] != 0.0) {
            temp = B[pvt];
            i = pvt + 2;
            for (int b_i{i}; b_i <= m; b_i++) {
                temp += V[(b_i + V.size(0) * pvt) - 1] * B[b_i - 1];
            }
            temp *= tau_data[pvt];
            if (temp != 0.0) {
                B[pvt] = B[pvt] - temp;
                for (int b_i{i}; b_i <= m; b_i++) {
                    B[b_i - 1] = B[b_i - 1] - V[(b_i + V.size(0) * pvt) - 1] * temp;
                }
            }
        }
    }
    for (int b_i{0}; b_i < kend; b_i++) {
        p[jpvt[b_i] - 1] = B[b_i];
    }
    for (pvt = kend; pvt >= 1; pvt--) {
        minmana = jpvt[pvt - 1] - 1;
        p[minmana] /= V[(pvt + V.size(0) * (pvt - 1)) - 1];
        for (int b_i{0}; b_i <= pvt - 2; b_i++) {
            p[jpvt[0] - 1] -= p[minmana] * V[V.size(0) * (pvt - 1)];
        }
    }
}

//
// Arguments    : int numerator
//                int denominator
// Return Type  : int
//
} // namespace coder
static int div_s32(int numerator, int denominator)
{
    int quotient;
    if (denominator == 0) {
        if (numerator >= 0) {
            quotient = MAX_int32_T;
        } else {
            quotient = MIN_int32_T;
        }
    } else {
        unsigned int tempAbsQuotient;
        unsigned int u;
        if (numerator < 0) {
            tempAbsQuotient = ~static_cast<unsigned int>(numerator) + 1U;
        } else {
            tempAbsQuotient = static_cast<unsigned int>(numerator);
        }
        if (denominator < 0) {
            u = ~static_cast<unsigned int>(denominator) + 1U;
        } else {
            u = static_cast<unsigned int>(denominator);
        }
        tempAbsQuotient /= u;
        if ((numerator < 0) != (denominator < 0)) {
            quotient = -static_cast<int>(tempAbsQuotient);
        } else {
            quotient = static_cast<int>(tempAbsQuotient);
        }
    }
    return quotient;
}

//
// Arguments    : double u0
//                double u1
// Return Type  : double
//
static double rt_atan2d_snf(double u0, double u1)
{
    double y;
    if (std::isnan(u0) || std::isnan(u1)) {
        y = rtNaN;
    } else if (std::isinf(u0) && std::isinf(u1)) {
        int i;
        int i1;
        if (u0 > 0.0) {
            i = 1;
        } else {
            i = -1;
        }
        if (u1 > 0.0) {
            i1 = 1;
        } else {
            i1 = -1;
        }
        y = std::atan2(static_cast<double>(i), static_cast<double>(i1));
    } else if (u1 == 0.0) {
        if (u0 > 0.0) {
            y = RT_PI / 2.0;
        } else if (u0 < 0.0) {
            y = -(RT_PI / 2.0);
        } else {
            y = 0.0;
        }
    } else {
        y = std::atan2(u0, u1);
    }
    return y;
}

//
// Arguments    : double u0
//                double u1
// Return Type  : double
//
static double rt_hypotd_snf(double u0, double u1)
{
    double a;
    double b;
    double y;
    a = std::abs(u0);
    b = std::abs(u1);
    if (a < b) {
        a /= b;
        y = b * std::sqrt(a * a + 1.0);
    } else if (a > b) {
        b /= a;
        y = a * std::sqrt(b * b + 1.0);
    } else if (std::isnan(b)) {
        y = rtNaN;
    } else {
        y = a * 1.4142135623730951;
    }
    return y;
}

//
// Arguments    : double u0
//                double u1
// Return Type  : double
//
static double rt_remd_snf(double u0, double u1)
{
    double y;
    if (std::isnan(u0) || std::isnan(u1) || std::isinf(u0)) {
        y = rtNaN;
    } else if (std::isinf(u1)) {
        y = u0;
    } else if ((u1 != 0.0) && (u1 != std::trunc(u1))) {
        double q;
        q = std::abs(u0 / u1);
        if (!(std::abs(q - std::floor(q + 0.5)) > DBL_EPSILON * q)) {
            y = 0.0 * u0;
        } else {
            y = std::fmod(u0, u1);
        }
    } else {
        y = std::fmod(u0, u1);
    }
    return y;
}

//
// Arguments    : void
// Return Type  : void
//
CSIPreprocessor::CSIPreprocessor()
{
    SD_.pd = &pd_;
    InterpolateCSIAndRemoveCSDAndAutoUnpermutation_init(this);
}

//
// Arguments    : void
// Return Type  : void
//
CSIPreprocessor::~CSIPreprocessor() = default;

//
// Arguments    : coder::array<creal_T, 4U> &CSI
//                const coder::array<short, 1U> &subcarrierIndex_int16
//                double numSTS
//                double numESS
//                double numRx
//                double numCSI
//                double format
//                double CBW
//                boolean_T removeCSD
//                boolean_T unpermute
//                coder::array<creal_T, 4U> &resultCSI
//                coder::array<double, 4U> &resultMag
//                coder::array<double, 4U> &resultPhase
//                coder::array<short, 1U> &interpedIndex_int16
//                coder::array<double, 3U> &phaseSlope
//                coder::array<double, 3U> &phaseIntercept
//                unsigned char *perm
//                double *permCoef
// Return Type  : void
//
void CSIPreprocessor::InterpolateCSIAndRemoveCSDAndAutoUnpermutation(
    coder::array<creal_T, 4U> &CSI, const coder::array<short, 1U> &subcarrierIndex_int16, double numSTS, double,
    double numRx, double numCSI, double format, double CBW, boolean_T removeCSD, boolean_T unpermute,
    coder::array<creal_T, 4U> &resultCSI, coder::array<double, 4U> &resultMag, coder::array<double, 4U> &resultPhase,
    coder::array<short, 1U> &interpedIndex_int16, coder::array<double, 3U> &phaseSlope,
    coder::array<double, 3U> &phaseIntercept, unsigned char *perm, double *permCoef)
{
    coder::array<creal_T, 4U> phaseShiftAll;
    coder::array<creal_T, 2U> phaseShiftPerRx;
    coder::array<double, 4U> b;
    coder::array<double, 4U> b_x;
    coder::array<double, 4U> c_resultPhase;
    coder::array<double, 4U> pivotPhase;
    coder::array<double, 4U> rawMag;
    coder::array<double, 4U> rawPhase;
    coder::array<double, 3U> b_r;
    coder::array<double, 3U> b_rawPhase;
    coder::array<double, 2U> Vq;
    coder::array<double, 2U> n1;
    coder::array<double, 2U> n2;
    coder::array<double, 2U> nmag;
    coder::array<double, 2U> y;
    coder::array<double, 1U> X;
    coder::array<double, 1U> Y;
    coder::array<double, 1U> b_X;
    coder::array<double, 1U> b_Y;
    coder::array<double, 1U> b_resultPhase;
    coder::array<double, 1U> c_X;
    coder::array<double, 1U> c_Y;
    coder::array<double, 1U> interpedIndex;
    coder::array<int, 1U> r3;
    coder::array<int, 1U> r4;
    coder::array<short, 1U> subcarrierIndex;
    coder::array<boolean_T, 4U> r1;
    coder::array<boolean_T, 4U> r2;
    coder::array<boolean_T, 1U> c_x;
    coder::array<boolean_T, 1U> positiveInput;
    double cumsum_dp_corr;
    double dp;
    double nfft;
    double r;
    double rp2;
    int b_exponent;
    int c_exponent;
    int d_exponent;
    int e_exponent;
    int exponent;
    int f_exponent;
    int g_exponent;
    int i;
    int i1;
    int ibmat;
    int ibtile;
    int ii_data;
    int k;
    int nblocks;
    int nrows;
    int ntilecols;
    int nx;
    int nycols;
    int nyrows;
    unsigned int pivotIndex_data;
    int xtmp;
    boolean_T exitg1;
    boolean_T nanflag;
    if (!pd_.lmag_not_empty) {
        pd_.lmag.set_size(0, 2);
    }
    subcarrierIndex.set_size(subcarrierIndex_int16.size(0));
    nblocks = subcarrierIndex_int16.size(0);
    for (i = 0; i < nblocks; i++) {
        subcarrierIndex[i] = subcarrierIndex_int16[i];
    }
    if (removeCSD) {
        creal_T b_csdInSamples_data[8];
        double csdInSamples_data[8];
        switch (static_cast<int>(numSTS)) {
        case 1:
            nrows = 1;
            csdInSamples_data[0] = 0.0;
            break;
        case 2:
            nrows = 2;
            csdInSamples_data[0] = -0.0;
            csdInSamples_data[1] = -400.0;
            break;
        case 3:
            nrows = 3;
            csdInSamples_data[0] = 0.0;
            csdInSamples_data[1] = -400.0;
            csdInSamples_data[2] = -200.0;
            break;
        case 4:
            nrows = 4;
            csdInSamples_data[0] = 0.0;
            csdInSamples_data[1] = -400.0;
            csdInSamples_data[2] = -200.0;
            csdInSamples_data[3] = -600.0;
            break;
        case 5:
            nrows = 5;
            csdInSamples_data[0] = 0.0;
            csdInSamples_data[1] = -400.0;
            csdInSamples_data[2] = -200.0;
            csdInSamples_data[3] = -600.0;
            csdInSamples_data[4] = -350.0;
            break;
        case 6:
            nrows = 6;
            csdInSamples_data[0] = 0.0;
            csdInSamples_data[1] = -400.0;
            csdInSamples_data[2] = -200.0;
            csdInSamples_data[3] = -600.0;
            csdInSamples_data[4] = -350.0;
            csdInSamples_data[5] = -650.0;
            break;
        case 7:
            nrows = 7;
            csdInSamples_data[0] = 0.0;
            csdInSamples_data[1] = -400.0;
            csdInSamples_data[2] = -200.0;
            csdInSamples_data[3] = -600.0;
            csdInSamples_data[4] = -350.0;
            csdInSamples_data[5] = -650.0;
            csdInSamples_data[6] = -100.0;
            break;
        case 8:
            nrows = 8;
            csdInSamples_data[0] = 0.0;
            csdInSamples_data[1] = -400.0;
            csdInSamples_data[2] = -200.0;
            csdInSamples_data[3] = -600.0;
            csdInSamples_data[4] = -350.0;
            csdInSamples_data[5] = -650.0;
            csdInSamples_data[6] = -100.0;
            csdInSamples_data[7] = -750.0;
            break;
        default:
            nrows = 1;
            csdInSamples_data[0] = 0.0;
            break;
        }
        nblocks = nrows - 1;
        for (i = 0; i <= nblocks; i++) {
            csdInSamples_data[i] = csdInSamples_data[i] * CBW * 0.001;
        }
        nfft = 64.0 * (CBW / 20.0);
        if (format >= 3.0) {
            nfft *= 4.0;
        }
        for (i = 0; i < nrows; i++) {
            b_csdInSamples_data[i].re = csdInSamples_data[i];
            b_csdInSamples_data[i].im = 0.0;
        }
        phaseShiftPerRx.set_size(subcarrierIndex.size(0), nrows);
        nblocks = subcarrierIndex.size(0);
        for (i = 0; i < nrows; i++) {
            for (i1 = 0; i1 < nblocks; i1++) {
                xtmp = subcarrierIndex[i1];
                dp = static_cast<double>(xtmp) * 6.2831853071795862;
                cumsum_dp_corr = b_csdInSamples_data[i].im;
                rp2 = b_csdInSamples_data[i].re;
                r = 0.0 * rp2 - dp * cumsum_dp_corr;
                dp = 0.0 * cumsum_dp_corr + dp * rp2;
                if (dp == 0.0) {
                    phaseShiftPerRx[i1 + phaseShiftPerRx.size(0) * i].re = r / nfft;
                    phaseShiftPerRx[i1 + phaseShiftPerRx.size(0) * i].im = 0.0;
                } else if (r == 0.0) {
                    phaseShiftPerRx[i1 + phaseShiftPerRx.size(0) * i].re = 0.0;
                    phaseShiftPerRx[i1 + phaseShiftPerRx.size(0) * i].im = dp / nfft;
                } else {
                    phaseShiftPerRx[i1 + phaseShiftPerRx.size(0) * i].re = r / nfft;
                    phaseShiftPerRx[i1 + phaseShiftPerRx.size(0) * i].im = dp / nfft;
                }
            }
        }
        nx = phaseShiftPerRx.size(0) * phaseShiftPerRx.size(1);
        for (k = 0; k < nx; k++) {
            if (phaseShiftPerRx[k].im == 0.0) {
                phaseShiftPerRx[k].re = std::exp(phaseShiftPerRx[k].re);
                phaseShiftPerRx[k].im = 0.0;
            } else if (std::isinf(phaseShiftPerRx[k].im) && std::isinf(phaseShiftPerRx[k].re) &&
                       (phaseShiftPerRx[k].re < 0.0)) {
                phaseShiftPerRx[k].re = 0.0;
                phaseShiftPerRx[k].im = 0.0;
            } else {
                r = std::exp(phaseShiftPerRx[k].re / 2.0);
                dp = phaseShiftPerRx[k].im;
                phaseShiftPerRx[k].re = r * (r * std::cos(phaseShiftPerRx[k].im));
                phaseShiftPerRx[k].im = r * (r * std::sin(dp));
            }
        }
        phaseShiftAll.set_size(phaseShiftPerRx.size(0), phaseShiftPerRx.size(1), static_cast<int>(numRx),
                               static_cast<int>(numCSI));
        nrows = phaseShiftPerRx.size(0);
        nx = phaseShiftPerRx.size(1);
        ntilecols = static_cast<int>(numRx) * static_cast<int>(numCSI);
        for (xtmp = 0; xtmp < ntilecols; xtmp++) {
            ibtile = xtmp * (nrows * nx) - 1;
            for (nyrows = 0; nyrows < nx; nyrows++) {
                nblocks = nyrows * nrows;
                ibmat = ibtile + nblocks;
                for (k = 0; k < nrows; k++) {
                    phaseShiftAll[(ibmat + k) + 1] = phaseShiftPerRx[nblocks + k];
                }
            }
        }
        nblocks = CSI.size(0) * CSI.size(1) * CSI.size(2) * CSI.size(3);
        for (i = 0; i < nblocks; i++) {
            cumsum_dp_corr = CSI[i].re;
            rp2 = phaseShiftAll[i].im;
            nfft = CSI[i].im;
            dp = phaseShiftAll[i].re;
            CSI[i].re = cumsum_dp_corr * dp - nfft * rp2;
            CSI[i].im = cumsum_dp_corr * rp2 + nfft * dp;
        }
    }
    if (subcarrierIndex[subcarrierIndex.size(0) - 1] < subcarrierIndex[0]) {
        y.set_size(1, 0);
    } else {
        y.set_size(1, (subcarrierIndex[subcarrierIndex.size(0) - 1] - subcarrierIndex[0]) + 1);
        nblocks = subcarrierIndex[subcarrierIndex.size(0) - 1] - subcarrierIndex[0];
        for (i = 0; i <= nblocks; i++) {
            y[i] = static_cast<short>(subcarrierIndex[0] + static_cast<short>(i));
        }
    }
    Y.set_size(subcarrierIndex.size(0));
    nblocks = subcarrierIndex.size(0);
    for (i = 0; i < nblocks; i++) {
        Y[i] = subcarrierIndex[i];
    }
    X.set_size(subcarrierIndex.size(0));
    nblocks = subcarrierIndex.size(0);
    for (i = 0; i < nblocks; i++) {
        X[i] = subcarrierIndex[i];
    }
    nx = subcarrierIndex.size(0) - 1;
    nblocks = y.size(1);
    Vq.set_size(1, y.size(1));
    for (i = 0; i < nblocks; i++) {
        Vq[i] = rtNaN;
    }
    if (y.size(1) != 0) {
        if (subcarrierIndex[1] < subcarrierIndex[0]) {
            i = subcarrierIndex.size(0) >> 1;
            for (ntilecols = 0; ntilecols < i; ntilecols++) {
                xtmp = static_cast<int>(X[ntilecols]);
                nblocks = nx - ntilecols;
                X[ntilecols] = X[nblocks];
                X[nblocks] = xtmp;
            }
            if (subcarrierIndex.size(0) > 1) {
                ntilecols = subcarrierIndex.size(0) - 1;
                ibtile = subcarrierIndex.size(0) >> 1;
                for (k = 0; k < ibtile; k++) {
                    nrows = static_cast<int>(Y[k]);
                    nx = ntilecols - k;
                    Y[k] = Y[nx];
                    Y[nx] = nrows;
                }
            }
        }
        ibtile = y.size(1);
        for (k = 0; k < ibtile; k++) {
            dp = y[k];
            if ((!(dp > X[X.size(0) - 1])) && (!(dp < X[0]))) {
                nrows = X.size(0);
                ibmat = 1;
                nx = 2;
                while (nrows > nx) {
                    ntilecols = (ibmat >> 1) + (nrows >> 1);
                    if (((ibmat & 1) == 1) && ((nrows & 1) == 1)) {
                        ntilecols++;
                    }
                    if (y[k] >= X[ntilecols - 1]) {
                        ibmat = ntilecols;
                        nx = ntilecols + 1;
                    } else {
                        nrows = ntilecols;
                    }
                }
                nfft = X[ibmat - 1];
                r = (y[k] - nfft) / (X[ibmat] - nfft);
                if (r == 0.0) {
                    Vq[k] = Y[ibmat - 1];
                } else if (r == 1.0) {
                    Vq[k] = Y[ibmat];
                } else if (Y[ibmat - 1] == Y[ibmat]) {
                    Vq[k] = Y[ibmat - 1];
                } else {
                    Vq[k] = (1.0 - r) * Y[ibmat - 1] + r * Y[ibmat];
                }
            }
        }
    }
    interpedIndex.set_size(Vq.size(1));
    nblocks = Vq.size(1);
    for (i = 0; i < nblocks; i++) {
        interpedIndex[i] = Vq[i];
    }
    nx = CSI.size(0) * CSI.size(1) * CSI.size(2) * CSI.size(3);
    rawPhase.set_size(CSI.size(0), CSI.size(1), CSI.size(2), CSI.size(3));
    for (k = 0; k < nx; k++) {
        rawPhase[k] = rt_atan2d_snf(CSI[k].im, CSI[k].re);
    }
    nx = CSI.size(0) * CSI.size(1) * CSI.size(2) * CSI.size(3);
    rawMag.set_size(CSI.size(0), CSI.size(1), CSI.size(2), CSI.size(3));
    for (k = 0; k < nx; k++) {
        rawMag[k] = rt_hypotd_snf(CSI[k].re, CSI[k].im);
    }
    *perm = 0U;
    *permCoef = 0.0;
    if (unpermute && (rawMag.size(2) >= 2)) {
        int emptyDimValue[2];
        boolean_T x[2];
        nrows = static_cast<int>(std::round(static_cast<double>(rawMag.size(0)) * 0.92));
        if (static_cast<double>(nrows) / 3.0 < 150.0) {
            ntilecols = 3;
        } else {
            ntilecols = static_cast<int>(std::floor(static_cast<double>(nrows) / 150.0));
        }
        if ((ntilecols == 0) || ((ntilecols > 0) && (nrows < 1))) {
            ntilecols = 1;
            i = -1;
        } else {
            i = nrows - 1;
        }
        nblocks = div_s32(i, ntilecols);
        b_r.set_size(nblocks + 1, 1, 2);
        nycols = static_cast<int>(static_cast<unsigned int>((div_s32(i, ntilecols) + 1) << 1) >> 1);
        nmag.set_size(nycols, 2);
        emptyDimValue[0] = nycols;
        emptyDimValue[1] = 2;
        for (i = 0; i < 2; i++) {
            for (i1 = 0; i1 <= nblocks; i1++) {
                b_r[i1 + b_r.size(0) * i] = rawMag[ntilecols * i1 + rawMag.size(0) * rawMag.size(1) * i];
            }
            for (i1 = 0; i1 < nycols; i1++) {
                nmag[i1 + nmag.size(0) * i] = rawMag[ntilecols * i1 + rawMag.size(0) * rawMag.size(1) * i];
            }
            x[i] = (pd_.lmag.size(i) == emptyDimValue[i]);
        }
        nanflag = true;
        k = 0;
        exitg1 = false;
        while ((!exitg1) && (k < 2)) {
            if (!x[k]) {
                nanflag = false;
                exitg1 = true;
            } else {
                k++;
            }
        }
        if (nanflag) {
            double D;
            double b_D;
            double rn1;
            double rn2;
            double rp1;
            boolean_T b_nanflag;
            //  the same pattern
            if (nycols == 0) {
                dp = 0.0;
            } else {
                if (nycols <= 1024) {
                    nx = nycols;
                    nrows = 0;
                    nblocks = 1;
                } else {
                    nx = 1024;
                    nblocks = nycols / 1024;
                    nrows = nycols - (nblocks << 10);
                    if (nrows > 0) {
                        nblocks++;
                    } else {
                        nrows = 1024;
                    }
                }
                dp = rawMag[0];
                for (k = 2; k <= nx; k++) {
                    dp += rawMag[ntilecols * (k - 1)];
                }
                for (ibtile = 2; ibtile <= nblocks; ibtile++) {
                    ibmat = (ibtile - 1) << 10;
                    nfft = rawMag[ntilecols * ibmat];
                    if (ibtile == nblocks) {
                        xtmp = nrows;
                    } else {
                        xtmp = 1024;
                    }
                    for (k = 2; k <= xtmp; k++) {
                        nfft += rawMag[ntilecols * ((ibmat + k) - 1)];
                    }
                    dp += nfft;
                }
            }
            rn1 = dp / static_cast<double>(nycols);
            if (nycols == 0) {
                dp = 0.0;
            } else {
                if (nycols <= 1024) {
                    nx = nycols;
                    nrows = 0;
                    nblocks = 1;
                } else {
                    nx = 1024;
                    nblocks = nycols / 1024;
                    nrows = nycols - (nblocks << 10);
                    if (nrows > 0) {
                        nblocks++;
                    } else {
                        nrows = 1024;
                    }
                }
                dp = rawMag[rawMag.size(0) * rawMag.size(1)];
                for (k = 2; k <= nx; k++) {
                    dp += rawMag[ntilecols * (k - 1) + rawMag.size(0) * rawMag.size(1)];
                }
                for (ibtile = 2; ibtile <= nblocks; ibtile++) {
                    ibmat = (ibtile - 1) << 10;
                    nfft = rawMag[ntilecols * ibmat + rawMag.size(0) * rawMag.size(1)];
                    if (ibtile == nblocks) {
                        xtmp = nrows;
                    } else {
                        xtmp = 1024;
                    }
                    for (k = 2; k <= xtmp; k++) {
                        nfft += rawMag[ntilecols * ((ibmat + k) - 1) + rawMag.size(0) * rawMag.size(1)];
                    }
                    dp += nfft;
                }
            }
            rn2 = dp / static_cast<double>(nycols);
            if (pd_.lmag.size(0) == 0) {
                dp = 0.0;
            } else {
                if (pd_.lmag.size(0) <= 1024) {
                    nx = pd_.lmag.size(0);
                    nrows = 0;
                    nblocks = 1;
                } else {
                    nx = 1024;
                    nblocks = static_cast<int>(static_cast<unsigned int>(pd_.lmag.size(0)) >> 10);
                    nrows = pd_.lmag.size(0) - (nblocks << 10);
                    if (nrows > 0) {
                        nblocks++;
                    } else {
                        nrows = 1024;
                    }
                }
                dp = pd_.lmag[0];
                for (k = 2; k <= nx; k++) {
                    dp += pd_.lmag[k - 1];
                }
                for (ibtile = 2; ibtile <= nblocks; ibtile++) {
                    ibmat = (ibtile - 1) << 10;
                    nfft = pd_.lmag[ibmat];
                    if (ibtile == nblocks) {
                        xtmp = nrows;
                    } else {
                        xtmp = 1024;
                    }
                    for (k = 2; k <= xtmp; k++) {
                        nfft += pd_.lmag[(ibmat + k) - 1];
                    }
                    dp += nfft;
                }
            }
            rp1 = dp / static_cast<double>(pd_.lmag.size(0));
            if (pd_.lmag.size(0) == 0) {
                dp = 0.0;
            } else {
                if (pd_.lmag.size(0) <= 1024) {
                    nx = pd_.lmag.size(0);
                    nrows = 0;
                    nblocks = 1;
                } else {
                    nx = 1024;
                    nblocks = static_cast<int>(static_cast<unsigned int>(pd_.lmag.size(0)) >> 10);
                    nrows = pd_.lmag.size(0) - (nblocks << 10);
                    if (nrows > 0) {
                        nblocks++;
                    } else {
                        nrows = 1024;
                    }
                }
                dp = pd_.lmag[pd_.lmag.size(0)];
                for (k = 2; k <= nx; k++) {
                    dp += pd_.lmag[(k + pd_.lmag.size(0)) - 1];
                }
                for (ibtile = 2; ibtile <= nblocks; ibtile++) {
                    ibmat = (ibtile - 1) << 10;
                    nfft = pd_.lmag[ibmat + pd_.lmag.size(0)];
                    if (ibtile == nblocks) {
                        xtmp = nrows;
                    } else {
                        xtmp = 1024;
                    }
                    for (k = 2; k <= xtmp; k++) {
                        nfft += pd_.lmag[((ibmat + k) + pd_.lmag.size(0)) - 1];
                    }
                    dp += nfft;
                }
            }
            rp2 = dp / static_cast<double>(pd_.lmag.size(0));
            n1.set_size(1, nycols);
            n2.set_size(1, nycols);
            for (i = 0; i < nycols; i++) {
                n1[i] = rawMag[ntilecols * i] - rn1;
                n2[i] = rawMag[ntilecols * i + rawMag.size(0) * rawMag.size(1)] - rn2;
            }
            ibmat = n1.size(1) - 1;
            cumsum_dp_corr = rtNaN;
            b_X.set_size(n1.size(1));
            nblocks = n1.size(1);
            for (i = 0; i < nblocks; i++) {
                b_X[i] = n1[i];
            }
            X.set_size(b_X.size(0));
            nblocks = b_X.size(0);
            for (i = 0; i < nblocks; i++) {
                X[i] = b_X[i];
            }
            b_Y.set_size(pd_.lmag.size(0));
            nblocks = pd_.lmag.size(0);
            for (i = 0; i < nblocks; i++) {
                b_Y[i] = pd_.lmag[i] - rp1;
            }
            Y.set_size(b_Y.size(0));
            nblocks = b_Y.size(0);
            for (i = 0; i < nblocks; i++) {
                Y[i] = b_Y[i];
            }
            nx = b_X.size(0);
            r = 0.0;
            nfft = 0.0;
            for (nrows = 0; nrows < nx; nrows++) {
                nfft += b_X[nrows] * b_X[nrows];
            }
            nfft = std::sqrt(nfft);
            nblocks = b_X.size(0);
            for (i = 0; i < nblocks; i++) {
                X[i] = b_X[i] / nfft;
            }
            if ((!(nfft < rtInf)) && (nfft > 0.0)) {
                r = rtInf;
            }
            if ((!std::isinf(r)) && (!(r < 4.4501477170144028E-308))) {
                std::frexp(r, &b_exponent);
            }
            nx = b_Y.size(0);
            r = 0.0;
            nfft = 0.0;
            for (nrows = 0; nrows < nx; nrows++) {
                nfft += b_Y[nrows] * b_Y[nrows];
            }
            nfft = std::sqrt(nfft);
            nblocks = b_Y.size(0);
            for (i = 0; i < nblocks; i++) {
                Y[i] = b_Y[i] / nfft;
            }
            if ((!(nfft < rtInf)) && (nfft > 0.0)) {
                r = rtInf;
            }
            if ((!std::isinf(r)) && (!(r < 4.4501477170144028E-308))) {
                std::frexp(r, &ii_data);
            }
            nanflag = false;
            nrows = 0;
            exitg1 = false;
            while ((!exitg1) && (nrows <= ibmat)) {
                if (std::isnan(X[nrows])) {
                    nanflag = true;
                    exitg1 = true;
                } else {
                    nrows++;
                }
            }
            b_nanflag = false;
            nrows = 0;
            exitg1 = false;
            while ((!exitg1) && (nrows <= ibmat)) {
                if (std::isnan(Y[nrows])) {
                    b_nanflag = true;
                    exitg1 = true;
                } else {
                    nrows++;
                }
            }
            if ((!b_nanflag) && (!nanflag)) {
                nfft = 0.0;
                i = n1.size(1);
                for (nrows = 0; nrows < i; nrows++) {
                    nfft += X[nrows] * Y[nrows];
                }
                cumsum_dp_corr = (1.0 - nfft) * static_cast<double>(nfft < 1.0);
            }
            ibmat = n2.size(1) - 1;
            D = rtNaN;
            c_X.set_size(n2.size(1));
            nblocks = n2.size(1);
            for (i = 0; i < nblocks; i++) {
                c_X[i] = n2[i];
            }
            X.set_size(c_X.size(0));
            nblocks = c_X.size(0);
            for (i = 0; i < nblocks; i++) {
                X[i] = c_X[i];
            }
            c_Y.set_size(pd_.lmag.size(0));
            nblocks = pd_.lmag.size(0);
            for (i = 0; i < nblocks; i++) {
                c_Y[i] = pd_.lmag[i + pd_.lmag.size(0)] - rp2;
            }
            Y.set_size(c_Y.size(0));
            nblocks = c_Y.size(0);
            for (i = 0; i < nblocks; i++) {
                Y[i] = c_Y[i];
            }
            nx = c_X.size(0);
            r = 0.0;
            nfft = 0.0;
            for (nrows = 0; nrows < nx; nrows++) {
                nfft += c_X[nrows] * c_X[nrows];
            }
            nfft = std::sqrt(nfft);
            nblocks = c_X.size(0);
            for (i = 0; i < nblocks; i++) {
                X[i] = c_X[i] / nfft;
            }
            if ((!(nfft < rtInf)) && (nfft > 0.0)) {
                r = rtInf;
            }
            if ((!std::isinf(r)) && (!(r < 4.4501477170144028E-308))) {
                std::frexp(r, &exponent);
            }
            nx = c_Y.size(0);
            r = 0.0;
            nfft = 0.0;
            for (nrows = 0; nrows < nx; nrows++) {
                nfft += c_Y[nrows] * c_Y[nrows];
            }
            nfft = std::sqrt(nfft);
            nblocks = c_Y.size(0);
            for (i = 0; i < nblocks; i++) {
                Y[i] = c_Y[i] / nfft;
            }
            if ((!(nfft < rtInf)) && (nfft > 0.0)) {
                r = rtInf;
            }
            if ((!std::isinf(r)) && (!(r < 4.4501477170144028E-308))) {
                std::frexp(r, &c_exponent);
            }
            nanflag = false;
            nrows = 0;
            exitg1 = false;
            while ((!exitg1) && (nrows <= ibmat)) {
                if (std::isnan(X[nrows])) {
                    nanflag = true;
                    exitg1 = true;
                } else {
                    nrows++;
                }
            }
            b_nanflag = false;
            nrows = 0;
            exitg1 = false;
            while ((!exitg1) && (nrows <= ibmat)) {
                if (std::isnan(Y[nrows])) {
                    b_nanflag = true;
                    exitg1 = true;
                } else {
                    nrows++;
                }
            }
            if ((!b_nanflag) && (!nanflag)) {
                nfft = 0.0;
                i = n2.size(1);
                for (nrows = 0; nrows < i; nrows++) {
                    nfft += X[nrows] * Y[nrows];
                }
                D = (1.0 - nfft) * static_cast<double>(nfft < 1.0);
            }
            ibmat = n1.size(1) - 1;
            b_D = rtNaN;
            nx = b_X.size(0);
            r = 0.0;
            nfft = 0.0;
            for (nrows = 0; nrows < nx; nrows++) {
                nfft += b_X[nrows] * b_X[nrows];
            }
            nfft = std::sqrt(nfft);
            X.set_size(b_X.size(0));
            nblocks = b_X.size(0);
            for (i = 0; i < nblocks; i++) {
                X[i] = b_X[i] / nfft;
            }
            nblocks = X.size(0);
            for (i = 0; i < nblocks; i++) {
                b_X[i] = X[i];
            }
            if ((!(nfft < rtInf)) && (nfft > 0.0)) {
                r = rtInf;
            }
            if ((!std::isinf(r)) && (!(r < 4.4501477170144028E-308))) {
                std::frexp(r, &d_exponent);
            }
            nx = c_Y.size(0);
            r = 0.0;
            nfft = 0.0;
            for (nrows = 0; nrows < nx; nrows++) {
                nfft += c_Y[nrows] * c_Y[nrows];
            }
            nfft = std::sqrt(nfft);
            Y.set_size(c_Y.size(0));
            nblocks = c_Y.size(0);
            for (i = 0; i < nblocks; i++) {
                Y[i] = c_Y[i] / nfft;
            }
            nblocks = Y.size(0);
            for (i = 0; i < nblocks; i++) {
                c_Y[i] = Y[i];
            }
            if ((!(nfft < rtInf)) && (nfft > 0.0)) {
                r = rtInf;
            }
            if ((!std::isinf(r)) && (!(r < 4.4501477170144028E-308))) {
                std::frexp(r, &e_exponent);
            }
            nanflag = false;
            nrows = 0;
            exitg1 = false;
            while ((!exitg1) && (nrows <= ibmat)) {
                if (std::isnan(b_X[nrows])) {
                    nanflag = true;
                    exitg1 = true;
                } else {
                    nrows++;
                }
            }
            b_nanflag = false;
            nrows = 0;
            exitg1 = false;
            while ((!exitg1) && (nrows <= ibmat)) {
                if (std::isnan(c_Y[nrows])) {
                    b_nanflag = true;
                    exitg1 = true;
                } else {
                    nrows++;
                }
            }
            if ((!b_nanflag) && (!nanflag)) {
                nfft = 0.0;
                i = n1.size(1);
                for (nrows = 0; nrows < i; nrows++) {
                    nfft += b_X[nrows] * c_Y[nrows];
                }
                b_D = (1.0 - nfft) * static_cast<double>(nfft < 1.0);
            }
            ibmat = n2.size(1) - 1;
            dp = rtNaN;
            nx = c_X.size(0);
            r = 0.0;
            nfft = 0.0;
            for (nrows = 0; nrows < nx; nrows++) {
                nfft += c_X[nrows] * c_X[nrows];
            }
            nfft = std::sqrt(nfft);
            X.set_size(c_X.size(0));
            nblocks = c_X.size(0);
            for (i = 0; i < nblocks; i++) {
                X[i] = c_X[i] / nfft;
            }
            nblocks = X.size(0);
            for (i = 0; i < nblocks; i++) {
                c_X[i] = X[i];
            }
            if ((!(nfft < rtInf)) && (nfft > 0.0)) {
                r = rtInf;
            }
            if ((!std::isinf(r)) && (!(r < 4.4501477170144028E-308))) {
                std::frexp(r, &f_exponent);
            }
            nx = b_Y.size(0);
            r = 0.0;
            nfft = 0.0;
            for (nrows = 0; nrows < nx; nrows++) {
                nfft += b_Y[nrows] * b_Y[nrows];
            }
            nfft = std::sqrt(nfft);
            Y.set_size(b_Y.size(0));
            nblocks = b_Y.size(0);
            for (i = 0; i < nblocks; i++) {
                Y[i] = b_Y[i] / nfft;
            }
            nblocks = Y.size(0);
            for (i = 0; i < nblocks; i++) {
                b_Y[i] = Y[i];
            }
            if ((!(nfft < rtInf)) && (nfft > 0.0)) {
                r = rtInf;
            }
            if ((!std::isinf(r)) && (!(r < 4.4501477170144028E-308))) {
                std::frexp(r, &g_exponent);
            }
            nanflag = false;
            nrows = 0;
            exitg1 = false;
            while ((!exitg1) && (nrows <= ibmat)) {
                if (std::isnan(c_X[nrows])) {
                    nanflag = true;
                    exitg1 = true;
                } else {
                    nrows++;
                }
            }
            b_nanflag = false;
            nrows = 0;
            exitg1 = false;
            while ((!exitg1) && (nrows <= ibmat)) {
                if (std::isnan(b_Y[nrows])) {
                    b_nanflag = true;
                    exitg1 = true;
                } else {
                    nrows++;
                }
            }
            if ((!b_nanflag) && (!nanflag)) {
                nfft = 0.0;
                i = n2.size(1);
                for (nrows = 0; nrows < i; nrows++) {
                    nfft += c_X[nrows] * b_Y[nrows];
                }
                dp = (1.0 - nfft) * static_cast<double>(nfft < 1.0);
            }
            *permCoef = permLogic(CBW, format, cumsum_dp_corr + 1.0E-9, b_D + 1.0E-9, dp + 1.0E-9, D + 1.0E-9, rn1, rn2,
                                  rp1, rp2);
            if (*permCoef < 0.0) {
                ntilecols = rawPhase.size(0);
                ibtile = rawPhase.size(1);
                nx = rawPhase.size(0);
                ibmat = rawPhase.size(1);
                nrows = rawPhase.size(0);
                xtmp = rawPhase.size(1);
                b_rawPhase.set_size(rawPhase.size(0), rawPhase.size(1), 2);
                for (i = 0; i < 2; i++) {
                    for (i1 = 0; i1 < ibmat; i1++) {
                        for (nyrows = 0; nyrows < nx; nyrows++) {
                            b_rawPhase[(nyrows + b_rawPhase.size(0) * i1) +
                                       b_rawPhase.size(0) * b_rawPhase.size(1) * i] =
                                rawPhase[(nyrows + ntilecols * i1) + ntilecols * ibtile * (1 - i)];
                        }
                    }
                }
                rawPhase.set_size(nrows, xtmp, 2, 1);
                nrows = (nrows * xtmp) << 1;
                for (i = 0; i < nrows; i++) {
                    rawPhase[i] = b_rawPhase[i];
                }
                ntilecols = rawMag.size(0);
                ibtile = rawMag.size(1);
                nx = rawMag.size(0);
                ibmat = rawMag.size(1);
                nrows = rawMag.size(0);
                xtmp = rawMag.size(1);
                b_rawPhase.set_size(rawMag.size(0), rawMag.size(1), 2);
                for (i = 0; i < 2; i++) {
                    for (i1 = 0; i1 < ibmat; i1++) {
                        for (nyrows = 0; nyrows < nx; nyrows++) {
                            b_rawPhase[(nyrows + b_rawPhase.size(0) * i1) +
                                       b_rawPhase.size(0) * b_rawPhase.size(1) * i] =
                                rawMag[(nyrows + ntilecols * i1) + ntilecols * ibtile * (1 - i)];
                        }
                    }
                }
                rawMag.set_size(nrows, xtmp, 2, 1);
                nrows = (nrows * xtmp) << 1;
                for (i = 0; i < nrows; i++) {
                    rawMag[i] = b_rawPhase[i];
                }
                nmag.set_size(nycols, 2);
                for (i = 0; i < 2; i++) {
                    for (i1 = 0; i1 < nycols; i1++) {
                        nmag[i1 + nmag.size(0) * i] = b_r[i1 + b_r.size(0) * (1 - i)];
                    }
                }
                *perm = 1U;
            }
        }
        pd_.lmag.set_size(nmag.size(0), 2);
        nblocks = nmag.size(0) * 2;
        for (i = 0; i < nblocks; i++) {
            pd_.lmag[i] = nmag[i];
        }
        pd_.lmag_not_empty = (pd_.lmag.size(0) != 0);
    }
    nx = subcarrierIndex.size(0);
    Y.set_size(subcarrierIndex.size(0));
    for (k = 0; k < nx; k++) {
        Y[k] = std::abs(static_cast<double>(subcarrierIndex[k]));
    }
    nx = subcarrierIndex.size(0);
    X.set_size(subcarrierIndex.size(0));
    for (k = 0; k < nx; k++) {
        X[k] = std::abs(static_cast<double>(subcarrierIndex[k]));
    }
    nrows = X.size(0);
    if (X.size(0) <= 2) {
        if (X.size(0) == 1) {
            nx = static_cast<int>(X[0]);
        } else if (X[0] > X[X.size(0) - 1]) {
            nx = static_cast<int>(X[X.size(0) - 1]);
        } else {
            nx = static_cast<int>(X[0]);
        }
    } else {
        nx = static_cast<int>(X[0]);
        for (k = 2; k <= nrows; k++) {
            i = static_cast<int>(X[k - 1]);
            if (nx > i) {
                nx = i;
            }
        }
    }
    positiveInput.set_size(Y.size(0));
    nblocks = Y.size(0);
    for (i = 0; i < nblocks; i++) {
        positiveInput[i] = (static_cast<int>(Y[i]) == nx);
    }
    exponent = (positiveInput.size(0) >= 1);
    nrows = 0;
    nx = 0;
    exitg1 = false;
    while ((!exitg1) && (nx <= positiveInput.size(0) - 1)) {
        if (positiveInput[nx]) {
            nrows = 1;
            ii_data = nx + 1;
            exitg1 = true;
        } else {
            nx++;
        }
    }
    if (exponent == 1) {
        if (nrows == 0) {
            exponent = 0;
        }
    } else {
        exponent = (nrows >= 1);
    }
    for (i = 0; i < exponent; i++) {
        pivotIndex_data = static_cast<unsigned int>(ii_data);
    }
    positiveInput.set_size(interpedIndex.size(0));
    nblocks = interpedIndex.size(0);
    for (i = 0; i < nblocks; i++) {
        positiveInput[i] = (interpedIndex[i] == subcarrierIndex[static_cast<int>(pivotIndex_data) - 1]);
    }
    exponent = (positiveInput.size(0) >= 1);
    nrows = 0;
    nx = 0;
    exitg1 = false;
    while ((!exitg1) && (nx <= positiveInput.size(0) - 1)) {
        if (positiveInput[nx]) {
            nrows = 1;
            ii_data = nx + 1;
            exitg1 = true;
        } else {
            nx++;
        }
    }
    if (exponent == 1) {
        if (nrows == 0) {
            exponent = 0;
        }
    } else {
        exponent = (nrows >= 1);
    }
    pivotPhase.set_size(1, rawPhase.size(1), rawPhase.size(2), rawPhase.size(3));
    nblocks = rawPhase.size(3);
    for (i = 0; i < nblocks; i++) {
        nrows = rawPhase.size(2);
        for (i1 = 0; i1 < nrows; i1++) {
            nx = rawPhase.size(1);
            for (nyrows = 0; nyrows < nx; nyrows++) {
                pivotPhase[(nyrows + pivotPhase.size(1) * i1) + pivotPhase.size(1) * pivotPhase.size(2) * i] =
                    rawPhase[(((static_cast<int>(pivotIndex_data) + rawPhase.size(0) * nyrows) +
                               rawPhase.size(0) * rawPhase.size(1) * i1) +
                              rawPhase.size(0) * rawPhase.size(1) * rawPhase.size(2) * i) -
                             1];
            }
        }
    }
    nrows = 1;
    if (rawPhase.size(0) != 1) {
        nrows = 0;
    } else if (rawPhase.size(1) == 1) {
        if (rawPhase.size(2) != 1) {
            nrows = 2;
        } else if (rawPhase.size(3) != 1) {
            nrows = 3;
        }
    }
    xtmp = rawPhase.size(nrows) - 1;
    X.set_size(rawPhase.size(nrows));
    ntilecols = 1;
    for (k = 0; k < nrows; k++) {
        ntilecols *= rawPhase.size(k);
    }
    ibtile = (rawPhase.size(nrows) - 1) * ntilecols;
    nblocks = 1;
    nrows += 2;
    for (k = nrows; k < 5; k++) {
        nblocks *= rawPhase.size(k - 1);
    }
    ibmat = 0;
    for (nyrows = 0; nyrows < nblocks; nyrows++) {
        nrows = ibmat - 1;
        ibmat += ibtile;
        for (b_exponent = 0; b_exponent < ntilecols; b_exponent++) {
            nrows++;
            ibmat++;
            for (k = 0; k <= xtmp; k++) {
                X[k] = rawPhase[nrows + k * ntilecols];
            }
            nx = X.size(0);
            cumsum_dp_corr = 0.0;
            pivotIndex_data = 1U;
            exitg1 = false;
            while ((!exitg1) && (static_cast<int>(pivotIndex_data) < nx)) {
                dp = X[static_cast<int>(pivotIndex_data) - 1];
                if (std::isinf(dp) || std::isnan(dp)) {
                    pivotIndex_data = static_cast<unsigned int>(static_cast<int>(pivotIndex_data) + 1);
                } else {
                    exitg1 = true;
                }
            }
            if (static_cast<int>(pivotIndex_data) < X.size(0)) {
                nfft = X[static_cast<int>(pivotIndex_data) - 1];
                int exitg2;
                do {
                    exitg2 = 0;
                    pivotIndex_data++;
                    exitg1 = false;
                    while ((!exitg1) && (pivotIndex_data <= static_cast<unsigned int>(nx))) {
                        dp = X[static_cast<int>(pivotIndex_data) - 1];
                        if (std::isinf(dp) || std::isnan(dp)) {
                            pivotIndex_data++;
                        } else {
                            exitg1 = true;
                        }
                    }
                    if (pivotIndex_data > static_cast<unsigned int>(nx)) {
                        exitg2 = 1;
                    } else {
                        r = X[static_cast<int>(pivotIndex_data) - 1];
                        dp = r - nfft;
                        nfft = dp / 6.2831853071795862;
                        if (std::abs(rt_remd_snf(nfft, 1.0)) <= 0.5) {
                            nfft = std::trunc(nfft);
                        } else {
                            nfft = std::round(nfft);
                        }
                        if (std::abs(dp) >= 3.1415926535897931) {
                            cumsum_dp_corr += nfft;
                        }
                        nfft = r;
                        X[static_cast<int>(pivotIndex_data) - 1] = r - 6.2831853071795862 * cumsum_dp_corr;
                    }
                } while (exitg2 == 0);
            }
            for (k = 0; k <= xtmp; k++) {
                rawPhase[nrows + k * ntilecols] = X[k];
            }
        }
    }
    X.set_size(subcarrierIndex.size(0));
    nblocks = subcarrierIndex.size(0);
    for (i = 0; i < nblocks; i++) {
        X[i] = subcarrierIndex[i];
    }
    nyrows = rawPhase.size(0);
    nycols = rawPhase.size(1) * rawPhase.size(2) * rawPhase.size(3) - 1;
    nx = subcarrierIndex.size(0) - 1;
    resultPhase.set_size(interpedIndex.size(0), rawPhase.size(1), rawPhase.size(2), rawPhase.size(3));
    nblocks = interpedIndex.size(0) * rawPhase.size(1) * rawPhase.size(2) * rawPhase.size(3);
    for (i = 0; i < nblocks; i++) {
        resultPhase[i] = rtNaN;
    }
    if (interpedIndex.size(0) != 0) {
        if (subcarrierIndex[1] < subcarrierIndex[0]) {
            i = subcarrierIndex.size(0) >> 1;
            for (ntilecols = 0; ntilecols < i; ntilecols++) {
                xtmp = static_cast<int>(X[ntilecols]);
                nblocks = nx - ntilecols;
                X[ntilecols] = X[nblocks];
                X[nblocks] = xtmp;
            }
            if ((rawPhase.size(0) != 0) && (rawPhase.size(1) != 0) && (rawPhase.size(2) != 0) &&
                (rawPhase.size(3) != 0) && (rawPhase.size(0) > 1)) {
                ntilecols = rawPhase.size(0) - 1;
                ibtile = rawPhase.size(0) >> 1;
                i = rawPhase.size(1) * rawPhase.size(2) * rawPhase.size(3) - 1;
                for (b_exponent = 0; b_exponent <= i; b_exponent++) {
                    nrows = b_exponent * rawPhase.size(0);
                    for (k = 0; k < ibtile; k++) {
                        xtmp = nrows + k;
                        nfft = rawPhase[xtmp];
                        i1 = (nrows + ntilecols) - k;
                        rawPhase[xtmp] = rawPhase[i1];
                        rawPhase[i1] = nfft;
                    }
                }
            }
        }
        ibtile = interpedIndex.size(0);
        for (k = 0; k < ibtile; k++) {
            if (std::isnan(interpedIndex[k])) {
                for (b_exponent = 0; b_exponent <= nycols; b_exponent++) {
                    resultPhase[k + b_exponent * ibtile] = rtNaN;
                }
            } else if ((!(interpedIndex[k] > X[X.size(0) - 1])) && (!(interpedIndex[k] < X[0]))) {
                nrows = X.size(0);
                ibmat = 1;
                nx = 2;
                while (nrows > nx) {
                    ntilecols = (ibmat >> 1) + (nrows >> 1);
                    if (((ibmat & 1) == 1) && ((nrows & 1) == 1)) {
                        ntilecols++;
                    }
                    if (interpedIndex[k] >= X[ntilecols - 1]) {
                        ibmat = ntilecols;
                        nx = ntilecols + 1;
                    } else {
                        nrows = ntilecols;
                    }
                }
                nfft = X[ibmat - 1];
                r = (interpedIndex[k] - nfft) / (X[ibmat] - nfft);
                if (r == 0.0) {
                    for (b_exponent = 0; b_exponent <= nycols; b_exponent++) {
                        resultPhase[k + b_exponent * ibtile] = rawPhase[(ibmat + b_exponent * nyrows) - 1];
                    }
                } else if (r == 1.0) {
                    for (b_exponent = 0; b_exponent <= nycols; b_exponent++) {
                        resultPhase[k + b_exponent * ibtile] = rawPhase[ibmat + b_exponent * nyrows];
                    }
                } else {
                    for (b_exponent = 0; b_exponent <= nycols; b_exponent++) {
                        nrows = ibmat + b_exponent * nyrows;
                        nfft = rawPhase[nrows - 1];
                        dp = rawPhase[nrows];
                        if (nfft == dp) {
                            resultPhase[k + b_exponent * ibtile] = nfft;
                        } else {
                            resultPhase[k + b_exponent * ibtile] = (1.0 - r) * nfft + r * dp;
                        }
                    }
                }
            }
        }
    }
    b_x.set_size(exponent, resultPhase.size(1), resultPhase.size(2), resultPhase.size(3));
    nblocks = resultPhase.size(3);
    for (i = 0; i < nblocks; i++) {
        nrows = resultPhase.size(2);
        for (i1 = 0; i1 < nrows; i1++) {
            nx = resultPhase.size(1);
            for (nyrows = 0; nyrows < nx; nyrows++) {
                for (ibtile = 0; ibtile < exponent; ibtile++) {
                    b_x[(b_x.size(0) * nyrows + b_x.size(0) * b_x.size(1) * i1) +
                        b_x.size(0) * b_x.size(1) * b_x.size(2) * i] =
                        resultPhase[(((ii_data + resultPhase.size(0) * nyrows) +
                                      resultPhase.size(0) * resultPhase.size(1) * i1) +
                                     resultPhase.size(0) * resultPhase.size(1) * resultPhase.size(2) * i) -
                                    1];
                }
            }
        }
    }
    nblocks = b_x.size(0) * b_x.size(1) * b_x.size(2) * b_x.size(3);
    pivotPhase.set_size(b_x.size(0), b_x.size(1), b_x.size(2), b_x.size(3));
    for (i = 0; i < nblocks; i++) {
        pivotPhase[i] = b_x[i] - pivotPhase[i];
    }
    b.set_size(interpedIndex.size(0), pivotPhase.size(1), pivotPhase.size(2), pivotPhase.size(3));
    if ((interpedIndex.size(0) != 0) && (pivotPhase.size(1) != 0) && (pivotPhase.size(2) != 0) &&
        (pivotPhase.size(3) != 0)) {
        nrows = pivotPhase.size(3);
        nx = pivotPhase.size(2);
        ibmat = pivotPhase.size(1);
        i = interpedIndex.size(0) - 1;
        for (k = 0; k < nrows; k++) {
            for (exponent = 0; exponent < nx; exponent++) {
                for (ntilecols = 0; ntilecols < ibmat; ntilecols++) {
                    for (ibtile = 0; ibtile <= i; ibtile++) {
                        b[((ibtile + b.size(0) * ntilecols) + b.size(0) * b.size(1) * exponent) +
                          b.size(0) * b.size(1) * b.size(2) * k] =
                            pivotPhase[(ntilecols + pivotPhase.size(1) * exponent) +
                                       pivotPhase.size(1) * pivotPhase.size(2) * k];
                    }
                }
            }
        }
    }
    nblocks = resultPhase.size(0) * resultPhase.size(1) * resultPhase.size(2) * resultPhase.size(3);
    for (i = 0; i < nblocks; i++) {
        resultPhase[i] = resultPhase[i] - b[i];
    }
    c_x.set_size(interpedIndex.size(0));
    nblocks = interpedIndex.size(0);
    for (i = 0; i < nblocks; i++) {
        c_x[i] = (interpedIndex[i] == 0.0);
    }
    nanflag = false;
    nrows = 1;
    exitg1 = false;
    while ((!exitg1) && (nrows <= c_x.size(0))) {
        if (c_x[nrows - 1]) {
            nanflag = true;
            exitg1 = true;
        } else {
            nrows++;
        }
    }
    if (nanflag) {
        positiveInput.set_size(interpedIndex.size(0));
        nblocks = interpedIndex.size(0);
        for (i = 0; i < nblocks; i++) {
            positiveInput[i] = (interpedIndex[i] == 0.0);
        }
        exponent = (positiveInput.size(0) >= 1);
        nrows = 0;
        nx = 0;
        exitg1 = false;
        while ((!exitg1) && (nx <= positiveInput.size(0) - 1)) {
            if (positiveInput[nx]) {
                nrows = 1;
                ii_data = nx + 1;
                exitg1 = true;
            } else {
                nx++;
            }
        }
        if (exponent == 1) {
            if (nrows == 0) {
                exponent = 0;
            }
        } else {
            exponent = (nrows >= 1);
        }
        for (i = 0; i < exponent; i++) {
            pivotIndex_data = static_cast<unsigned int>(ii_data);
        }
        pivotPhase.set_size(1, resultPhase.size(1), resultPhase.size(2), resultPhase.size(3));
        nblocks = resultPhase.size(3);
        for (i = 0; i < nblocks; i++) {
            nrows = resultPhase.size(2);
            for (i1 = 0; i1 < nrows; i1++) {
                nx = resultPhase.size(1);
                for (nyrows = 0; nyrows < nx; nyrows++) {
                    pivotPhase[(nyrows + pivotPhase.size(1) * i1) + pivotPhase.size(1) * pivotPhase.size(2) * i] =
                        resultPhase[(((static_cast<int>(pivotIndex_data) + resultPhase.size(0) * nyrows) +
                                      resultPhase.size(0) * resultPhase.size(1) * i1) +
                                     resultPhase.size(0) * resultPhase.size(1) * resultPhase.size(2) * i) -
                                    1];
                }
            }
        }
        r1.set_size(1, resultPhase.size(1), resultPhase.size(2), resultPhase.size(3));
        nblocks = resultPhase.size(3);
        for (i = 0; i < nblocks; i++) {
            nrows = resultPhase.size(2);
            for (i1 = 0; i1 < nrows; i1++) {
                nx = resultPhase.size(1);
                for (nyrows = 0; nyrows < nx; nyrows++) {
                    r1[(nyrows + r1.size(1) * i1) + r1.size(1) * r1.size(2) * i] =
                        (resultPhase[(((static_cast<int>(pivotIndex_data) + resultPhase.size(0) * nyrows) +
                                       resultPhase.size(0) * resultPhase.size(1) * i1) +
                                      resultPhase.size(0) * resultPhase.size(1) * resultPhase.size(2) * i) -
                                     1] < -3.1415926535897931);
                }
            }
        }
        r2.set_size(1, resultPhase.size(1), resultPhase.size(2), resultPhase.size(3));
        nblocks = resultPhase.size(3);
        for (i = 0; i < nblocks; i++) {
            nrows = resultPhase.size(2);
            for (i1 = 0; i1 < nrows; i1++) {
                nx = resultPhase.size(1);
                for (nyrows = 0; nyrows < nx; nyrows++) {
                    r2[(nyrows + r2.size(1) * i1) + r2.size(1) * r2.size(2) * i] =
                        (resultPhase[(((static_cast<int>(pivotIndex_data) + resultPhase.size(0) * nyrows) +
                                       resultPhase.size(0) * resultPhase.size(1) * i1) +
                                      resultPhase.size(0) * resultPhase.size(1) * resultPhase.size(2) * i) -
                                     1] > 3.1415926535897931);
                }
            }
        }
        nx = r1.size(1) * (r1.size(2) * r1.size(3)) - 1;
        nrows = 0;
        for (nyrows = 0; nyrows <= nx; nyrows++) {
            if (r1[nyrows] || r2[nyrows]) {
                nrows++;
            }
        }
        r3.set_size(nrows);
        nrows = 0;
        for (nyrows = 0; nyrows <= nx; nyrows++) {
            if (r1[nyrows] || r2[nyrows]) {
                r3[nrows] = nyrows + 1;
                nrows++;
            }
        }
        c_resultPhase.set_size(1, resultPhase.size(1), resultPhase.size(2), resultPhase.size(3));
        nblocks = resultPhase.size(3);
        for (i = 0; i < nblocks; i++) {
            nrows = resultPhase.size(2);
            for (i1 = 0; i1 < nrows; i1++) {
                nx = resultPhase.size(1);
                for (nyrows = 0; nyrows < nx; nyrows++) {
                    c_resultPhase[(nyrows + c_resultPhase.size(1) * i1) +
                                  c_resultPhase.size(1) * c_resultPhase.size(2) * i] =
                        resultPhase[(((static_cast<int>(pivotIndex_data) + resultPhase.size(0) * nyrows) +
                                      resultPhase.size(0) * resultPhase.size(1) * i1) +
                                     resultPhase.size(0) * resultPhase.size(1) * resultPhase.size(2) * i) -
                                    1];
                }
            }
        }
        X.set_size(r3.size(0));
        nblocks = r3.size(0);
        for (i = 0; i < nblocks; i++) {
            X[i] = c_resultPhase[r3[i] - 1] + 3.1415926535897931;
        }
        positiveInput.set_size(X.size(0));
        nblocks = X.size(0);
        for (i = 0; i < nblocks; i++) {
            positiveInput[i] = (X[i] > 0.0);
        }
        nblocks = X.size(0);
        for (i = 0; i < nblocks; i++) {
            nfft = X[i];
            if (std::isnan(nfft) || std::isinf(nfft)) {
                dp = rtNaN;
            } else if (nfft == 0.0) {
                dp = 0.0;
            } else {
                dp = std::fmod(nfft, 6.2831853071795862);
                nanflag = (dp == 0.0);
                if (!nanflag) {
                    r = std::abs(nfft / 6.2831853071795862);
                    nanflag = !(std::abs(r - std::floor(r + 0.5)) > 2.2204460492503131E-16 * r);
                }
                if (nanflag) {
                    dp = 0.0;
                } else if (nfft < 0.0) {
                    dp += 6.2831853071795862;
                }
            }
            X[i] = dp;
        }
        if (X.size(0) == positiveInput.size(0)) {
            positiveInput.set_size(X.size(0));
            nblocks = X.size(0);
            for (i = 0; i < nblocks; i++) {
                positiveInput[i] = ((X[i] == 0.0) && positiveInput[i]);
            }
        } else {
            binary_expand_op(positiveInput, X);
        }
        nx = positiveInput.size(0);
        for (nyrows = 0; nyrows < nx; nyrows++) {
            if (positiveInput[nyrows]) {
                X[nyrows] = 6.2831853071795862;
            }
        }
        nx = r1.size(1) * (r1.size(2) * r1.size(3)) - 1;
        nrows = 0;
        for (nyrows = 0; nyrows <= nx; nyrows++) {
            if (r1[nyrows] || r2[nyrows]) {
                nrows++;
            }
        }
        r4.set_size(nrows);
        nrows = 0;
        for (nyrows = 0; nyrows <= nx; nyrows++) {
            if (r1[nyrows] || r2[nyrows]) {
                r4[nrows] = nyrows + 1;
                nrows++;
            }
        }
        nblocks = X.size(0);
        for (i = 0; i < nblocks; i++) {
            pivotPhase[r4[i] - 1] = X[i] - 3.1415926535897931;
        }
        pivotPhase.set_size(1, resultPhase.size(1), resultPhase.size(2), resultPhase.size(3));
        nblocks = resultPhase.size(3);
        for (i = 0; i < nblocks; i++) {
            nrows = resultPhase.size(2);
            for (i1 = 0; i1 < nrows; i1++) {
                nx = resultPhase.size(1);
                for (nyrows = 0; nyrows < nx; nyrows++) {
                    pivotPhase[(nyrows + pivotPhase.size(1) * i1) + pivotPhase.size(1) * pivotPhase.size(2) * i] =
                        resultPhase[(((static_cast<int>(pivotIndex_data) + resultPhase.size(0) * nyrows) +
                                      resultPhase.size(0) * resultPhase.size(1) * i1) +
                                     resultPhase.size(0) * resultPhase.size(1) * resultPhase.size(2) * i) -
                                    1] -
                        pivotPhase[(nyrows + pivotPhase.size(1) * i1) + pivotPhase.size(1) * pivotPhase.size(2) * i];
                }
            }
        }
        positiveInput.set_size(pivotPhase.size(1) * pivotPhase.size(2) * pivotPhase.size(3));
        nblocks = pivotPhase.size(1) * pivotPhase.size(2) * pivotPhase.size(3);
        for (i = 0; i < nblocks; i++) {
            positiveInput[i] = (pivotPhase[i] == 0.0);
        }
        nanflag = true;
        nrows = 1;
        exitg1 = false;
        while ((!exitg1) && (nrows <= positiveInput.size(0))) {
            if (!positiveInput[nrows - 1]) {
                nanflag = false;
                exitg1 = true;
            } else {
                nrows++;
            }
        }
        if (!nanflag) {
            b.set_size(interpedIndex.size(0), pivotPhase.size(1), pivotPhase.size(2), pivotPhase.size(3));
            if ((interpedIndex.size(0) != 0) && (pivotPhase.size(1) != 0) && (pivotPhase.size(2) != 0) &&
                (pivotPhase.size(3) != 0)) {
                nrows = pivotPhase.size(3);
                nx = pivotPhase.size(2);
                ibmat = pivotPhase.size(1);
                i = interpedIndex.size(0) - 1;
                for (k = 0; k < nrows; k++) {
                    for (exponent = 0; exponent < nx; exponent++) {
                        for (ntilecols = 0; ntilecols < ibmat; ntilecols++) {
                            for (ibtile = 0; ibtile <= i; ibtile++) {
                                b[((ibtile + b.size(0) * ntilecols) + b.size(0) * b.size(1) * exponent) +
                                  b.size(0) * b.size(1) * b.size(2) * k] =
                                    pivotPhase[(ntilecols + pivotPhase.size(1) * exponent) +
                                               pivotPhase.size(1) * pivotPhase.size(2) * k];
                            }
                        }
                    }
                }
            }
            nblocks = resultPhase.size(0) * resultPhase.size(1) * resultPhase.size(2) * resultPhase.size(3);
            for (i = 0; i < nblocks; i++) {
                resultPhase[i] = resultPhase[i] - b[i];
            }
        }
    }
    i = static_cast<int>(numSTS);
    phaseSlope.set_size(i, static_cast<int>(numRx), static_cast<int>(numCSI));
    phaseIntercept.set_size(i, static_cast<int>(numRx), static_cast<int>(numCSI));
    for (nrows = 0; nrows < i; nrows++) {
        i1 = static_cast<int>(numRx);
        for (nx = 0; nx < i1; nx++) {
            nyrows = static_cast<int>(numCSI);
            for (ntilecols = 0; ntilecols < nyrows; ntilecols++) {
                nblocks = resultPhase.size(0);
                b_resultPhase.set_size(resultPhase.size(0));
                for (ibtile = 0; ibtile < nblocks; ibtile++) {
                    b_resultPhase[ibtile] =
                        resultPhase[((ibtile + resultPhase.size(0) * nrows) +
                                     resultPhase.size(0) * resultPhase.size(1) * nx) +
                                    resultPhase.size(0) * resultPhase.size(1) * resultPhase.size(2) * ntilecols];
                }
                double outsize[2];
                coder::polyfit(interpedIndex, b_resultPhase, outsize);
                phaseSlope[(nrows + phaseSlope.size(0) * nx) + phaseSlope.size(0) * phaseSlope.size(1) * ntilecols] =
                    outsize[0];
                phaseIntercept[(nrows + phaseIntercept.size(0) * nx) +
                               phaseIntercept.size(0) * phaseIntercept.size(1) * ntilecols] = outsize[1];
            }
        }
    }
    nyrows = rawMag.size(0);
    nycols = rawMag.size(1) * rawMag.size(2) * rawMag.size(3) - 1;
    nx = subcarrierIndex.size(0) - 1;
    resultMag.set_size(interpedIndex.size(0), rawMag.size(1), rawMag.size(2), rawMag.size(3));
    nblocks = interpedIndex.size(0) * rawMag.size(1) * rawMag.size(2) * rawMag.size(3);
    for (i = 0; i < nblocks; i++) {
        resultMag[i] = rtNaN;
    }
    if (interpedIndex.size(0) != 0) {
        if (subcarrierIndex[1] < subcarrierIndex[0]) {
            i = subcarrierIndex.size(0) >> 1;
            for (ntilecols = 0; ntilecols < i; ntilecols++) {
                xtmp = subcarrierIndex[ntilecols];
                nrows = nx - ntilecols;
                subcarrierIndex[ntilecols] = subcarrierIndex[nrows];
                subcarrierIndex[nrows] = static_cast<short>(xtmp);
            }
            if ((rawMag.size(0) != 0) && (rawMag.size(1) != 0) && (rawMag.size(2) != 0) && (rawMag.size(3) != 0) &&
                (rawMag.size(0) > 1)) {
                ntilecols = rawMag.size(0) - 1;
                ibtile = rawMag.size(0) >> 1;
                i = rawMag.size(1) * rawMag.size(2) * rawMag.size(3) - 1;
                for (b_exponent = 0; b_exponent <= i; b_exponent++) {
                    nrows = b_exponent * rawMag.size(0);
                    for (k = 0; k < ibtile; k++) {
                        xtmp = nrows + k;
                        nfft = rawMag[xtmp];
                        i1 = (nrows + ntilecols) - k;
                        rawMag[xtmp] = rawMag[i1];
                        rawMag[i1] = nfft;
                    }
                }
            }
        }
        ibtile = interpedIndex.size(0);
        for (k = 0; k < ibtile; k++) {
            if (std::isnan(interpedIndex[k])) {
                for (b_exponent = 0; b_exponent <= nycols; b_exponent++) {
                    resultMag[k + b_exponent * ibtile] = rtNaN;
                }
            } else if ((!(interpedIndex[k] > subcarrierIndex[subcarrierIndex.size(0) - 1])) &&
                       (!(interpedIndex[k] < subcarrierIndex[0]))) {
                nrows = subcarrierIndex.size(0);
                ibmat = 1;
                nx = 2;
                while (nrows > nx) {
                    ntilecols = (ibmat >> 1) + (nrows >> 1);
                    if (((ibmat & 1) == 1) && ((nrows & 1) == 1)) {
                        ntilecols++;
                    }
                    if (interpedIndex[k] >= subcarrierIndex[ntilecols - 1]) {
                        ibmat = ntilecols;
                        nx = ntilecols + 1;
                    } else {
                        nrows = ntilecols;
                    }
                }
                r = (interpedIndex[k] - static_cast<double>(subcarrierIndex[ibmat - 1])) /
                    static_cast<double>(subcarrierIndex[ibmat] - subcarrierIndex[ibmat - 1]);
                if (r == 0.0) {
                    for (b_exponent = 0; b_exponent <= nycols; b_exponent++) {
                        resultMag[k + b_exponent * ibtile] = rawMag[(ibmat + b_exponent * nyrows) - 1];
                    }
                } else if (r == 1.0) {
                    for (b_exponent = 0; b_exponent <= nycols; b_exponent++) {
                        resultMag[k + b_exponent * ibtile] = rawMag[ibmat + b_exponent * nyrows];
                    }
                } else {
                    for (b_exponent = 0; b_exponent <= nycols; b_exponent++) {
                        nrows = ibmat + b_exponent * nyrows;
                        nfft = rawMag[nrows - 1];
                        dp = rawMag[nrows];
                        if (nfft == dp) {
                            resultMag[k + b_exponent * ibtile] = nfft;
                        } else {
                            resultMag[k + b_exponent * ibtile] = (1.0 - r) * nfft + r * dp;
                        }
                    }
                }
            }
        }
    }
    resultCSI.set_size(resultPhase.size(0), resultPhase.size(1), resultPhase.size(2), resultPhase.size(3));
    nblocks = resultPhase.size(0) * resultPhase.size(1) * resultPhase.size(2) * resultPhase.size(3);
    for (i = 0; i < nblocks; i++) {
        resultCSI[i].re = resultPhase[i] * 0.0;
        resultCSI[i].im = resultPhase[i];
    }
    nx = resultCSI.size(0) * resultCSI.size(1) * resultCSI.size(2) * resultCSI.size(3);
    for (k = 0; k < nx; k++) {
        if (resultCSI[k].im == 0.0) {
            resultCSI[k].re = std::exp(resultCSI[k].re);
            resultCSI[k].im = 0.0;
        } else if (std::isinf(resultCSI[k].im) && std::isinf(resultCSI[k].re) && (resultCSI[k].re < 0.0)) {
            resultCSI[k].re = 0.0;
            resultCSI[k].im = 0.0;
        } else {
            r = std::exp(resultCSI[k].re / 2.0);
            dp = resultCSI[k].im;
            resultCSI[k].re = r * (r * std::cos(resultCSI[k].im));
            resultCSI[k].im = r * (r * std::sin(dp));
        }
    }
    nblocks = resultMag.size(0) * resultMag.size(1) * resultMag.size(2) * resultMag.size(3);
    resultCSI.set_size(resultMag.size(0), resultMag.size(1), resultMag.size(2), resultMag.size(3));
    for (i = 0; i < nblocks; i++) {
        resultCSI[i].re = resultMag[i] * resultCSI[i].re;
        resultCSI[i].im = resultMag[i] * resultCSI[i].im;
    }
    interpedIndex_int16.set_size(interpedIndex.size(0));
    nblocks = interpedIndex.size(0);
    for (i = 0; i < nblocks; i++) {
        interpedIndex_int16[i] = static_cast<short>(std::round(interpedIndex[i]));
    }
}

//
// File trailer for CSIPreprocessor.cpp
//
// [EOF]
//
