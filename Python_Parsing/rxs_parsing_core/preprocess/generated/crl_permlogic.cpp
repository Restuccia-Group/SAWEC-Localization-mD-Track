#include "crl_permlogic.h"

#ifdef PERMLOGIC_EXISTS
#include "PermLogicCore.hxx"
#endif

double permLogic(double cbw, double format, double u1, double u2, double u3, double u4, double u5, double u6, double u7, double u8) {
#ifdef PERMLOGIC_EXISTS
    return permlogic_core<double>(cbw, format, u1, u2, u3, u4, u5, u6, u7, u8);
#endif
    return 1.0;
}