//
// Created by Zhiping Jiang on 2020/11/6.
//

#ifndef PICOSCENES_PLATFORM_BASEBANDSIGNALSEGMENT_HXX
#define PICOSCENES_PLATFORM_BASEBANDSIGNALSEGMENT_HXX

#include "AbstractPicoScenesFrameSegment.hxx"
#include "PicoScenesCommons.hxx"

class BasebandSignalSegment : public AbstractPicoScenesFrameSegment {
public:
    BasebandSignalSegment();

    BasebandSignalSegment(const uint8_t *buffer, uint32_t bufferLength);

    [[maybe_unused]] [[nodiscard]] const SignalMatrix<std::complex<double>> &getSignalMatrix() const;

    [[maybe_unused]] [[nodiscard]] const SignalMatrix<std::complex<float>> &getFloat32SignalMatrix() const;

    void setSignalMatrix(const SignalMatrix<std::complex<double>> &bbsignalsV);

    void setSignalMatrix(const SignalMatrix<std::complex<float>> &bbsignalsV);

    [[nodiscard]] std::string toString() const override;

private:
    SignalMatrix<std::complex<double>> bbsignals;

    SignalMatrix<std::complex<float>> bbsignalsFloat32;
};

std::ostream &operator<<(std::ostream &os, const BasebandSignalSegment &csiSegment);


#endif //PICOSCENES_PLATFORM_BASEBANDSIGNALSEGMENT_HXX
