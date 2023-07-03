//
// Created by Zhiping Jiang on 2020/11/6.
//

#include "BasebandSignalSegment.hxx"

static auto v1Parser = [](const uint8_t *buffer, uint32_t bufferLength, void *bbsignal) {
    *(SignalMatrix<std::complex<double>> *) bbsignal = SignalMatrix<std::complex<double>>::fromBuffer(buffer, buffer + bufferLength, SignalMatrixStorageMajority::ColumnMajor);
};

static auto v2Parser = [](const uint8_t *buffer, uint32_t bufferLength, void *bbsignal) {
    *(SignalMatrix<std::complex<float>> *) bbsignal = SignalMatrix<std::complex<float>>::fromBuffer(buffer, buffer + bufferLength, SignalMatrixStorageMajority::ColumnMajor);
};

BasebandSignalSegment::BasebandSignalSegment() : AbstractPicoScenesFrameSegment("BasebandSignal", 0x2U) {}

BasebandSignalSegment::BasebandSignalSegment(const uint8_t *buffer, uint32_t bufferLength) : AbstractPicoScenesFrameSegment(buffer, bufferLength) {
    if (segmentName != "BasebandSignal")
        throw std::runtime_error("BasebandSignalSegment cannot parse the segment named " + segmentName + ".");
    if (segmentVersionId != 1 && segmentVersionId != 2)
        throw std::runtime_error("BasebandSignalSegment cannot parse the segment with version v" + std::to_string(segmentVersionId) + ".");

    if (segmentVersionId == 0x1U)
        v1Parser(segmentPayload.data(), segmentPayload.size(), (void *) &bbsignals);
    else if (segmentVersionId == 0x2U)
        v2Parser(segmentPayload.data(), segmentPayload.size(), (void *) &bbsignalsFloat32);
}

[[maybe_unused]] const SignalMatrix<std::complex<double>> &BasebandSignalSegment::getSignalMatrix() const {
    return bbsignals;
}

[[maybe_unused]] const SignalMatrix<std::complex<float>> &BasebandSignalSegment::getFloat32SignalMatrix() const {
    return bbsignalsFloat32;
}

void BasebandSignalSegment::setSignalMatrix(const SignalMatrix<std::complex<double>> &bbsignalsV) {
    bbsignals = bbsignalsV;
    setSegmentPayload(bbsignals.toBuffer());
}

void BasebandSignalSegment::setSignalMatrix(const SignalMatrix<std::complex<float>> &bbsignalsV) {
    bbsignalsFloat32 = bbsignalsV;
    setSegmentPayload(bbsignalsFloat32.toBuffer());
}

void BasebandSignalSegment::setSignalMatrix(const SignalMatrix<std::complex<int16_t>> &bbsignalsV) {
    bbsignalsInt16 = bbsignalsV;
    setSegmentPayload(bbsignalsInt16.toBuffer());
}

std::string BasebandSignalSegment::toString() const {
    std::stringstream ss;
    if (!bbsignals.empty())
        ss << segmentName + ":[(double) " + std::to_string(bbsignals.dimensions[0]) + "x" + std::to_string(bbsignals.dimensions[1]) + "]";
    else if (!bbsignalsFloat32.empty())
        ss << segmentName + ":[(float) " + std::to_string(bbsignalsFloat32.dimensions[0]) + "x" + std::to_string(bbsignalsFloat32.dimensions[1]) + "]";
    else if (!bbsignalsInt16.empty())
        ss << segmentName + ":[(int16) " + std::to_string(bbsignalsInt16.dimensions[0]) + "x" + std::to_string(bbsignalsInt16.dimensions[1]) + "]";
    auto temp = ss.str();
    return temp;
}

std::ostream &operator<<(std::ostream &os, const BasebandSignalSegment &basebandSignalSegment) {
    os << basebandSignalSegment.toString();
    return os;
}
