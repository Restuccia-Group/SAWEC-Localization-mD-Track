//
// Created by Zhiping Jiang on 2020/12/19.
//

#include "PreEQSymbolsSegment.hxx"

static auto v1Parser = [](const uint8_t *buffer, uint32_t bufferLength) -> SignalMatrix<std::complex<double>> {
    auto signal = SignalMatrix<std::complex<double>>::fromBuffer(buffer, buffer + bufferLength, SignalMatrixStorageMajority::ColumnMajor);
    return signal;
};

std::map<uint16_t, std::function<SignalMatrix<std::complex<double>>(const uint8_t *, uint32_t)>> PreEQSymbolsSegment::initializeSolutionMap() noexcept {
    return std::map<uint16_t, std::function<SignalMatrix<std::complex<double>>(const uint8_t *, uint32_t)>>{{0x1U, v1Parser}};
}

std::map<uint16_t, std::function<SignalMatrix<std::complex<double>>(const uint8_t *, uint32_t)>> PreEQSymbolsSegment::versionedSolutionMap = initializeSolutionMap();

PreEQSymbolsSegment::PreEQSymbolsSegment() : AbstractPicoScenesFrameSegment("PreEQSymbols", 0x1U) {}

PreEQSymbolsSegment::PreEQSymbolsSegment(const uint8_t *buffer, uint32_t bufferLength) : AbstractPicoScenesFrameSegment(buffer, bufferLength) {
    if (segmentName != "PreEQSymbols")
        throw std::runtime_error("PreEQSymbolsSegment cannot parse the segment named " + segmentName + ".");
    if (!versionedSolutionMap.count(segmentVersionId))
        throw std::runtime_error("PreEQSymbolsSegment cannot parse the segment with version v" + std::to_string(segmentVersionId) + ".");

    preEQSymbols = versionedSolutionMap.at(segmentVersionId)(segmentPayload.data(), segmentPayload.size());
}

const SignalMatrix<std::complex<double>> &PreEQSymbolsSegment::getPreEqSymbols() const {
    return preEQSymbols;
}

void PreEQSymbolsSegment::setPreEqSymbols(const SignalMatrix<std::complex<double>> &preEqSymbolsV) {
    preEQSymbols = preEqSymbolsV;
    setSegmentPayload(preEQSymbols.toBuffer());
}

std::string PreEQSymbolsSegment::toString() const {
    std::stringstream ss;
    ss << segmentName + ":[" + std::to_string(preEQSymbols.dimensions[0]) + "x" + std::to_string(preEQSymbols.dimensions[1]) + (preEQSymbols.dimensions.size() >= 3 ? (std::string{"x"} + std::to_string(preEQSymbols.dimensions[2]) + "]") : std::string{"]"});
    auto temp = ss.str();
    return temp;
}

std::ostream &operator<<(std::ostream &os, const PreEQSymbolsSegment &preEqSymbolsSegment) {
    os << preEqSymbolsSegment.toString();
    return os;
}
