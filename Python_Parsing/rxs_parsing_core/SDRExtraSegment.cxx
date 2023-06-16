//
// Created by Zhiping Jiang on 10/23/22.
//

#include "SDRExtraSegment.hxx"


std::vector<uint8_t> SDRExtra::toBuffer() {
    return std::vector<uint8_t>{(uint8_t *) this, (uint8_t *) this + sizeof(SDRExtra)};
}

std::string SDRExtra::toString() const {
    std::stringstream ss;
    ss << "SDRExtra:[scrambler=" + std::to_string(scramblerInit) + ", packetStartInternal=" + std::to_string(packetStartInternal) + ", rxIndex=" + std::to_string(hardwareRxSamplingIndex) + ", rxTime=" + std::to_string(preciseRxTime) + ", lastTxTime=" + std::to_string(lastTxTime) + "]";
    return ss.str();
}

std::ostream &operator<<(std::ostream &os, const SDRExtra &sdrExtra) {
    os << sdrExtra.toString();
    return os;
}

struct SDRExtraV1 {
    int8_t scramblerInit;
    int64_t packetStartInternal;
    double lastTxTime;
} __attribute__ ((__packed__));

struct SDRExtraV2 {
    int8_t scramblerInit;
    int64_t packetStartInternal;
    double preciseRxTime;
    double lastTxTime;
} __attribute__ ((__packed__));

struct SDRExtraV3 {
    int8_t scramblerInit;
    int64_t packetStartInternal;
    int64_t hardwareRxSamplingIndex;
    double preciseRxTime;
    double lastTxTime;
} __attribute__ ((__packed__));

static auto v1Parser = [](const uint8_t *buffer, uint32_t bufferLength) -> SDRExtra {
    uint32_t pos = 0;
    if (bufferLength < sizeof(SDRExtraV1))
        throw std::runtime_error("SDRExtra v1Parser cannot parse the segment with insufficient buffer length.");

    auto r = SDRExtra();
    std::memset(&r, 0, sizeof(r));
    r.scramblerInit = *(int8_t *) (buffer + pos);
    pos += sizeof(int8_t);
    r.packetStartInternal = *(int64_t *) (buffer + pos);
    pos += sizeof(int64_t);
    r.preciseRxTime = 0;
    r.lastTxTime = *(double *) (buffer + pos);
    pos += sizeof(double);

    if (pos != bufferLength)
        throw std::runtime_error("SDRExtra v1Parser cannot parse the segment with mismatched buffer length.");

    return r;
};

static auto v2Parser = [](const uint8_t *buffer, uint32_t bufferLength) -> SDRExtra {
    uint32_t pos = 0;
    if (bufferLength < sizeof(SDRExtraV2))
        throw std::runtime_error("SDRExtra v2Parser cannot parse the segment with insufficient buffer length.");

    auto r = SDRExtra();
    std::memset(&r, 0, sizeof(r));
    r.scramblerInit = *(int8_t *) (buffer + pos);
    pos += sizeof(int8_t);
    r.packetStartInternal = *(int64_t *) (buffer + pos);
    pos += sizeof(int64_t);
    r.preciseRxTime = *(double *) (buffer + pos);
    pos += sizeof(double);
    r.lastTxTime = *(double *) (buffer + pos);
    pos += sizeof(double);

    if (pos != bufferLength)
        throw std::runtime_error("SDRExtra v2Parser cannot parse the segment with mismatched buffer length.");

    return r;
};

static auto v3Parser = [](const uint8_t *buffer, uint32_t bufferLength) -> SDRExtra {
    uint32_t pos = 0;
    if (bufferLength < sizeof(SDRExtraV3))
        throw std::runtime_error("SDRExtra v3Parser cannot parse the segment with insufficient buffer length.");

    auto r = SDRExtra();
    std::memset(&r, 0, sizeof(r));
    r.scramblerInit = *(int8_t *) (buffer + pos);
    pos += sizeof(int8_t);
    r.packetStartInternal = *(int64_t *) (buffer + pos);
    pos += sizeof(int64_t);
    r.hardwareRxSamplingIndex = *(int64_t *) (buffer + pos);
    pos += sizeof(int64_t);
    r.preciseRxTime = *(double *) (buffer + pos);
    pos += sizeof(double);
    r.lastTxTime = *(double *) (buffer + pos);
    pos += sizeof(double);

    if (pos != bufferLength)
        throw std::runtime_error("SDRExtra v3Parser cannot parse the segment with mismatched buffer length.");

    return r;
};

std::map<uint16_t, std::function<SDRExtra(const uint8_t *, uint32_t)>> SDRExtraSegment::versionedSolutionMap = initializeSolutionMap();

std::map<uint16_t, std::function<SDRExtra(const uint8_t *, uint32_t)>> SDRExtraSegment::initializeSolutionMap() noexcept {
    return std::map<uint16_t, std::function<SDRExtra(const uint8_t *, uint32_t)>>{{0x1U, v1Parser},
                                                                                  {0x2U, v2Parser},
                                                                                  {0x3U, v3Parser}};
}

SDRExtraSegment::SDRExtraSegment() : AbstractPicoScenesFrameSegment("SDRExtra", 0x3U) {}

SDRExtraSegment::SDRExtraSegment(const uint8_t *buffer, uint32_t bufferLength) : AbstractPicoScenesFrameSegment(buffer, bufferLength) {
    if (segmentName != "SDRExtra")
        throw std::runtime_error("SDRExtraSegment cannot parse the segment named " + segmentName + ".");
    if (!versionedSolutionMap.count(segmentVersionId))
        throw std::runtime_error("SDRExtraSegment cannot parse the segment with version v" + std::to_string(segmentVersionId) + ".");

    sdrExtra = versionedSolutionMap.at(segmentVersionId)(segmentPayload.data(), segmentPayload.size());
    setSdrExtra(sdrExtra);
}

SDRExtraSegment::SDRExtraSegment(const SDRExtra &sdrExtra) : SDRExtraSegment() {
    setSdrExtra(sdrExtra);
}

const SDRExtra &SDRExtraSegment::getSdrExtra() const {
    return sdrExtra;
}

void SDRExtraSegment::setSdrExtra(const SDRExtra &sdrExtraV) {
    sdrExtra = sdrExtraV;
    setSegmentPayload(sdrExtra.toBuffer());
}

std::string SDRExtraSegment::toString() const {
    return sdrExtra.toString();
}
