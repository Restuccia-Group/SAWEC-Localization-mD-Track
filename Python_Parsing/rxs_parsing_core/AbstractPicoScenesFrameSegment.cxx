//
// Created by Zhiping Jiang on 2020/11/5.
//

#include "AbstractPicoScenesFrameSegment.hxx"

AbstractPicoScenesFrameSegment::AbstractPicoScenesFrameSegment(std::string segmentName, uint16_t segmentVersionId) : segmentName(segmentName), segmentVersionId(segmentVersionId) {}

AbstractPicoScenesFrameSegment::AbstractPicoScenesFrameSegment(const uint8_t *buffer, size_t bufferLength) {
    auto [segmentNameV, segmentLengthV, versionIdV, offset] = extractSegmentMetaData(buffer, bufferLength);
    if (segmentLengthV + 4 > bufferLength)
        throw std::underflow_error("Cannot parse the segment [" + segmentNameV + "] version [" + std::to_string(versionIdV) + "[with less than " + std::to_string(segmentLengthV + 4) + "B.");

    segmentName = segmentNameV;
    segmentVersionId = versionIdV;
    std::copy(buffer, buffer + bufferLength, std::back_inserter(rawBuffer));
    std::copy(buffer + offset, buffer + bufferLength, std::back_inserter(segmentPayload));
}

uint32_t AbstractPicoScenesFrameSegment::totalLength() const {
    /**
     * 1B segment name length;
     * segmentName.size() + 1B, for segment name and the ending \0
     * 2B of version Id
     */
    return 2 + segmentName.size() + sizeof(segmentVersionId) + segmentPayload.size();;
}

std::vector<uint8_t> AbstractPicoScenesFrameSegment::toBuffer() const {
    auto result = std::vector<uint8_t>(totalLengthIncludingLeading4ByteLength());
    toBuffer(result.data(), result.size());
    return result;
}


uint32_t AbstractPicoScenesFrameSegment::toBuffer(uint8_t *buffer, std::optional<uint32_t> bufferLength) const {
    if (rawBuffer.size() != totalLengthIncludingLeading4ByteLength())
        throw std::runtime_error("Segment toBuffer failed");

    if (bufferLength.has_value() && rawBuffer.size() > *bufferLength)
        throw std::underflow_error("bufferLength not long enough");

    std::copy(rawBuffer.cbegin(), rawBuffer.cend(), buffer);
    return rawBuffer.size();
}

std::string AbstractPicoScenesFrameSegment::toString() const {
    return segmentName + ":" + std::to_string(rawBuffer.size()) + "B";
}

std::tuple<std::string, uint32_t, uint16_t, uint32_t> AbstractPicoScenesFrameSegment::extractSegmentMetaData(const uint8_t *buffer, uint32_t bufferLength) {
    uint32_t rxPos = 0;
    uint32_t segmentLength = *(uint32_t *) (buffer + rxPos);
    rxPos += 4;
    if (segmentLength > bufferLength)
        throw std::runtime_error("corrupted segment...");
    auto segNameLength = *(buffer + rxPos++);
    auto segmentName = std::string((char *) (buffer + rxPos), *(char *) (buffer + rxPos + segNameLength - 1) == '\0' ? ((char *) (buffer + rxPos + segNameLength - 1)) : ((char *) (buffer + rxPos + segNameLength)));
    rxPos += segNameLength;
    uint16_t segmentVersionId = *(uint16_t *) (buffer + rxPos);
    rxPos += 2;

    return std::make_tuple(segmentName, segmentLength, segmentVersionId, rxPos);
}

const std::vector<uint8_t> &AbstractPicoScenesFrameSegment::getSegmentPayload() const {
    return segmentPayload;
}

std::vector<uint8_t> AbstractPicoScenesFrameSegment::getSegmentPayload() {
    return segmentPayload;
}

void AbstractPicoScenesFrameSegment::setSegmentPayload(const std::vector<uint8_t> &payload) {
    segmentPayload = payload;
    rebuildBuffer();
}

void AbstractPicoScenesFrameSegment::rebuildBuffer() {
    uint32_t totalLengthV = totalLength();
    uint8_t segNameLength = segmentName.length() + 1;
    rawBuffer.clear();
    std::copy((uint8_t *) &totalLengthV, (uint8_t *) &totalLengthV + sizeof(totalLengthV), std::back_inserter(rawBuffer));
    std::copy((uint8_t *) &segNameLength, (uint8_t *) &segNameLength + sizeof(segNameLength), std::back_inserter(rawBuffer));
    std::copy((uint8_t *) segmentName.data(), (uint8_t *) (segmentName.data() + segmentName.size() + 1), std::back_inserter(rawBuffer));
    std::copy((uint8_t *) &segmentVersionId, (uint8_t *) &segmentVersionId + sizeof(segmentVersionId), std::back_inserter(rawBuffer));
    std::copy(segmentPayload.cbegin(), segmentPayload.cend(), std::back_inserter(rawBuffer));
}

DynamicFieldInterpreter AbstractPicoScenesFrameSegment::getDynamicInterpreter() const {
    return DynamicFieldInterpreter{segmentName, segmentVersionId, segmentPayload.data()};
}

std::ostream &operator<<(std::ostream &os, const AbstractPicoScenesFrameSegment &segment) {
    os << segment.toString();
    return os;
}
