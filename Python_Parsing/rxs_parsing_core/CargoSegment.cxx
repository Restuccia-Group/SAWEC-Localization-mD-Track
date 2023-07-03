//
// Created by Zhiping Jiang on 4/20/22.
//

#include "CargoSegment.hxx"

std::function<std::optional<std::vector<uint8_t>>(const uint8_t *, size_t)> CargoCompression::compressor;
std::function<std::optional<std::vector<uint8_t>>(const uint8_t *, size_t)> CargoCompression::decompressor;

class PayloadCargoV1 {
public:
    uint16_t taskId{0};
    uint8_t numSegments{0};
    uint8_t sequence{0};
    uint8_t totalParts{0};
    bool compressed{false};
    uint32_t payloadLength{0};
    std::vector<uint8_t> payloadData;
};

size_t PayloadCargo::totalLength() const {
    return sizeof(taskId) + sizeof(numSegments) + sizeof(totalParts) + sizeof(sequence) + +sizeof(compressed) + sizeof(payloadLength) + payloadData.size();
}

std::vector<uint8_t> PayloadCargo::toBuffer() const {
    auto buffer = std::vector<uint8_t>();
    uint32_t cargoLength = totalLength();
    std::copy((uint8_t *) &cargoLength, (uint8_t *) &cargoLength + sizeof(cargoLength), std::back_inserter(buffer));
    std::copy((uint8_t *) &taskId, (uint8_t *) &taskId + sizeof(taskId), std::back_inserter(buffer));
    std::copy((uint8_t *) &numSegments, (uint8_t *) &numSegments + sizeof(numSegments), std::back_inserter(buffer));
    std::copy((uint8_t *) &sequence, (uint8_t *) &sequence + sizeof(sequence), std::back_inserter(buffer));
    std::copy((uint8_t *) &totalParts, (uint8_t *) &totalParts + sizeof(totalParts), std::back_inserter(buffer));
    std::copy((uint8_t *) &compressed, (uint8_t *) &compressed + sizeof(compressed), std::back_inserter(buffer));
    std::copy((uint8_t *) &payloadLength, (uint8_t *) &payloadLength + sizeof(payloadLength), std::back_inserter(buffer));
    std::copy(payloadData.cbegin(), payloadData.cend(), std::back_inserter(buffer));

    return buffer;
}

PayloadCargo PayloadCargo::fromBuffer(const uint8_t *buffer, uint32_t bufferLength) {
    uint32_t cargoLength = *(uint32_t *) buffer;

    if (cargoLength != bufferLength - 4)[[unlikely]] {
        throw std::runtime_error("cargo segment length inconsistent");
    }

    auto pos = 4;
    PayloadCargo cargo;
    cargo.taskId = *(decltype(taskId) *) (buffer + pos);
    pos += sizeof(taskId);
    cargo.numSegments = *(decltype(numSegments) *) (buffer + pos);
    pos += sizeof(numSegments);
    cargo.sequence = *(decltype(sequence) *) (buffer + pos);
    pos += sizeof(sequence);
    cargo.totalParts = *(decltype(totalParts) *) (buffer + pos);
    pos += sizeof(totalParts);
    cargo.compressed = *(decltype(compressed) *) (buffer + pos);
    pos += sizeof(compressed);
    cargo.payloadLength = *(decltype(payloadLength) *) (buffer + pos);
    pos += sizeof(payloadLength);
    std::copy(buffer + pos, buffer + bufferLength, std::back_inserter(cargo.payloadData));

    return cargo;
}

PayloadCargo PayloadCargo::fromBuffer(const std::vector<uint8_t> &buffer) {
    return fromBuffer(buffer.data(), buffer.size());
}

Uint8Vector PayloadCargo::mergeAndValidateCargo(const std::vector<PayloadCargo> &cargos) {
    Uint8Vector rawPayload, decompressedPayload;
    std::for_each(cargos.cbegin(), cargos.cend(), [&](const PayloadCargo &cargo) {
        std::copy(cargo.payloadData.cbegin(), cargo.payloadData.cend(), std::back_inserter(rawPayload));
    });

    if (cargos.front().compressed && CargoCompression::isAlgorithmRegistered()) {
        decompressedPayload = CargoCompression::getDecompressor()(rawPayload.data(), rawPayload.size()).value_or(Uint8Vector());
    } else {
        decompressedPayload = rawPayload;
    }

    auto pos = 0, numSegment = 0;
    while (pos < decompressedPayload.size()) {
        auto [segmentName, segmentLength, versionId, contentOffset] = AbstractPicoScenesFrameSegment::extractSegmentMetaData(decompressedPayload.data() + pos, decompressedPayload.size() - pos);
        pos += segmentLength + 4;
        numSegment++;
    }

    if (pos != decompressedPayload.size() && numSegment != cargos.front().numSegments) [[unlikely]] {
        throw std::runtime_error("incorrect cargo length");
    }

    return decompressedPayload;
}

static auto v1Parser = [](const uint8_t *buffer, uint32_t bufferLength) -> PayloadCargo {
    auto cargo = PayloadCargo::fromBuffer(buffer, bufferLength);
    return cargo;
};

std::map<uint16_t, std::function<PayloadCargo(const uint8_t *, uint32_t)>> CargoSegment::versionedSolutionMap = initializeSolutionMap();

std::map<uint16_t, std::function<PayloadCargo(const uint8_t *, uint32_t)>> CargoSegment::initializeSolutionMap() noexcept {
    return std::map<uint16_t, std::function<PayloadCargo(const uint8_t *, uint32_t)>>{{0x1U, v1Parser}};
}

CargoSegment::CargoSegment() : AbstractPicoScenesFrameSegment("Cargo", 0x1U) {}

CargoSegment::CargoSegment(const uint8_t *buffer, uint32_t bufferLength) : AbstractPicoScenesFrameSegment(buffer, bufferLength) {
    if (segmentName != "Cargo")
        throw std::runtime_error("CargoSegment cannot parse the segment named " + segmentName + ".");
    if (!versionedSolutionMap.count(segmentVersionId)) {
        throw std::runtime_error("CargoSegment cannot parse the segment with version v" + std::to_string(segmentVersionId) + ".");
    }
    cargo = versionedSolutionMap.at(segmentVersionId)(segmentPayload.data(), segmentPayload.size());
}

CargoSegment::CargoSegment(const PayloadCargo &cargoV) : CargoSegment() {
    setCargo(cargoV);
}

const PayloadCargo &CargoSegment::getCargo() const {
    return cargo;
}

void CargoSegment::setCargo(const PayloadCargo &cargoV) {
    cargo = cargoV;
    setSegmentPayload(cargo.toBuffer());
}

std::string CargoSegment::toString() const {
    return segmentName + ":[taskId=" + std::to_string(cargo.taskId) + ", numSeg=" + std::to_string(cargo.numSegments) + ", seq=" + std::to_string(cargo.sequence) + ", total_part=" + std::to_string(cargo.totalParts) + ", compressed=" + std::to_string(cargo.compressed) + ", total_length=" + std::to_string(cargo.payloadLength) + ", part_len=" + std::to_string(cargo.payloadData.size()) + "]";
}

std::ostream &operator<<(std::ostream &os, const CargoSegment &cargoSegment) {
    os << cargoSegment.toString();
    return os;
}
