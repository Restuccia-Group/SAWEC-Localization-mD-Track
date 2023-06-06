//
// Created by Zhiping Jiang on 12/13/20.
//

#ifndef PICOSCENES_PLATFORM_PAYLOADSEGMENT_HXX
#define PICOSCENES_PLATFORM_PAYLOADSEGMENT_HXX

#include "AbstractPicoScenesFrameSegment.hxx"
#include "PicoScenesCommons.hxx"

enum class PayloadDataType : uint8_t {
    RawData = 0,
    SegmentData,
    SignalMatrix,
    CSIData,
    FullMSDUPacket,
    FullPicoScenesPacket,
};

std::ostream &operator<<(std::ostream &os, const PayloadDataType &payloadDataType);


class PayloadData {
public:
    PayloadDataType dataType = PayloadDataType::RawData;
    std::string payloadDescription;
    std::vector<uint8_t> payloadData;

    std::vector<uint8_t> toBuffer();

    static PayloadData fromBuffer(const uint8_t *buffer, uint32_t bufferLength);

    static PayloadData fromBuffer(const std::vector<uint8_t> &buffer);
};

class PayloadSegment : public AbstractPicoScenesFrameSegment {

public:
    PayloadSegment();

    PayloadSegment(const uint8_t *buffer, uint32_t bufferLength);

    PayloadSegment(const std::string &description, const std::vector<uint8_t> &payload, std::optional<PayloadDataType> payloadType = std::nullopt);

    PayloadSegment(const std::string &description, const uint8_t *payloadBuffer, size_t bufferLength, std::optional<PayloadDataType> payloadType = std::nullopt);

    PayloadSegment(const std::string &description, const std::string &stringPayload, std::optional<PayloadDataType> payloadType = std::nullopt);

    const PayloadData &getPayloadData() const;

    void setPayloadData(const PayloadData &payload);

    std::string toString() const override;

private:

    static std::map<uint16_t, std::function<PayloadData(const uint8_t *, uint32_t)>> versionedSolutionMap;

    static std::map<uint16_t, std::function<PayloadData(const uint8_t *, uint32_t)>> initializeSolutionMap() noexcept;

    PayloadData payloadData;
};

std::ostream &operator<<(std::ostream &os, const PayloadSegment &payloadSegment);


#endif //PICOSCENES_PLATFORM_PAYLOADSEGMENT_HXX
