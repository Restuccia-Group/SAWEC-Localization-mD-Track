//
// Created by Zhiping Jiang on 2020/11/5.
//

#ifndef PICOSCENES_PLATFORM_ABSTRACTPICOSCENESFRAMESEGMENT_HXX
#define PICOSCENES_PLATFORM_ABSTRACTPICOSCENESFRAMESEGMENT_HXX

#include "PicoScenesCommons.hxx"
#include "DynamicFieldInterpretation.hxx"

/**
 * @brief PicoScenes Frame Segment is a typed, versioned and named data structure used to carrier various of data from the Tx end, PicoScenes driver or SDR baseband to the PicoScenes Platform.
 * @details AbstractPicoScenesFrameSegment is the root class for various of Frame segment classes.
 */
class AbstractPicoScenesFrameSegment {
public:
    std::string segmentName; ///< The name of the segment
    uint16_t segmentVersionId; ///< The version of the segment
    std::vector<uint8_t> rawBuffer; ///< The raw buffer of the segment, including the 4-byte length itself
    std::vector<uint8_t> segmentPayload; ///< The actual paylaod part of the segment

    static std::tuple<std::string, uint32_t, uint16_t, uint32_t> extractSegmentMetaData(const uint8_t *buffer, uint32_t bufferLength);

    AbstractPicoScenesFrameSegment(std::string segmentName, uint16_t segmentVersionId);

    AbstractPicoScenesFrameSegment(const uint8_t *buffer, size_t bufferLength);

    [[nodiscard]] const std::vector<uint8_t> &getSegmentPayload() const;

    std::vector<uint8_t> getSegmentPayload();

    /**
     * @brief Change the payload data, the rawBuffer is also updated.
     * @param payload
     */
    void setSegmentPayload(const std::vector<uint8_t> &payload);

    DynamicFieldInterpreter getDynamicInterpreter() const;

    /**
     * @brief Return the total length of the segment buffer
     * 
     * @return uint32_t 
     */
    [[nodiscard]] uint32_t totalLength() const;

    [[nodiscard]] inline uint32_t totalLengthIncludingLeading4ByteLength() const {
        return totalLength() + 4;
    }

    /**
     * @brief ModularPicoScenesTxFrame and ModularPicoScenesRxFrame call this method to get the whole data of this segment
     * @details By default, this method performs 3 things:
     * 1. call totalLength method to obtain the length of the raw buffer;
     * 2. pre-allocate a std::vector<uint8_t> buffer;
     * 3. fill the buffer with the other AbstractPicoScenesFrameSegment#toBuffer method
     * 
     * @param totalLengthIncluded where the returned buffer includes the leading 4-byte segment length field
     */
    [[nodiscard]] virtual std::vector<uint8_t> toBuffer() const;

    uint32_t toBuffer(uint8_t *buffer, std::optional<uint32_t> bufferLength = std::nullopt) const;

    /**
     * @brief Return a short description for this segment
     */
    [[nodiscard]] virtual std::string toString() const;

    virtual ~AbstractPicoScenesFrameSegment() {};

protected:
    void rebuildBuffer();
};

std::ostream &operator<<(std::ostream &os, const AbstractPicoScenesFrameSegment &segment);


#endif //PICOSCENES_PLATFORM_ABSTRACTPICOSCENESFRAMESEGMENT_HXX
