//
// Created by  Zhiping Jiang on 10/23/22.
//

#ifndef PICOSCENES_PLATFORM_SDREXTRASEGMENT_HXX
#define PICOSCENES_PLATFORM_SDREXTRASEGMENT_HXX

#include <unordered_map>
#include "AbstractPicoScenesFrameSegment.hxx"
#include "PicoScenesCommons.hxx"

struct SDRExtra {
    int8_t scramblerInit{0};
    int64_t packetStartInternal{0};
    int64_t hardwareRxSamplingIndex{0};
    double preciseRxTime{0};
    double lastTxTime{0};
    double signalInputSystemTime{0};
    double signalDecodeSystemTime{0};
    double sigEVM{0};

    [[nodiscard]] inline double decodingDelay() const {
        return signalDecodeSystemTime - signalInputSystemTime;
    }

    [[nodiscard]] std::string toString() const;

    std::vector<uint8_t> toBuffer();

} __attribute__ ((__packed__));

std::ostream &operator<<(std::ostream &os, const SDRExtra &sdrExtra);


class SDRExtraSegment : public AbstractPicoScenesFrameSegment {
public:
    SDRExtraSegment();

    SDRExtraSegment(const uint8_t *buffer, uint32_t bufferLength);

    explicit SDRExtraSegment(const SDRExtra &sdrExtra);

    const SDRExtra &getSdrExtra() const;

    void setSdrExtra(const SDRExtra &sdrExtra);

    std::string toString() const override;

private:
    static std::map<uint16_t, std::function<SDRExtra(const uint8_t *, uint32_t)>> versionedSolutionMap;

    static std::map<uint16_t, std::function<SDRExtra(const uint8_t *, uint32_t)>> initializeSolutionMap() noexcept;

    SDRExtra sdrExtra{};

};


#endif //PICOSCENES_PLATFORM_SDREXTRASEGMENT_HXX
