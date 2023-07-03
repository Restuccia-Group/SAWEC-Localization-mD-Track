//
// Created by Zhiping Jiang on 4/20/22.
//

#ifndef PICOSCENES_PLATFORM_CARGOSEGMENT_HXX
#define PICOSCENES_PLATFORM_CARGOSEGMENT_HXX

#include "AbstractPicoScenesFrameSegment.hxx"
#include "PicoScenesCommons.hxx"

class CargoCompression {
public:
    static bool isAlgorithmRegistered() {
        return compressor && decompressor;
    }

    static void registerAlgorithm(const std::function<std::optional<std::vector<uint8_t>>(const uint8_t *, size_t)> &compressorV,
                                  const std::function<std::optional<std::vector<uint8_t>>(const uint8_t *, size_t)> &decompressorV) {
        static std::once_flag onceFlag;
        std::call_once(onceFlag, [&] {
            compressor = compressorV;
            decompressor = decompressorV;
        });
    }

    static const std::function<std::optional<std::vector<uint8_t>>(const uint8_t *, size_t)> &getCompressor() {
        return compressor;
    }

    static const std::function<std::optional<std::vector<uint8_t>>(const uint8_t *, size_t)> &getDecompressor() {
        return decompressor;
    }

private:
    static std::function<std::optional<std::vector<uint8_t>>(const uint8_t *, size_t)> compressor;
    static std::function<std::optional<std::vector<uint8_t>>(const uint8_t *, size_t)> decompressor;
};

class PayloadCargo {
public:
    uint16_t taskId{0};
    uint8_t numSegments{0};
    uint8_t sequence{0};
    uint8_t totalParts{0};
    bool compressed{false};
    uint32_t payloadLength{0};
    std::vector<uint8_t> payloadData;

    [[nodiscard]] size_t totalLength() const;

    [[nodiscard]] std::vector<uint8_t> toBuffer() const;

    static PayloadCargo fromBuffer(const uint8_t *buffer, uint32_t bufferLength);

    static PayloadCargo fromBuffer(const std::vector<uint8_t> &buffer);

    static Uint8Vector mergeAndValidateCargo(const std::vector<PayloadCargo> &cargos);
};

class CargoSegment : public AbstractPicoScenesFrameSegment {
public:
    CargoSegment();

    explicit CargoSegment(const PayloadCargo &cargo);

    CargoSegment(const uint8_t *buffer, uint32_t bufferLength);

    [[nodiscard]] const PayloadCargo &getCargo() const;

    void setCargo(const PayloadCargo &cargo);

    [[nodiscard]] std::string toString() const override;

private:
    static std::map<uint16_t, std::function<PayloadCargo(const uint8_t *, uint32_t)>> versionedSolutionMap;

    static std::map<uint16_t, std::function<PayloadCargo(const uint8_t *, uint32_t)>> initializeSolutionMap() noexcept;

    PayloadCargo cargo;
};

std::ostream &operator<<(std::ostream &os, const CargoSegment &cargoSegment);


#endif //PICOSCENES_PLATFORM_CARGOSEGMENT_HXX
