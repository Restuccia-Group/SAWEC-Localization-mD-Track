//
// Created by Zhiping Jiang on 2020/11/6.
//

#ifndef PICOSCENES_PLATFORM_PICOSCENESFRAMETXPARAMETERS_HXX
#define PICOSCENES_PLATFORM_PICOSCENESFRAMETXPARAMETERS_HXX

#include "PicoScenesCommons.hxx"
#include "FrontEndModePreset.hxx"
#include "SDRResamplingPreset.hxx"

class TxPrecodingParameters {
public:
    SignalMatrix<std::complex<double>> userSpecifiedSpatialMappingMatrix;

    double simulateTxDistance{0};

    SignalMatrix<std::complex<double>> rxCSI4TxBeamforming; /// N_{sc}-by-N_{tx_sts}-by-N_{rx}
    uint8_t txNumSTS4TxBeamforming{1};

    std::vector<std::array<double, 3>> antennaPositions;
    double txSteeringAngles4PhasedArray{0};
    double carrierFrequency{0};

    [[nodiscard]] inline bool isActivated() const {
        return !userSpecifiedSpatialMappingMatrix.empty() || simulateTxDistance != 0 || !rxCSI4TxBeamforming.empty() || (txSteeringAngles4PhasedArray != 0 && !antennaPositions.empty());
    }
};

class PicoScenesFrameTxParameters {
public:
    std::shared_ptr<FrontEndModePreset> preset = FrontEndModePreset::Customized;
    std::optional<double> preciseTxTime = std::nullopt;
    bool NDPFrame;
    PacketFormatEnum frameType;
    std::vector<uint8_t> mcs;
    std::vector<uint8_t> numSTS;
    double numAntenna;
    uint16_t txcm;
    ChannelBandwidthEnum cbw;
    GuardIntervalEnum guardInterval;
    std::vector<ChannelCodingEnum> coding;
    double numExtraSounding;
    bool forceSounding;
    double prefixPaddingTime;
    double idleTime;
    double postfixPaddingTime;
    double scramblerState;
    double txIQAmplitudeImbalance_dB;
    double txIQPhaseImbalance_rad;
    bool maxPowerScaleTo1;
    double hardwareSamplingRate;
    double samplingRateOffset;
    double carrierFrequencyOffset;
    double resampleRatio;
    double simulateTxDistance;
    double vhtGroupId;
    std::vector<uint8_t> heAllocationIndex;
    bool txHEExtendedRange;
    bool heHighDoppler;
    double heMidamblePeriodicity;
    double heLTFType;
    bool hePreHESpatialMapping;
    double additiveNoiseVarianceDb;
    TxPrecodingParameters precodingParameters;

    PicoScenesFrameTxParameters() {
        reset();
    }

    void reset() {
        NDPFrame = false;
        frameType = PacketFormatEnum::PacketFormat_HT;
        mcs = std::vector<uint8_t>(1, 0);
        numSTS = std::vector<uint8_t>(1, 1);
        numAntenna = 1;
        txcm = 1;
        cbw = ChannelBandwidthEnum::CBW_20;
        guardInterval = GuardIntervalEnum::GI_800;
        coding = std::vector<ChannelCodingEnum>(1, ChannelCodingEnum::BCC);
        numExtraSounding = 0;
        forceSounding = true;
        prefixPaddingTime = 4e-6;
        idleTime = 4e-6;
        postfixPaddingTime = 0;
        scramblerState = 39;
        txIQAmplitudeImbalance_dB = 0;
        txIQPhaseImbalance_rad = 0;
        maxPowerScaleTo1 = true;
        hardwareSamplingRate = 20e6;
        samplingRateOffset = 0;
        carrierFrequencyOffset = 0;
        resampleRatio = 1.0;
        simulateTxDistance = 0;
        vhtGroupId = 0;
        heAllocationIndex = std::vector<uint8_t>(1, 0);
        txHEExtendedRange = false;
        heHighDoppler = false;
        heMidamblePeriodicity = 10;
        heLTFType = 4;
        hePreHESpatialMapping = false;
        additiveNoiseVarianceDb = -45;
    }

    void applyPreset(const std::string &presetName) {
        if (!FrontEndModePreset::getPresetMap().contains(presetName))
            throw std::invalid_argument("invalid frontend mode preset name: " + presetName + "\n" + FrontEndModePreset::printHelpContentForFrontEndModePreset());
        applyPreset(FrontEndModePreset::getPresetMap().at(presetName));
    }

    void applyPreset(const std::shared_ptr<FrontEndModePreset> &presetV) {
        preset = presetV;
        if (preset->txSamplingRate && preset->txResampleRatio && preset->txCBW) {
            hardwareSamplingRate = preset->txSamplingRate.value();
            cbw = ChannelBandwidthEnum(preset->txCBW.value());
            frameType = preset->format;
            coding = std::vector<ChannelCodingEnum>{preset->coding};
            resampleRatio = preset->txResampleRatio.value();
        }
    }

    void validate() const;

    std::string toString() const;
};

std::ostream &operator<<(std::ostream &os, const PicoScenesFrameTxParameters &parameters);

#endif //PICOSCENES_PLATFORM_PICOSCENESFRAMETXPARAMETERS_HXX
