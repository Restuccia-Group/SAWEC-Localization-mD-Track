//
// Created by Zhiping Jiang on 2020/11/5.
//

#ifndef PICOSCENES_PLATFORM_CSISEGMENT_HXX
#define PICOSCENES_PLATFORM_CSISEGMENT_HXX

#include "AbstractPicoScenesFrameSegment.hxx"
#include "PicoScenesCommons.hxx"

template<typename Iterator>
void parseQCA9300CSIData(Iterator outputArray, const uint8_t *csiData, int nSTS, int nRx, int nTones) {

    auto parse10bitsValues = [](const uint8_t rawByte[5], int outputArray[4]) {
        static uint16_t negativeSignBit = (1 << (10 - 1));
        static uint16_t minNegativeValue = (1 << 10);
        outputArray[0] = ((rawByte[0] & 0xffU) >> 0U) + ((rawByte[1] & 0x03u) << 8u);
        outputArray[1] = ((rawByte[1] & 0xfcU) >> 2U) + ((rawByte[2] & 0x0fU) << 6U);
        outputArray[2] = ((rawByte[2] & 0xf0U) >> 4U) + ((rawByte[3] & 0x3fU) << 4U);
        outputArray[3] = ((rawByte[3] & 0xc0U) >> 6U) + ((rawByte[4] & 0xffU) << 2U);
        for (auto i = 0; i < 4; i++) {
            if (outputArray[i] & negativeSignBit)
                outputArray[i] -= minNegativeValue;
        }
    };

    int valuePos, pos, rxIndex, txIndex, toneIndex, totalTones = nRx * nSTS * nTones;
    int tempArray[4];
    for (auto i = 0; i < totalTones / 2; i++) {
        parse10bitsValues(csiData + i * 5, tempArray);

        valuePos = i * 2;
        rxIndex = valuePos % nRx;
        txIndex = (valuePos / nRx) % nSTS;
        toneIndex = valuePos / (nRx * nSTS);
        pos = rxIndex * (nSTS * nTones) + txIndex * nTones + toneIndex;
        outputArray[pos].real(tempArray[1]);
        outputArray[pos].imag(tempArray[0]);

        valuePos = i * 2 + 1;
        rxIndex = valuePos % nRx;
        txIndex = (valuePos / nRx) % nSTS;
        toneIndex = valuePos / (nRx * nSTS);
        pos = rxIndex * (nSTS * nTones) + txIndex * nTones + toneIndex;
        outputArray[pos].real(tempArray[3]);
        outputArray[pos].imag(tempArray[2]);
    }
}

template<typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v) {

    // initialize original index locations
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    std::sort(idx.begin(), idx.end(),
              [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });

    return idx;
}

template<typename Iterator>
void parseIWL5300CSIData(Iterator csi_matrix, const uint8_t *payload, int ntx, int nrx, uint8_t ant_sel) {

    auto positionComputationWRTPermutation = [](int ntx, int num_tones, int ntxIdx, int nrxIdx, int subcarrierIdx, std::optional<Uint8Vector> ant_sel) -> int {
        auto new_nrxIdx = nrxIdx;
        if (ant_sel && !ant_sel->empty()) {
            auto sorted_indexes = sort_indexes(*ant_sel);
            auto sorted_indexes2 = sort_indexes(sorted_indexes); // double sort, shit but works !
            new_nrxIdx = sorted_indexes2[nrxIdx];
        }

        return new_nrxIdx * (ntx * num_tones) + ntxIdx * num_tones + subcarrierIdx;
    };

    std::vector<uint8_t> antSelVector = [&](uint8_t ant_sel) {
        auto v = std::vector<uint8_t>();
        if (nrx > 1) {
            v.emplace_back(static_cast<unsigned char>(((unsigned) ant_sel & 0x1U) + 1));
            v.emplace_back(static_cast<unsigned char>((((unsigned) ant_sel >> 0x2U) & 0x3U) + 1));
        };
        if (nrx > 2)
            v.emplace_back(static_cast<unsigned char>((((unsigned) ant_sel >> 0x4U) & 0x3U) + 1));
        return v;
    }(ant_sel);

    uint32_t index = 0;
    uint8_t remainder;

//    auto position = 0;
    for (auto subcarrierIdx = 0; subcarrierIdx < 30; subcarrierIdx++) {
        index += 3;
        remainder = index % 8;

        for (auto nrxIdx = 0; nrxIdx < nrx; nrxIdx++) {
            for (auto ntxIdx = 0; ntxIdx < ntx; ntxIdx++) {
                auto position = positionComputationWRTPermutation(ntx, 30, ntxIdx, nrxIdx, subcarrierIdx, antSelVector);
                char tmp1 = (payload[index / 8] >> remainder) | (payload[index / 8 + 1] << (8 - remainder));
                char tmp2 = (payload[index / 8 + 1] >> remainder) | (payload[index / 8 + 2] << (8 - remainder));
                csi_matrix[position].real((double) tmp1);
                csi_matrix[position].imag((double) tmp2);
                index += 16;
            }
        }
    }
}

/**
 * @brief Describe the dimensions of @see CSI class
 */
class CSIDimension {
public:
    uint16_t numTones = 1; ///< Number of OFDM subcarriers
    uint8_t numTx = 1; ///< Number of Tx spatial streams
    uint8_t numRx = 1; ///< Number of Rx radio chains
    uint8_t numESS = 0; /// Number of Extra Spatial Soudning (ESS) streams
    uint16_t numCSI = 1; /// Number of 3-D CSI matrix, usually 1. However, for LegacyCSI and PilotCSI, since these ARE multiple OFDM symbols, numCSI maybe greater than 1.

    /**
     * Return the total number of Tx spatial streams
     * @return numTx + numESS
     */
    inline uint16_t numTxSpatialStreams() const {
        return numTx + numESS;
    }

    inline uint16_t numStreamsPerCSI() const {
        return numTxSpatialStreams() * numRx;
    }

    inline uint16_t numTotalSubcarriersPerCSI() const {
        return numStreamsPerCSI() * numTones;
    }

    inline uint16_t numStreamsAll() const {
        return numStreamsPerCSI() * numCSI;
    }

    inline uint16_t numTotalSubcarriersAll() const {
        return numStreamsAll() * numTones;
    }
};

/**
 * The core CSI data class
 */
class CSI {
public:
    PicoScenesDeviceType deviceType; ///< The type of the device that measures this CSI
    uint8_t firmwareVersion; ///< The firmware version, only used for AX200/AX210
    PacketFormatEnum packetFormat; ///< The packet format of this frame
    ChannelBandwidthEnum cbw; ///< The channel bandwidth (CBW) parameter of this frame
    uint64_t carrierFreq; ///< The actual carrier frequency
    uint64_t samplingRate; ///< The actual hardware sampling rate
    uint32_t subcarrierBandwidth; ///< The bandwidth of OFDM subcarriers, usually 312.5e3 for 802.11/a/g/n/ac and 78.125e3 for 802.11ax
    CSIDimension dimensions; ///< The dimension information of the CSI data
    uint8_t antSel; ///< The antenna selection (antenna permutation) information, only used for IWL5300 NIC
    int16_t subcarrierOffset; ///< the index offset of the subcarrierIndices. For example, for a HT20-rate packet, subcarrierOffset = 0; for HT40+, subcarrierOffset = 32; for HT40-, subcarrierOffset = -32;
    std::vector<int16_t> subcarrierIndices; ///< The CSI subcarrier index. For example, [-28, 28] for 11n format frame.
    SignalMatrix<std::complex<double>> CSIArray; ///< The core CSI data matrix
    SignalMatrix<double> magnitudeArray; ///< The magnitude data matrix
    SignalMatrix<double> phaseArray; ///< The phase data matrix
    SignalMatrix<double> phaseSlope; ///< The slope of the interpolated & unwrapped phase
    SignalMatrix<double> phaseIntercept; ///< The intercept of the interpolated & unwrapped phase

    Uint8Vector rawCSIData; ///< The raw CSI data

    /**
     * Perform cyclic shift delay (CSD) removal and CSI interpolation
     * @details The CSD removal operation removes the additionally added phase slope for tha 2nd/3rd spatial streams;
     * The CSI interpolation operation interpolates the 0-th subcarrier, meanwhile generates the magnitudeArray and phaseArray
     */
    void removeCSDAndInterpolateCSI();

    /**
     * Convert current CSI object to raw buffer, for storage or transfer
     * @return
     */
    std::vector<uint8_t> toBuffer();

    /**
     * Create CSI object from the QCA9300 NIC returned raw bytes
     */
    static CSI fromQCA9300(const uint8_t *buffer, uint32_t bufferLength, uint8_t numTx, uint8_t numRx, uint8_t numTones, ChannelBandwidthEnum cbw, int16_t subcarrierIndexOffset);

    /**
     * Create CSI object from the IWL5300 NIC returned raw bytes
     */
    static CSI fromIWL5300(const uint8_t *buffer, uint32_t bufferLength, uint8_t numTx, uint8_t numRx, uint8_t numTones, ChannelBandwidthEnum cbw, int16_t subcarrierIndexOffset, uint8_t ant_sel);

    /**
     * Create CSI object from the AX200/AX210 NIC returned raw bytes
     */
    static std::optional<CSI> fromIWLMVM(const uint8_t *buffer, uint32_t bufferLength, uint8_t firmwareVersion, uint8_t numTx, uint8_t numRx, uint16_t numTones, PacketFormatEnum format, ChannelBandwidthEnum cbw, int16_t subcarrierIndexOffset, bool skipPilotSubcarriers = true, uint8_t antSelByte = 0);

    /**
     * A CSI data format converter
     * @tparam OutputValueType
     * @tparam InputValueType
     * @param inputArray
     * @return
     */
    template<typename OutputValueType, typename InputValueType>
    static std::vector<std::complex<OutputValueType>> convertCSIArrayType(const std::vector<std::complex<InputValueType>> &inputArray) {
        std::vector<std::complex<OutputValueType>> outputArray(inputArray.size());
        for (auto i = 0; i < inputArray.size(); i++) {
            outputArray[i] = std::complex<OutputValueType>(inputArray[i].real(), inputArray[i].imag());
        }
        return outputArray;
    }

    /**
     * @brief get the index of the 0-th subcarrier (DC subcarrier)
     * @return An int index if the measurement contains 0-th subcarrier, else a std::nullopt value.
     */
    std::optional<int16_t> get0thSubcarrierIndex() const;

    /**
     * Return the CSI data with the specified @param subcarrierIndex, @param stsIndexStartingFrom0, @param rxIndexStartingFrom0, @param csiIndexStartingFrom0
     * @see getMagnitude, @see getPhase
     * @param subcarrierIndex
     * @param stsIndexStartingFrom0
     * @param rxIndexStartingFrom0
     * @param csiIndexStartingFrom0
     * @return
     */
    std::optional<std::complex<double>> getCSI(int16_t subcarrierIndex, uint8_t stsIndexStartingFrom0 = 0, uint8_t rxIndexStartingFrom0 = 0, uint16_t csiIndexStartingFrom0 = 0) const;

    std::optional<double> getMagnitude(int16_t subcarrierIndex, uint8_t stsIndex = 0, uint8_t rxIndex = 0, uint16_t csiIndexStartingFrom0 = 0) const;

    std::optional<double> getPhase(int16_t subcarrierIndex, uint8_t stsIndex = 0, uint8_t rxIndex = 0, uint16_t csiIndexStartingFrom0 = 0) const;

    static void setAutoUnperm(bool autoUnperm);

private:
    static bool autoUnperm;
};

class CSISubcarrierIndex {
public:
    static const std::vector<int16_t> &getAllSubcarrierIndices(PacketFormatEnum format, ChannelBandwidthEnum cbw);

    static const std::vector<int16_t> &getPilotSubcarrierIndices(PacketFormatEnum format, ChannelBandwidthEnum cbw);

    static const std::vector<int16_t> &getDataSubcarrierIndices(PacketFormatEnum format, ChannelBandwidthEnum cbw);

    static const std::vector<int16_t> &getIWL5300Subcarriers(ChannelBandwidthEnum cbw);

private:
    static std::vector<int16_t> NonHT20_52Subcarriers_Indices;
    static std::vector<int16_t> HTVHT20_56Subcarriers_Indices;
    static std::vector<int16_t> HTVHT40_114Subcarriers_Indices;
    static std::vector<int16_t> VHT80_242Subcarriers_Indices;
    static std::vector<int16_t> VHT160_484Subcarriers_Indices;
    static std::vector<int16_t> HE20_242Subcarriers_Indices;
    static std::vector<int16_t> HE40_484Subcarriers_Indices;
    static std::vector<int16_t> HE80_996Subcarriers_Indices;
    static std::vector<int16_t> HE160_1992Subcarriers_Indices;

    static std::vector<int16_t> NonHT20_52Subcarriers_DataIndices;
    static std::vector<int16_t> HTVHT20_56Subcarriers_DataIndices;
    static std::vector<int16_t> HTVHT40_114Subcarriers_DataIndices;
    static std::vector<int16_t> VHT80_242Subcarriers_DataIndices;
    static std::vector<int16_t> VHT160_484Subcarriers_DataIndices;
    static std::vector<int16_t> HE20_242Subcarriers_DataIndices;
    static std::vector<int16_t> HE40_484Subcarriers_DataIndices;
    static std::vector<int16_t> HE80_996Subcarriers_DataIndices;
    static std::vector<int16_t> HE160_1992Subcarriers_DataIndices;

    static std::vector<int16_t> NonHT20_52Subcarriers_PilotIndices;
    static std::vector<int16_t> HTVHT20_56Subcarriers_PilotIndices;
    static std::vector<int16_t> HTVHT40_114Subcarriers_PilotIndices;
    static std::vector<int16_t> VHT80_242Subcarriers_PilotIndices;
    static std::vector<int16_t> VHT160_484Subcarriers_PilotIndices;
    static std::vector<int16_t> HE20_242Subcarriers_PilotIndices;
    static std::vector<int16_t> HE40_484Subcarriers_PilotIndices;
    static std::vector<int16_t> HE80_996Subcarriers_PilotIndices;
    static std::vector<int16_t> HE160_1992Subcarriers_PilotIndices;

    static std::vector<int16_t> IWL5300SubcarrierIndices_CBW20;
    static std::vector<int16_t> IWL5300SubcarrierIndices_CBW40;
};

/**
 * The CSI Segment data
 */
class CSISegment : public AbstractPicoScenesFrameSegment {
public:
    CSISegment();

    CSISegment(const uint8_t *buffer, uint32_t bufferLength);

    [[nodiscard]] std::string toString() const override;

    CSI &getCSI();

    [[nodiscard]] const CSI &getCSI() const;

    void setCSI(const CSI &csi);

private:
    static std::map<uint16_t, std::function<std::optional<CSI>(const uint8_t *, uint32_t)>> versionedSolutionMap;

    static std::map<uint16_t, std::function<std::optional<CSI>(const uint8_t *, uint32_t)>> initializeSolutionMap() noexcept;

    CSI csi;
};

std::ostream &operator<<(std::ostream &os, const CSISegment &csiSegment);

#endif //PICOSCENES_PLATFORM_CSISEGMENT_HXX
