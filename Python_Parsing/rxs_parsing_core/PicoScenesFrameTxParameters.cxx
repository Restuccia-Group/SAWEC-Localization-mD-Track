//
// Created by Zhiping Jiang on 2020/11/6.
//

#include "PicoScenesFrameTxParameters.hxx"

std::string PicoScenesFrameTxParameters::toString() const {
    std::stringstream ss;
    ss << "tx_param[preset=" << (preset ? preset->presetName : "NULL") << ", type=" << PacketFormat2String(frameType) << ", CBW=" << ChannelBandwidth2String(cbw) << ", MCS=" << std::to_string(mcs[0]) << ", numSTS=" << int(numSTS[0]) << ", Coding=" << ChannelCoding2String(coding[0]) << ", GI=" << GuardInterval2String(guardInterval);
    if (frameType == PacketFormatEnum::PacketFormat_HT)
        ss << ", numESS=" << int(numExtraSounding) << ", sounding(11n)=" << forceSounding;
    ss << (preciseTxTime ? ", TxTime=" + std::to_string(*preciseTxTime) : "") << "]";
    return ss.str();
}

void PicoScenesFrameTxParameters::validate() const {
    if (frameType == PacketFormatEnum::PacketFormat_NonHT) {

        if (NDPFrame)
            throw std::invalid_argument("NDP frame is only valid for VHT and HE-SU frame.");

        if (int(cbw) > int(ChannelBandwidthEnum::CBW_20))
            throw std::invalid_argument("Invalid Tx CBW: " + ChannelBandwidth2String(cbw) + " for NonHT frame.");

        if (coding[0] == ChannelCodingEnum::LDPC)
            throw std::invalid_argument("Invalid Tx Coding: " + ChannelCoding2String(coding[0]) + " for NonHT frame.");

        if (mcs[0] > 6)
            throw std::invalid_argument("Invalid Tx MCS: " + std::to_string(mcs[0]) + " for NonHT frame.");

        if (numSTS[0] > 1)
            throw std::invalid_argument("Invalid Tx numSTS: " + std::to_string(numSTS[0]) + " for NonHT frame.");

        if (guardInterval != GuardIntervalEnum::GI_800)
            throw std::invalid_argument("Invalid Tx GI: " + GuardInterval2String(guardInterval) + " for NonHT frame.");

    } else if (frameType == PacketFormatEnum::PacketFormat_HT) {

        if (NDPFrame)
            throw std::invalid_argument("NDP frame is only valid for VHT and HE-SU frame.");

        if (int(cbw) > int(ChannelBandwidthEnum::CBW_40))
            throw std::invalid_argument("Invalid Tx CBW: " + ChannelBandwidth2String(cbw) + " for HT frame.");

        if (mcs[0] > 7)
            throw std::invalid_argument("Invalid Tx MCS: " + std::to_string(mcs[0]) + " for HT frame.");

        if (numSTS[0] > 4)
            throw std::invalid_argument("Invalid Tx numSTS: " + std::to_string(numSTS[0]) + " for HT frame.");

        if (guardInterval > GuardIntervalEnum::GI_800)
            throw std::invalid_argument("Invalid Tx GI: " + GuardInterval2String(guardInterval) + " for HT frame.");

    } else if (frameType == PacketFormatEnum::PacketFormat_VHT) {

        if (mcs[0] > 9)
            throw std::invalid_argument("Invalid Tx MCS: " + std::to_string(mcs[0]) + " for VHT frame.");

        if (numSTS[0] > 8)
            throw std::invalid_argument("Invalid Tx numSTS: " + std::to_string(numSTS[0]) + " for VHT frame.");

        if (guardInterval > GuardIntervalEnum::GI_800)
            throw std::invalid_argument("Invalid Tx GI: " + GuardInterval2String(guardInterval) + " for VHT frame.");

    } else if (frameType == PacketFormatEnum::PacketFormat_HESU) {

        if (coding[0] == ChannelCodingEnum::BCC && cbw > ChannelBandwidthEnum::CBW_20)
            throw std::invalid_argument("Invalid Tx Coding: " + ChannelCoding2String(coding[0]) + " for HE-SU frame.");

        if (mcs[0] > 11)
            throw std::invalid_argument("Invalid Tx MCS: " + std::to_string(mcs[0]) + " for HE-SU frame.");

        if (numSTS[0] > 8)
            throw std::invalid_argument("Invalid Tx numSTS: " + std::to_string(numSTS[0]) + " for HE-SU frame.");

        if (txHEExtendedRange && numSTS[0] > 2)
            throw std::invalid_argument("Invalid Tx numSTS: " + std::to_string(numSTS[0]) + " for HE-EXT-SU frame. Must be smaller than 3.");

        if (txHEExtendedRange && mcs[0] > 2)
            throw std::invalid_argument("Invalid Tx MCS: " + std::to_string(mcs[0]) + " for HE-EXT-SU frame. Must be smaller than 3.");

        if (heHighDoppler && numSTS[0] > 4)
            throw std::invalid_argument("Invalid Tx numSTS: " + std::to_string(numSTS[0]) + " for HE-SU High-Doppler frame. Must be smaller than 4.");

        if (heHighDoppler && heMidamblePeriodicity != 10 && heMidamblePeriodicity != 20)
            throw std::invalid_argument("Invalid TX mid amble periodicity: " + std::to_string(heMidamblePeriodicity) + " for HE-SU frame. Must be 10 or 20.");
    } else if (frameType == PacketFormatEnum::PacketFormat_HEMU) {
        // TODO add this section
    }
}

std::ostream &operator<<(std::ostream &os, const PicoScenesFrameTxParameters &parameters) {
    os << parameters.toString();
    return os;
}
