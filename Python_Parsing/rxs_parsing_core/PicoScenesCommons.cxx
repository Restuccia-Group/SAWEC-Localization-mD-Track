//
// Created by Zhiping Jiang on 2020/11/6.
//

#include "PicoScenesCommons.hxx"

bool isIntelMVMTypeNIC(PicoScenesDeviceType psdt) {
    return psdt == PicoScenesDeviceType::IWLMVM_AX200 || psdt == PicoScenesDeviceType::IWLMVM_AX210;
}

bool isCOTSNIC(PicoScenesDeviceType psdt) {
    return isIntelMVMTypeNIC(psdt) || psdt == PicoScenesDeviceType::QCA9300 || psdt == PicoScenesDeviceType::IWL5300 || psdt == PicoScenesDeviceType::MAC80211Compatible;
}

bool isSDR(PicoScenesDeviceType psdt) {
    return psdt == PicoScenesDeviceType::USRP || psdt == PicoScenesDeviceType::SoapySDR;
}

std::string DeviceType2String(PicoScenesDeviceType type) {
    switch (type) {
        case PicoScenesDeviceType::QCA9300:
            return "QCA9300";
        case PicoScenesDeviceType::IWL5300:
            return "IWL5300";
        case PicoScenesDeviceType::IWLMVM_AX200:
            return "AX200";
        case PicoScenesDeviceType::IWLMVM_AX210:
            return "AX210";
        case PicoScenesDeviceType::MAC80211Compatible:
            return "MAC80211 Compatible NIC";
        case PicoScenesDeviceType::USRP:
            return "USRP(SDR)";
        case PicoScenesDeviceType::VirtualSDR:
            return "Virtual(SDR)";
        case PicoScenesDeviceType::SoapySDR:
            return "SoapySDR Compatible SDR";
        case PicoScenesDeviceType::Unknown:
            return "Unknown";
        default:
            throw std::runtime_error("unrecognized PicoScenesDeviceType.");
    }
}

std::string DeviceSubtype2String(PicoScenesDeviceSubtype subtype) {
    switch (subtype) {
        case PicoScenesDeviceSubtype::Unknown:
            return "Unknown";
        case PicoScenesDeviceSubtype::QCA9300:
            return "QCA9300";
        case PicoScenesDeviceSubtype::IWL5300:
            return "IWL5300";
        case PicoScenesDeviceSubtype::AX200:
            return "AX200";
        case PicoScenesDeviceSubtype::AX210:
            return "AX210";
        case PicoScenesDeviceSubtype::AX211:
            return "AX211";
        case PicoScenesDeviceSubtype::USRP_N2x0:
            return "USRP N2x0";
        case PicoScenesDeviceSubtype::USRP_B2x0:
            return "USRP B2x0";
        case PicoScenesDeviceSubtype::USRP_E3x0:
            return "USRP E3x0";
        case PicoScenesDeviceSubtype::USRP_N3x0:
            return "USRP N3x0";
        case PicoScenesDeviceSubtype::USRP_X3x0:
            return "USRP X3x0";
        case PicoScenesDeviceSubtype::USRP_X4x0:
            return "USRP X4x0";
        case PicoScenesDeviceSubtype::HackRFOne:
            return "HackRF One";
        case PicoScenesDeviceSubtype::LimeSDR:
            return "LimeSDR";
        case PicoScenesDeviceSubtype::BladeRF:
            return "BladeRF";
        case PicoScenesDeviceSubtype::SOAPYSDR_UHD:
            return "SoapySDR_UHD";
        default:
            throw std::runtime_error("unrecognized PicoScenesDeviceSubtype.");
    }
}

std::string PacketFormat2String(PacketFormatEnum format) {
    switch (format) {
        case PacketFormatEnum::PacketFormat_NonHT:
            return "NonHT";
        case PacketFormatEnum::PacketFormat_HT:
            return "HT";
        case PacketFormatEnum::PacketFormat_VHT:
            return "VHT";
        case PacketFormatEnum::PacketFormat_HESU:
            return "HE-SU";
        case PacketFormatEnum::PacketFormat_HEMU:
            return "HE-MU";
        case PacketFormatEnum::PacketFormat_HETB:
            return "HE-TB";
        case PacketFormatEnum::PacketFormat_EHTMU:
            return "EHT-MU";
        case PacketFormatEnum::PacketFormat_EHTTB:
            return "EHT-TB";
        case PacketFormatEnum::PacketFormat_Unknown:
            return "Unknown";
    }

    throw std::runtime_error("Unsupported packet format.");
}

std::string ChannelBandwidth2String(ChannelBandwidthEnum cbw) {
    switch (cbw) {
        case ChannelBandwidthEnum::CBW_20:
            return "20";
        case ChannelBandwidthEnum::CBW_40:
            return "40";
        case ChannelBandwidthEnum::CBW_80:
            return "80";
        case ChannelBandwidthEnum::CBW_160:
            return "160";
        case ChannelBandwidthEnum::CBW_320:
            return "320";
        default:
            return "Unknown Channel Bandwidth (CBW)";
    }
}

std::string channelModel2String(ChannelModeEnum mode) {
    switch (mode) {
        case ChannelModeEnum::HT40_PLUS:
            return "HT40_PLUS";
        case ChannelModeEnum::HT40_MINUS:
            return "HT40_MINUS";
        case ChannelModeEnum::HT20:
            return "HT20";
    }
    return "channel mode error.";
}

std::string GuardInterval2String(GuardIntervalEnum gi) {
    switch (gi) {
        case GuardIntervalEnum::GI_400:
            return "0.4us";
        case GuardIntervalEnum::GI_800:
            return "0.8us";
        case GuardIntervalEnum::GI_1600:
            return "1.6us";
        case GuardIntervalEnum::GI_3200:
            return "3.2us";
        default:
            throw std::runtime_error("Unsupported GuardIntervalEnum...");
    }
}

std::string ChannelCoding2String(ChannelCodingEnum coding) {
    switch (coding) {
        case ChannelCodingEnum::LDPC:
            return "LDPC";
        case ChannelCodingEnum::BCC:
            return "BCC";
        default:
            return "Unknown";
    }
}

std::ostream &operator<<(std::ostream &os, const PicoScenesDeviceType &deviceType) {
    os << DeviceType2String(deviceType);
    return os;
}

std::ostream &operator<<(std::ostream &os, const PacketFormatEnum &format) {
    os << PacketFormat2String(format);
    return os;
}

std::ostream &operator<<(std::ostream &os, const ChannelBandwidthEnum &cbw) {
    os << ChannelBandwidth2String(cbw);
    return os;
}

std::ostream &operator<<(std::ostream &os, const ChannelModeEnum &channelMode) {
    os << channelModel2String(channelMode);
    return os;
}

std::ostream &operator<<(std::ostream &os, const GuardIntervalEnum &gi) {
    os << GuardInterval2String(gi);
    return os;
}

std::ostream &operator<<(std::ostream &os, const ChannelCodingEnum &coding) {
    os << ChannelCoding2String(coding);
    return os;
}

std::ostream &operator<<(std::ostream &os, const PicoScenesDeviceSubtype &deviceSubtype) {
    os << DeviceSubtype2String(deviceSubtype);
    return os;
}