//
// Created by Zhiping Jiang on 11/27/22.
//

#include "FrontEndModePreset.hxx"

const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::TX_CBW_160_HESU = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("TX_CBW_160_HESU", PacketFormatEnum::PacketFormat_HESU, ChannelCodingEnum::LDPC, 160, 200, std::nullopt, std::nullopt, "Equivalent to \"--format hesu --coding ldpc --rate 200e6 --tx-resample-ratio 1.25 --cbw 160\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::TX_CBW_160_VHT = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("TX_CBW_160_VHT", PacketFormatEnum::PacketFormat_VHT, ChannelCodingEnum::BCC, 160, 200, std::nullopt, std::nullopt, "Equivalent to \"--format vht --coding bcc --rate 200e6 --tx-resample-ratio 1.25 --cbw 160\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::TX_CBW_160_VHT_LDPC = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("TX_CBW_160_VHT_LDPC", PacketFormatEnum::PacketFormat_VHT, ChannelCodingEnum::LDPC, 160, 200, std::nullopt, std::nullopt, "Equivalent to \"--format vht --coding ldpc --rate 200e6 --tx-resample-ratio 1.25 --cbw 160\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::TX_CBW_80_HESU = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("TX_CBW_80_HESU", PacketFormatEnum::PacketFormat_HESU, ChannelCodingEnum::LDPC, 80, 100, std::nullopt, std::nullopt, "Equivalent to \"--format hesu --coding ldpc --rate 100e6 --tx-resample-ratio 1.25 --cbw 80\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::TX_CBW_80_VHT = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("TX_CBW_80_VHT", PacketFormatEnum::PacketFormat_VHT, ChannelCodingEnum::BCC, 80, 100, std::nullopt, std::nullopt, "Equivalent to \"--format vht --coding bcc --rate 100e6 --tx-resample-ratio 1.25 --cbw 80\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::TX_CBW_80_VHT_LDPC = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("TX_CBW_80_VHT_LDPC", PacketFormatEnum::PacketFormat_VHT, ChannelCodingEnum::LDPC, 80, 100, std::nullopt, std::nullopt, "Equivalent to \"--format vht --coding ldpc --rate 100e6 --tx-resample-ratio 1.25 --cbw 80\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::TX_CBW_40_HESU = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("TX_CBW_40_HESU", PacketFormatEnum::PacketFormat_HESU, ChannelCodingEnum::LDPC, 40, 40, std::nullopt, std::nullopt, "Equivalent to \"--format hesu --coding ldpc --rate 40e6 --tx-resample-ratio 1.0 --cbw 40\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::TX_CBW_40_VHT = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("TX_CBW_40_VHT", PacketFormatEnum::PacketFormat_VHT, ChannelCodingEnum::BCC, 40, 40, std::nullopt, std::nullopt, "Equivalent to \"--format vht --coding bcc --rate 40e6 --tx-resample-ratio 1.0 --cbw 40\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::TX_CBW_40_VHT_LDPC = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("TX_CBW_40_VHT_LDPC", PacketFormatEnum::PacketFormat_VHT, ChannelCodingEnum::LDPC, 40, 40, std::nullopt, std::nullopt, "Equivalent to \"--format vht --coding ldpc --rate 40e6 --tx-resample-ratio 1.0 --cbw 40\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::TX_CBW_40_HT = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("TX_CBW_40_HT", PacketFormatEnum::PacketFormat_HT, ChannelCodingEnum::BCC, 40, 40, std::nullopt, std::nullopt, "Equivalent to \"--format ht --coding bcc --rate 40e6 --tx-resample-ratio 1.0 --cbw 40\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::TX_CBW_40_HT_LDPC = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("TX_CBW_40_HT_LDPC", PacketFormatEnum::PacketFormat_HT, ChannelCodingEnum::LDPC, 40, 40, std::nullopt, std::nullopt, "Equivalent to \"--format ht --coding ldpc --rate 40e6 --tx-resample-ratio 1.0 --cbw 40\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::TX_CBW_20_HESU = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("TX_CBW_20_HESU", PacketFormatEnum::PacketFormat_HESU, ChannelCodingEnum::BCC, 20, 20, std::nullopt, std::nullopt, "Equivalent to \"--format hesu --coding ldpc --rate 20e6 --tx-resample-ratio 1.0 --cbw 20\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::TX_CBW_20_HESU_LDPC = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("TX_CBW_20_HESU_LDPC", PacketFormatEnum::PacketFormat_HESU, ChannelCodingEnum::LDPC, 20, 20, std::nullopt, std::nullopt, "Equivalent to \"--format hesu --coding ldpc --rate 20e6 --tx-resample-ratio 1.0 --cbw 20\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::TX_CBW_20_VHT = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("TX_CBW_20_VHT", PacketFormatEnum::PacketFormat_VHT, ChannelCodingEnum::BCC, 20, 20, std::nullopt, std::nullopt, "Equivalent to \"--format vht --coding bcc --rate 20e6 --tx-resample-ratio 1.0 --cbw 20\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::TX_CBW_20_VHT_LDPC = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("TX_CBW_20_VHT_LDPC", PacketFormatEnum::PacketFormat_VHT, ChannelCodingEnum::LDPC, 20, 20, std::nullopt, std::nullopt, "Equivalent to \"--format vht --coding ldpc --rate 20e6 --tx-resample-ratio 1.0 --cbw 20\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::TX_CBW_20_HT = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("TX_CBW_20_HT", PacketFormatEnum::PacketFormat_HT, ChannelCodingEnum::BCC, 20, 20, std::nullopt, std::nullopt, "Equivalent to \"--format ht --coding bcc --rate 20e6 --tx-resample-ratio 1.0 --cbw 20\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::TX_CBW_20_HT_LDPC = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("TX_CBW_20_HT_LDPC", PacketFormatEnum::PacketFormat_HT, ChannelCodingEnum::LDPC, 20, 20, std::nullopt, std::nullopt, "Equivalent to \"--format ht --coding ldpc --rate 20e6 --tx-resample-ratio 1.0 --cbw 20\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::TX_CBW_20_NonHT = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("TX_CBW_20_NONHT", PacketFormatEnum::PacketFormat_NonHT, ChannelCodingEnum::BCC, 20, 20, std::nullopt, std::nullopt, "Equivalent to \"--format nonht --coding bcc --rate 20e6 --tx-resample-ratio 1.0 --cbw 20\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::RX_CBW_160 = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("RX_CBW_160", PacketFormatEnum::PacketFormat_Unknown, ChannelCodingEnum::BCC, std::nullopt, std::nullopt, 160, 200, "Equivalent to \"--rate 200e6 --rx-resample-ratio 0.8 --rx-cbw 160\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::RX_CBW_80 = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("RX_CBW_80", PacketFormatEnum::PacketFormat_Unknown, ChannelCodingEnum::BCC, std::nullopt, std::nullopt, 80, 100, "Equivalent to \"--rate 100e6 --rx-resample-ratio 0.8 --rx-cbw 80\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::RX_CBW_40 = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("RX_CBW_40", PacketFormatEnum::PacketFormat_Unknown, ChannelCodingEnum::BCC, std::nullopt, std::nullopt, 40, 40, "Equivalent to \"--rate 40e6 --rx-resample-ratio 1.0 --rx-cbw 40\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::RX_CBW_20 = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("RX_CBW_20", PacketFormatEnum::PacketFormat_Unknown, ChannelCodingEnum::BCC, std::nullopt, std::nullopt, 20, 20, "Equivalent to \"--rate 20e6 --rx-resample-ratio 1.0 --rx-cbw 20\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::TR_CBW_160_HESU = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("TR_CBW_160_HESU", PacketFormatEnum::PacketFormat_HESU, ChannelCodingEnum::LDPC, 160, 200, 160, 200, "Equivalent to \"--format hesu --coding ldpc --rate 200e6 --tx-resample-ratio 1.25 --cbw 160 --rx-resample-ratio 0.8 --rx-cbw 160\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::TR_CBW_160_VHT = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("TR_CBW_160_VHT", PacketFormatEnum::PacketFormat_VHT, ChannelCodingEnum::BCC, 160, 200, 160, 200, "Equivalent to \"--format vht --coding bcc --rate 200e6 --tx-resample-ratio 1.25 --cbw 160 --rx-resample-ratio 0.8 --rx-cbw 160\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::TR_CBW_160_VHT_LDPC = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("TR_CBW_160_VHT_LDPC", PacketFormatEnum::PacketFormat_VHT, ChannelCodingEnum::LDPC, 160, 200, 160, 200, "Equivalent to \"--format vht --coding ldpc --rate 200e6 --tx-resample-ratio 1.25 --cbw 160 --rx-resample-ratio 0.8 --rx-cbw 160\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::TR_CBW_80_HESU = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("TR_CBW_80_HESU", PacketFormatEnum::PacketFormat_HESU, ChannelCodingEnum::LDPC, 80, 100, 80, 100, "Equivalent to \"--format hesu --coding ldpc --rate 100e6 --tx-resample-ratio 1.25 --cbw 80 --rx-resample-ratio 0.8 --rx-cbw 80\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::TR_CBW_80_VHT = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("TR_CBW_80_VHT", PacketFormatEnum::PacketFormat_VHT, ChannelCodingEnum::BCC, 80, 100, 80, 100, "Equivalent to \"--format vht --coding bcc --rate 100e6 --tx-resample-ratio 1.25 --cbw 80 --rx-resample-ratio 0.8 --rx-cbw 80\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::TR_CBW_80_VHT_LDPC = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("TR_CBW_80_VHT_LDPC", PacketFormatEnum::PacketFormat_VHT, ChannelCodingEnum::LDPC, 80, 100, 80, 100, "Equivalent to \"--format vht --coding ldpc --rate 100e6 --tx-resample-ratio 1.25 --cbw 80 --rx-resample-ratio 0.8 --rx-cbw 80\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::TR_CBW_40_HESU = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("TR_CBW_40_HESU", PacketFormatEnum::PacketFormat_HESU, ChannelCodingEnum::LDPC, 40, 40, 40, 40, "Equivalent to \"--format hesu --coding ldpc --rate 40e6 --tx-resample-ratio 1.0 --cbw 40 --rx-resample-ratio 1.0 --rx-cbw 40\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::TR_CBW_40_VHT = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("TR_CBW_40_VHT", PacketFormatEnum::PacketFormat_VHT, ChannelCodingEnum::BCC, 40, 40, 40, 40, "Equivalent to \"--format vht --coding bcc --rate 40e6 --tx-resample-ratio 1.0 --cbw 40 --rx-resample-ratio 1.0 --rx-cbw 40\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::TR_CBW_40_VHT_LDPC = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("TR_CBW_40_VHT_LDPC", PacketFormatEnum::PacketFormat_VHT, ChannelCodingEnum::LDPC, 40, 40, 40, 40, "Equivalent to \"--format vht --coding ldpc --rate 40e6 --tx-resample-ratio 1.0 --cbw 40 --rx-resample-ratio 1.0 --rx-cbw 40\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::TR_CBW_40_HT = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("TR_CBW_40_HT", PacketFormatEnum::PacketFormat_HT, ChannelCodingEnum::BCC, 40, 40, 40, 40, "Equivalent to \"--format ht --coding bcc --rate 40e6 --tx-resample-ratio 1.0 --cbw 40 --rx-resample-ratio 1.0 --rx-cbw 40\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::TR_CBW_40_HT_LDPC = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("TR_CBW_40_HT_LDPC", PacketFormatEnum::PacketFormat_HT, ChannelCodingEnum::LDPC, 40, 40, 40, 40, "Equivalent to \"--format ht --coding ldpc --rate 40e6 --tx-resample-ratio 1.0 --cbw 40 --rx-resample-ratio 1.0 --rx-cbw 40\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::TR_CBW_20_HESU = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("TR_CBW_20_HESU", PacketFormatEnum::PacketFormat_HESU, ChannelCodingEnum::BCC, 20, 20, 20, 20, "Equivalent to \"--format hesu --coding ldpc --rate 20e6 --tx-resample-ratio 1.0 --cbw 20 --rx-resample-ratio 1.0 --rx-cbw 20\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::TR_CBW_20_HESU_LDPC = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("TR_CBW_20_HESU_LDPC", PacketFormatEnum::PacketFormat_HESU, ChannelCodingEnum::LDPC, 20, 20, 20, 20, "Equivalent to \"--format hesu --coding ldpc --rate 20e6 --tx-resample-ratio 1.0 --cbw 20 --rx-resample-ratio 1.0 --rx-cbw 20\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::TR_CBW_20_VHT = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("TR_CBW_20_VHT", PacketFormatEnum::PacketFormat_VHT, ChannelCodingEnum::BCC, 20, 20, 20, 20, "Equivalent to \"--format vht --coding bcc --rate 20e6 --tx-resample-ratio 1.0 --cbw 20 --rx-resample-ratio 1.0 --rx-cbw 20\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::TR_CBW_20_VHT_LDPC = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("TR_CBW_20_VHT_LDPC", PacketFormatEnum::PacketFormat_VHT, ChannelCodingEnum::LDPC, 20, 20, 20, 20, "Equivalent to \"--format vht --coding ldpc --rate 20e6 --tx-resample-ratio 1.0 --cbw 20 --rx-resample-ratio 1.0 --rx-cbw 20\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::TR_CBW_20_HT = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("TR_CBW_20_HT", PacketFormatEnum::PacketFormat_HT, ChannelCodingEnum::BCC, 20, 20, 20, 20, "Equivalent to \"--format ht --coding bcc --rate 20e6 --tx-resample-ratio 1.0 --cbw 20 --rx-resample-ratio 1.0 --rx-cbw 20\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::TR_CBW_20_HT_LDPC = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("TR_CBW_20_HT_LDPC", PacketFormatEnum::PacketFormat_HT, ChannelCodingEnum::LDPC, 20, 20, 20, 20, "Equivalent to \"--format ht --coding ldpc --rate 20e6 --tx-resample-ratio 1.0 --cbw 20 --rx-resample-ratio 1.0 --rx-cbw 20\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::TR_CBW_20_NonHT = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("TR_CBW_20_NONHT", PacketFormatEnum::PacketFormat_NonHT, ChannelCodingEnum::BCC, 20, 20, 20, 20, "Equivalent to \"--format nonht --coding bcc --rate 20e6 --tx-resample-ratio 1.0 --cbw 20 --rx-resample-ratio 1.0 --rx-cbw 20\""));
const std::shared_ptr<FrontEndModePreset> FrontEndModePreset::DEFAULT = std::shared_ptr<FrontEndModePreset>(new FrontEndModePreset("DEFAULT", PacketFormatEnum::PacketFormat_HT, ChannelCodingEnum::BCC, 20, 20, 20, 20, "Equivalent to TR_CBW_20_HT"));

const std::vector<std::shared_ptr<FrontEndModePreset>> &FrontEndModePreset::getPresetList() {
    static std::vector<std::shared_ptr<FrontEndModePreset>> list{
            TX_CBW_160_HESU,
            TX_CBW_160_VHT,
            TX_CBW_160_VHT_LDPC,
            TX_CBW_80_HESU,
            TX_CBW_80_VHT,
            TX_CBW_80_VHT_LDPC,
            TX_CBW_40_HESU,
            TX_CBW_40_VHT,
            TX_CBW_40_VHT_LDPC,
            TX_CBW_40_HT,
            TX_CBW_40_HT_LDPC,
            TX_CBW_20_HESU,
            TX_CBW_20_HESU_LDPC,
            TX_CBW_20_VHT,
            TX_CBW_20_VHT_LDPC,
            TX_CBW_20_HT,
            TX_CBW_20_HT_LDPC,
            TX_CBW_20_NonHT,
            RX_CBW_160,
            RX_CBW_80,
            RX_CBW_40,
            RX_CBW_20,
            TR_CBW_160_HESU,
            TR_CBW_160_VHT,
            TR_CBW_160_VHT_LDPC,
            TR_CBW_80_HESU,
            TR_CBW_80_VHT,
            TR_CBW_80_VHT_LDPC,
            TR_CBW_40_HESU,
            TR_CBW_40_VHT,
            TR_CBW_40_VHT_LDPC,
            TR_CBW_40_HT,
            TR_CBW_40_HT_LDPC,
            TR_CBW_20_HESU,
            TR_CBW_20_HESU_LDPC,
            TR_CBW_20_VHT,
            TR_CBW_20_VHT_LDPC,
            TR_CBW_20_HT,
            TR_CBW_20_HT_LDPC,
            TR_CBW_20_NonHT,
            DEFAULT,
    };

    return list;
}

const std::map<std::string, std::shared_ptr<FrontEndModePreset>> &FrontEndModePreset::getPresetMap() {
    static std::map<std::string, std::shared_ptr<FrontEndModePreset>> presetMap;
    static std::once_flag flag;
    std::call_once(flag, [&] {
        for (const auto &preset: getPresetList())
            presetMap.emplace(preset->presetName, preset);
    });

    return presetMap;
}

FrontEndModePreset::FrontEndModePreset(std::string presetName, PacketFormatEnum format, ChannelCodingEnum coding,
                                       const std::optional<double> &txCbw, const std::optional<double> &txSamplingRate,
                                       const std::optional<double> &rxCbw, const std::optional<double> &rxSamplingRate,
                                       std::string description) :
        presetName(std::move(presetName)), format(format), coding(coding),
        txCBW(txCbw), txSamplingRate(txSamplingRate),
        rxCBW(rxCbw), rxSamplingRate(rxSamplingRate),
        description(std::move(description)),
        txResampleRatio(txCBW && txSamplingRate ? std::make_optional(*txSamplingRate / *txCBW) : std::nullopt),
        rxResampleRatio(rxCBW && rxSamplingRate ? std::make_optional(*rxCBW / *rxSamplingRate) : std::nullopt) {}

std::string FrontEndModePreset::printHelpContentForFrontEndModePreset() {
    std::stringstream ss;
    ss << "\n\nPicoScenes supports the following COTS NIC/SDR frontend mode presets:\n";
    ss << "-------------------------------------------------------------------------------------\n";
    for (const auto &preset: getPresetList())
        ss << std::setw(20) << preset->presetName << ": " << preset->description << "\n";
    ss << "-------------------------------------------------------------------------------------\n\n";

    return ss.str();
}

bool FrontEndModePreset::operator==(const FrontEndModePreset &rhs) const {
    return std::tie(presetName, format, coding, txCBW, txSamplingRate, rxCBW, rxSamplingRate) == std::tie(rhs.presetName, rhs.format, rhs.coding, rhs.txCBW, rhs.txSamplingRate, rhs.rxCBW, rhs.rxSamplingRate);
}

bool FrontEndModePreset::operator<(const FrontEndModePreset &rhs) const {
    return std::tie(presetName, format, coding, txCBW, txSamplingRate, rxCBW, rxSamplingRate) < std::tie(rhs.presetName, rhs.format, rhs.coding, rhs.txCBW, rhs.txSamplingRate, rhs.rxCBW, rhs.rxSamplingRate);
}

std::ostream &operator<<(std::ostream &os, const FrontEndModePreset &preset) {
    os << "presetName: " << preset.presetName << " description: " << preset.description;
    return os;
}
