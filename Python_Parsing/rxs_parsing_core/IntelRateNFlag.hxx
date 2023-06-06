//
// Created by Zhiping Jiang on 8/21/22.
//

#ifndef PICOSCENES_PLATFORM_INTELRATENFLAG_HXX
#define PICOSCENES_PLATFORM_INTELRATENFLAG_HXX

#include "PicoScenesCommons.hxx"

/**
 * @brief Hold and parse the Intel MVM rate_n_flag value
 */
class IntelRateNFlag {
public:
    explicit IntelRateNFlag(uint32_t rateNFlag) : rateNFlag(rateNFlag) {}

    const uint32_t rateNFlag;

    inline uint8_t getMCSPerStream() const {
        return rateNFlag & 0xFU;
    }

    inline uint8_t getSTS() const {
        return (rateNFlag >> 17 & 1U) == 0 ? 1 : 2;
    }

    inline PacketFormatEnum getFormat() const {
        switch (rateNFlag >> 8 & 7U) {
            case 0:
                [[fallthrough]];
            case 1:
                return PacketFormatEnum::PacketFormat_NonHT;
            case 2:
                return PacketFormatEnum::PacketFormat_HT;
            case 3:
                return PacketFormatEnum::PacketFormat_VHT;
            case 4:
                switch (rateNFlag >> 23 & 3U) {
                    case 0:
                        [[fallthrough]];
                    case 1:
                        return PacketFormatEnum::PacketFormat_HESU;
                    case 2:
                        return PacketFormatEnum::PacketFormat_HEMU;
                    case 3:
                        return PacketFormatEnum::PacketFormat_HETB;
                }
            case 5:
                return PacketFormatEnum::PacketFormat_EHTMU;
        }
        return PacketFormatEnum::PacketFormat_NonHT;
    }

    inline ChannelBandwidthEnum getChannelBandwidth() const {
        switch (rateNFlag >> 11 & 7U) {
            case 0:
                return ChannelBandwidthEnum::CBW_20;
            case 1:
                return ChannelBandwidthEnum::CBW_40;
            case 2:
                return ChannelBandwidthEnum::CBW_80;
            case 3:
                return ChannelBandwidthEnum::CBW_160;
            case 4:
                return ChannelBandwidthEnum::CBW_320;
        }
        return ChannelBandwidthEnum::CBW_20;
    }

    inline uint8_t activeRxChainmask() const {
        return rateNFlag >> 14 & 3U;
    }

    inline ChannelCodingEnum getCodingScheme() const {
        if (rateNFlag >> 16 & 1U)
            return ChannelCodingEnum::LDPC;
        return ChannelCodingEnum::BCC;
    }

    inline bool isBeamforming() const {
        return rateNFlag >> 19 & 1U;
    }

    inline GuardIntervalEnum getGuardInterval() const {
        auto format = getFormat();
        if (format == PacketFormatEnum::PacketFormat_HT || format == PacketFormatEnum::PacketFormat_VHT) {
            return (rateNFlag >> 20 & 1U) == 0 ? GuardIntervalEnum::GI_800 : GuardIntervalEnum::GI_400;
        }

        if (format == PacketFormatEnum::PacketFormat_HESU || format == PacketFormatEnum::PacketFormat_HEMU) {
            switch (rateNFlag >> 20 & 7U) {
                case 0:
                    [[fallthrough]];
                case 1:
                    return GuardIntervalEnum::GI_800;
                case 2:
                    return GuardIntervalEnum::GI_1600;
                case 3:
                    return GuardIntervalEnum::GI_3200;
                case 4:
                    return GuardIntervalEnum::GI_800;
            }
        }

        if (format == PacketFormatEnum::PacketFormat_HETB) {
            switch (rateNFlag >> 20 & 7U) {
                case 0:
                    [[fallthrough]];
                case 1:
                    return GuardIntervalEnum::GI_1600;
                case 2:
                    return GuardIntervalEnum::GI_3200;
            }
        }

        return GuardIntervalEnum::GI_800;
    }
};


#endif //PICOSCENES_PLATFORM_INTELRATENFLAG_HXX
