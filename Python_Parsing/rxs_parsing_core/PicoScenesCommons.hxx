//
// Created by Zhiping Jiang on 2020/11/6.
//

#ifndef PICOSCENES_PLATFORM_PICOSCENESCOMMONS_HXX
#define PICOSCENES_PLATFORM_PICOSCENESCOMMONS_HXX

#include <iostream>
#include <vector>
#include <map>
#include <array>
#include <string>
#include <complex>
#include <numeric>
#include <mutex>
#include <bitset>
#include <iomanip>
#include "SignalMatrix.hxx"
#include "SDRResamplingPreset.hxx"

#ifdef BUILD_WITH_MEX
#include "mex.h"
#define printf mexPrintf
#endif

using ComplexData = std::complex<double>;
using ComplexFloatData = std::complex<float>;
using ComplexArray = std::vector<ComplexData>;
using ComplexFloatArray = std::vector<ComplexData>;
using Uint8Vector = std::vector<uint8_t>;

/**
 * @brief PicoScenes supported device type
 * @see PicoScenesDeviceSubtype
 */
enum class PicoScenesDeviceType : uint16_t {
    QCA9300 = 0x9300, ///< QCA9300 NIC
    IWL5300 = 0x5300, ///< IWL5300 NIC
    IWLMVM_AX200 = 0x2000, ///< AX200 NIC
    IWLMVM_AX210 = 0x2100, ///< AX210 NIC
    MAC80211Compatible = 0x802, ///< All packet-injectable NIC models
    USRP = 0x1234, ///< USRP
    VirtualSDR = 0x1000, ///< Virtual SDR
    SoapySDR = 0x24D2, ///< SoapySDR supported SDR devices
    Unknown = 0x404, ///< Unknown
};

/**
 * Check the specified device type is Intel MVM-based NIC (AX200 or AX210)
 * @return
 */
bool isIntelMVMTypeNIC(PicoScenesDeviceType psdt);

/**
 * Check the specified device type is COTS Wi-Fi NICS (210/200/9300/5300/802)
 * @return
 */
bool isCOTSNIC(PicoScenesDeviceType psdt);

/**
 * Check the specified device type is SDR (USRP or SoapySDR)
 * @param psdt
 * @return
 */
bool isSDR(PicoScenesDeviceType psdt);

/**
 * Return the device type string for the given PicoScenesDeviceType type
 * @return
 */
std::string DeviceType2String(PicoScenesDeviceType type);

std::ostream &operator<<(std::ostream &os, const PicoScenesDeviceType &deviceType);

/**
 * PicoScenes supported device sub-type
 * @see PicoScenesDeviceType
 * @note Never try to alter the subtype order
 */
enum class PicoScenesDeviceSubtype : uint16_t {
    Unknown = 0,
    QCA9300,
    IWL5300,
    AX200,
    AX210,
    VirtualSDR,
    USRP_N2x0, // usrp2
    USRP_B100, // b100
    USRP_B2x0, // b200
    USRP_E3x0, // e3x0
    USRP_N3x0, // n3xx
    USRP_X3x0, // x300
    USRP_X4x0, // x4xx
    HackRFOne,
    LimeSDR,
    BladeRF,
    SOAPYSDR_UHD,
    AX211,
};

/**
 * Return the device sub-type string
 * @param subtype
 * @return A string with identical enum name
 */
std::string DeviceSubtype2String(PicoScenesDeviceSubtype subtype);

std::ostream &operator<<(std::ostream &os, const PicoScenesDeviceSubtype &deviceSubtype);

/**
 * PicoScenes supported packet format
 */
enum class PacketFormatEnum : int8_t {
    PacketFormat_NonHT = 0, ///< 11a or 11g
    PacketFormat_HT = 1, ///< 11n or Wi-Fi 4
    PacketFormat_VHT = 2, ///< 11ac or Wi-Fi 5
    PacketFormat_HESU = 3, ///< 11ax Single-User
    PacketFormat_HEMU = 4, ///< 11ax Multi-User
    PacketFormat_HETB = 5, ///< 11ax Trigger-Based
    PacketFormat_EHTMU = 6, ///< 11be Multi-User
    PacketFormat_EHTTB = 7, ///< 11be Trigger-Based
    PacketFormat_Unknown = -1
};


std::string PacketFormat2String(PacketFormatEnum format);

std::ostream &operator<<(std::ostream &os, const PacketFormatEnum &format);

/**
 * Tx/Rx Channel Bandwidth (CBW) parameter
 * @note CBW doesn't necessarily equals to hardware sampling rate.
 */
enum class ChannelBandwidthEnum : uint16_t {
    CBW_20 = 20, ///< 20 MHz
    CBW_40 = 40, ///< 40 MHz
    CBW_80 = 80, ///< 80 MHz
    CBW_160 = 160, ///< 160 MHz
    CBW_320 = 320,  ///< 320 MHz
};

std::string ChannelBandwidth2String(ChannelBandwidthEnum cbw);

std::ostream &operator<<(std::ostream &os, const ChannelBandwidthEnum &cbw);

/**
 * 802.11n Channel Mode
 * @deprecated
 */
enum class ChannelModeEnum : uint8_t {
    HT20 = 8,
    HT40_MINUS = 24,
    HT40_PLUS = 40,
};

std::string channelModel2String(ChannelModeEnum mode);

std::ostream &operator<<(std::ostream &os, const ChannelModeEnum &channelMode);

/**
 * The guarding interval (ns) between successive OFDM symbols
 */
enum class GuardIntervalEnum : uint16_t {
    GI_400 = 400, ///< 802.11n/ac "Short Guard Interval"
    GI_800 = 800,
    GI_1600 = 1600,
    GI_3200 = 3200
};


std::string GuardInterval2String(GuardIntervalEnum gi);

std::ostream &operator<<(std::ostream &os, const GuardIntervalEnum &gi);

/**
 * Wi-Fi Coding schemes, BCC or LDPC
 */
enum class ChannelCodingEnum : uint8_t {
    BCC = 0,
    LDPC = 1,
};

std::string ChannelCoding2String(ChannelCodingEnum coding);

std::ostream &operator<<(std::ostream &os, const ChannelCodingEnum &coding);

#endif //PICOSCENES_PLATFORM_PICOSCENESCOMMONS_HXX
