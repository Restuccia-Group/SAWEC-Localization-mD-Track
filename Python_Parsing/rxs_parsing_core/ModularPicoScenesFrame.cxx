//
// Created by Zhiping Jiang on 2020/11/6.
//

#include <random>
#include <cassert>
#include "ModularPicoScenesFrame.hxx"

std::string ieee80211_mac_frame_header_frame_control_field::getTypeString() const {
    if (type == 0) {
        switch (subtype) {
            case 0:
                return "[MF]Assoc. Req.";
            case 1:
                return "[MF]Assoc. Resp.";
            case 2:
                return "[MF]Re-assoc. Req.";
            case 3:
                return "[MF]Re-assoc. Resp.";
            case 4:
                return "[MF]Probe Req.";
            case 5:
                return "[MF]Probe Resp.";
            case 8:
                return "[MF]Beacon";
            case 9:
                return "[MF]ATIM";
            case 10:
                return "[MF]Disassoc.";
            case 11:
                return "[MF]Authen.";
            case 12:
                return "[MF]Deauthen.";
            case 13:
                return "[MF]Action";
            default:
                return "[MF]Reserved_" + std::to_string(subtype);
        }
    } else if (type == 1) {
        switch (subtype) {
            case 2:
                return "[CF]Trigger";
            case 5:
                return "[CF]VHT/HE NDP Ann.";
            case 7:
                return "[CF]Wrapper";
            case 8:
                return "[CF]BA Req.";
            case 9:
                return "[CF]BA";
            case 10:
                return "[CF]PS-Poll";
            case 11:
                return "[CF]RTS";
            case 12:
                return "[CF]CTS";
            case 13:
                return "[CF]ACK";
            case 14:
                return "[CF]CF-End";
            case 15:
                return "[CF]CF-End+CF-ACK";
            default:
                return "[CF]Reserved_" + std::to_string(subtype);
        }
    } else if (type == 2) {
        switch (subtype) {
            case 0:
                return "[DF]Data";
            case 1:
                return "[DF]Data+CF-ACK";
            case 2:
                return "[DF]Data+CF-Poll";
            case 3:
                return "[DF]Data-CF-ACK+CF-Poll";
            case 4:
                return "[DF]Null";
            case 5:
                return "[DF]CF-ACK";
            case 6:
                return "[DF]CF-Poll";
            case 7:
                return "[DF]CF-ACK+CF-Poll";
            case 8:
                return "[DF]QoS Data";
            case 9:
                return "[DF]QoS Data+CF-ACK";
            case 10:
                return "[DF]QoS Data+CF-Poll";
            case 11:
                return "[DF]QoS Data+CF-ACK+CF-Poll";
            case 12:
                return "[DF]QoS Null";
            case 13:
                return "[DF]Reserved13";
            case 14:
                return "[DF]QoS+CF-Poll(Null)";
            case 15:
                return "[DF]QoS+CF-ACK(Null)";
            default:
                return "[DF]Reserved_" + std::to_string(subtype);
        }
    }

    return "[" + std::to_string(type) + "]Reserved_" + std::to_string(subtype);
}


std::ostream &operator<<(std::ostream &os, const ieee80211_mac_frame_header_frame_control_field &fc) {
    os << fc.getTypeString();
    return os;
}

ieee80211_mac_frame_header ieee80211_mac_frame_header::createFromBuffer(const uint8_t *buffer, std::optional<uint32_t> bufferLength) {
    ieee80211_mac_frame_header result{};
    uint32_t pos = 0;
    result.fc = *((ieee80211_mac_frame_header_frame_control_field *) (buffer + pos));

    if (result.fc.type == 0 || result.fc.type == 2 || result.fc.type == 3) { // Management Frames, Data Frames and all type=3 reserved
        result = *((ieee80211_mac_frame_header *) buffer);
        return result;
    }

    pos += sizeof(ieee80211_mac_frame_header_frame_control_field);
    result.dur = *((uint16_t *) (buffer + pos));
    pos += 2;
    std::copy(buffer + pos, buffer + pos + 6, result.addr1.begin());
    pos += 6;

    switch (result.fc.subtype) {
        case 2:
        case 8:
        case 9:
        case 10:
        case 11:
        case 15:
            std::copy(buffer + pos, buffer + pos + 6, result.addr2.begin());
            pos += 6;
            break;
        default:
            break;
    }

    return result;
}

std::string ieee80211_mac_frame_header::toString() const {
    std::stringstream ss;
    ss << "MACHeader:[type=" << fc << ", ";
    ss << "dest=" << std::nouppercase << std::setfill('0') << std::setw(2) << std::right << std::hex << int(addr1[0]) << ":" << int(addr1[1]) << ":" << int(addr1[2]) << ":" << int(addr1[3]) << ":" << int(addr1[4]) << ":" << int(addr1[5]) << ", ";
    ss << "src=" << std::nouppercase << std::setfill('0') << std::setw(2) << std::right << std::hex << int(addr2[0]) << ":" << int(addr2[1]) << ":" << int(addr2[2]) << ":" << int(addr2[3]) << ":" << int(addr2[4]) << ":" << int(addr2[5]) << ", ";
    ss << "seq=" << std::dec << seq << ", frag=" << frag << ", ";
    ss << "mfrags=" << std::to_string(fc.moreFrags) << "]";
    return ss.str();
}

std::ostream &operator<<(std::ostream &os, const ieee80211_mac_frame_header &header) {
    os << header.toString();
    return os;
}

std::optional<PicoScenesFrameHeader> PicoScenesFrameHeader::fromBuffer(const uint8_t *buffer) {
    auto magicValue = *((uint32_t *) (buffer));
    if (magicValue == 0x20150315) {
        auto frameHeader = *((PicoScenesFrameHeader *) (buffer));
        return frameHeader;
    }
    return std::nullopt;
}

std::string PicoScenesFrameHeader::toString() const {
    std::stringstream ss;
    ss << "PSFHeader:[ver=0x" << std::hex << version << std::dec << ", device=" << deviceType << ", numSegs=" << int(numSegments) << ", type=" << int(frameType) << ", taskId=" << int(taskId) << ", txId=" << int(txId) << "]";
    return ss.str();
}

std::ostream &operator<<(std::ostream &os, const PicoScenesFrameHeader &frameHeader) {
    os << frameHeader.toString();
    return os;
}

std::optional<ModularPicoScenesRxFrame> ModularPicoScenesRxFrame::fromBuffer(const uint8_t *constBuffer, uint32_t bufferLength, bool interpolateCSI) {
    uint32_t pos = 0;
    auto rxFrameHeader = *(ModularPicoScenesRxFrameHeader *) constBuffer;
    if (rxFrameHeader.frameLength + 4 != bufferLength ||
        rxFrameHeader.magicWord != 0x20150315 ||
        rxFrameHeader.frameVersion != 0x1U) {
        return {};
    }
    pos += sizeof(ModularPicoScenesRxFrameHeader);

    auto frame = ModularPicoScenesRxFrame();
    try {
        std::copy(constBuffer, constBuffer + bufferLength, std::back_inserter(frame.rawBuffer));
        auto *buffer = frame.rawBuffer.data();
        frame.rxFrameHeader = rxFrameHeader;
        for (auto i = 0; i < frame.rxFrameHeader.numRxSegments; i++) {
            auto [segmentName, segmentLength, versionId, offset] = AbstractPicoScenesFrameSegment::extractSegmentMetaData(buffer + pos, bufferLength - pos);
            if (segmentName == "RxSBasic") {
                frame.rxSBasicSegment = RxSBasicSegment(buffer + pos, segmentLength + 4);
            } else if (segmentName == "ExtraInfo") {
                frame.rxExtraInfoSegment = ExtraInfoSegment(buffer + pos, segmentLength + 4);
            } else if (segmentName == "MVMExtra") {
                frame.mvmExtraSegment = MVMExtraSegment(buffer + pos, segmentLength + 4);
                std::copy(frame.mvmExtraSegment->rawBuffer.cbegin(), frame.mvmExtraSegment->rawBuffer.cend(), frame.rawBuffer.begin() + pos);
            } else if (segmentName == "SDRExtra") {
                frame.sdrExtraSegment = SDRExtraSegment(buffer + pos, segmentLength + 4);
            } else if (segmentName == "CSI") {
                frame.csiSegment = CSISegment(buffer + pos, segmentLength + 4);
                if (interpolateCSI) {
                    frame.csiSegment.getCSI().removeCSDAndInterpolateCSI();
                }
            } else if (segmentName == "PilotCSI") {
                frame.pilotCSISegment = CSISegment(buffer + pos, segmentLength + 4);
            } else if (segmentName == "LegacyCSI") {
                frame.legacyCSISegment = CSISegment(buffer + pos, segmentLength + 4);
                if (interpolateCSI) {
                    frame.legacyCSISegment->getCSI().removeCSDAndInterpolateCSI();
                }
            } else if (segmentName == "BasebandSignal") {
                frame.basebandSignalSegment = BasebandSignalSegment(buffer + pos, segmentLength + 4);
            } else if (segmentName == "PreEQSymbols") {
                frame.preEQSymbolsSegment = PreEQSymbolsSegment(buffer + pos, segmentLength + 4);
            } else {
                frame.rxUnknownSegments.emplace(segmentName, AbstractPicoScenesFrameSegment(buffer + pos, segmentLength + 4));
            }
            pos += (segmentLength + 4);
        }

        auto mpduPos = pos;
        frame.standardHeader = ieee80211_mac_frame_header::createFromBuffer(buffer + pos, bufferLength - pos);
        pos += sizeof(ieee80211_mac_frame_header); // TODO this is somewhat dangerous.

        if (auto PSHeader = PicoScenesFrameHeader::fromBuffer(buffer + pos)) {
            frame.PicoScenesHeader = PSHeader;
            pos += sizeof(PicoScenesFrameHeader);

            for (auto i = 0; i < frame.PicoScenesHeader->numSegments; i++) {
                auto [segmentName, segmentLength, versionId, offset] = AbstractPicoScenesFrameSegment::extractSegmentMetaData(buffer + pos, bufferLength);
                if (segmentName == "ExtraInfo") {
                    frame.txExtraInfoSegment = ExtraInfoSegment(buffer + pos, segmentLength + 4);
                } else if (segmentName == "Payload") {
                    frame.payloadSegments.emplace_back(PayloadSegment(buffer + pos, segmentLength + 4));
                } else if (segmentName == "Cargo") {
                    frame.cargoSegment = CargoSegment(buffer + pos, segmentLength + 4);
                } else {
                    frame.txUnknownSegments.emplace(segmentName, AbstractPicoScenesFrameSegment(buffer + pos, segmentLength + 4));
                }
                pos += segmentLength + 4;
            }
        }

        std::copy(buffer + mpduPos, buffer + bufferLength, std::back_inserter(frame.mpdu));
    } catch (const std::exception &exception) {
        std::cout << "Error occurs during Rx frame parsing:" << exception.what() << ". Error skipped" << std::endl;
        return {};
    }

    return frame;
}

std::string ModularPicoScenesRxFrame::toString() const {
    std::stringstream ss;
    ss << "RxFrame:{";
    ss << rxSBasicSegment.getBasic();
    ss << ", Rx" << rxExtraInfoSegment.getExtraInfo();
    if (mvmExtraSegment)
        ss << ", " << *mvmExtraSegment;
    if (sdrExtraSegment)
        ss << ", " << *sdrExtraSegment;
    ss << ", " << "(" << PacketFormat2String(csiSegment.getCSI().packetFormat) << ")" << csiSegment;
    if (pilotCSISegment)
        ss << ", " << *pilotCSISegment;
    if (legacyCSISegment)
        ss << ", " << *legacyCSISegment;
    if (basebandSignalSegment)
        ss << ", " << *basebandSignalSegment;
    if (preEQSymbolsSegment)
        ss << ", " << *preEQSymbolsSegment;
    if (!rxUnknownSegments.empty()) {
        std::stringstream segss;
        segss << "RxSegments:(";
        for (const auto &segment: rxUnknownSegments) {
            segss << segment.first << ":" << segment.second.totalLengthIncludingLeading4ByteLength() << "B, ";
        }
        auto temp = segss.str();
        temp.erase(temp.end() - 2, temp.end());
        temp.append(")");
        ss << ", " << temp;
    }

    if (!mpdu.empty())
        ss << ", " << standardHeader;
    else
        ss << ", NDP frame";
    if (PicoScenesHeader)
        ss << ", " << *PicoScenesHeader;
    if (txExtraInfoSegment)
        ss << ", Tx" << txExtraInfoSegment->getExtraInfo();
    if (!payloadSegments.empty()) {
        std::stringstream segss;
        segss << "Payloads:(";
        for (const auto &segment: payloadSegments) {
            segss << segment << ", ";
        }
        auto temp = segss.str();
        temp.erase(temp.end() - 2, temp.end());
        temp.append(")");
        ss << ", " << temp;
    }
    if (cargoSegment)
        ss << ", " << *cargoSegment;
    if (!txUnknownSegments.empty()) {
        std::stringstream segss;
        segss << "TxSegments:(";
        for (const auto &segment: txUnknownSegments) {
            segss << segment.second << ", ";
        }
        auto temp = segss.str();
        temp.erase(temp.end() - 2, temp.end());
        temp.append(")");
        ss << ", " << temp;
    }
    ss << ", MPDU=" << mpdu.size() << "B";
    ss << "}";
    return ss.str();
}

std::optional<ModularPicoScenesRxFrame> ModularPicoScenesRxFrame::concatenateFragmentedPicoScenesRxFrames(const std::vector<ModularPicoScenesRxFrame> &frameQueue) {
    ModularPicoScenesRxFrame baseFrame = frameQueue.front();
    std::vector<PayloadCargo> cargos;
    std::for_each(frameQueue.cbegin(), frameQueue.cend(), [&](const ModularPicoScenesRxFrame &frame) {
        cargos.emplace_back(frame.cargoSegment->getCargo());
    });
    auto mergedPayload = PayloadCargo::mergeAndValidateCargo(cargos);

    auto pos = 0, numSegment = 0;
    while (pos < mergedPayload.size()) {
        auto [segmentName, segmentLength, versionId, contentOffset] = AbstractPicoScenesFrameSegment::extractSegmentMetaData(mergedPayload.data() + pos, mergedPayload.size() - pos);
        if (segmentName == "ExtraInfo") {
            baseFrame.txExtraInfoSegment = ExtraInfoSegment(mergedPayload.data() + pos, segmentLength + 4);
        } else if (segmentName == "Payload") {
            baseFrame.payloadSegments.emplace_back(PayloadSegment(mergedPayload.data() + pos, segmentLength + 4));
        } else if (segmentName == "Cargo") {
            baseFrame.cargoSegment = CargoSegment(mergedPayload.data() + pos, segmentLength + 4);
        } else {
            baseFrame.txUnknownSegments.emplace(segmentName, AbstractPicoScenesFrameSegment(mergedPayload.data() + pos, segmentLength + 4));
        }
        pos += segmentLength + 4;
        numSegment++;
    }
    baseFrame.cargoSegment = std::nullopt;
    baseFrame.PicoScenesHeader->numSegments = numSegment;
    baseFrame.rebuildRawBuffer();

    return baseFrame;
}

Uint8Vector ModularPicoScenesRxFrame::toBuffer() const {
    if (!rawBuffer.empty())
        return rawBuffer;

    // Rx segments
    Uint8Vector rxSegmentBuffer;
    auto modularFrameHeader = ModularPicoScenesRxFrameHeader().initialize2Default();
    modularFrameHeader.numRxSegments = 3;

    auto rxsBasicBuffer = rxSBasicSegment.toBuffer();
    std::copy(rxsBasicBuffer.cbegin(), rxsBasicBuffer.cend(), std::back_inserter(rxSegmentBuffer));

    auto rxsExtraInfoBuffer = rxExtraInfoSegment.toBuffer();
    std::copy(rxsExtraInfoBuffer.cbegin(), rxsExtraInfoBuffer.cend(), std::back_inserter(rxSegmentBuffer));

    auto csiBuffer = csiSegment.toBuffer();
    std::copy(csiBuffer.cbegin(), csiBuffer.cend(), std::back_inserter(rxSegmentBuffer));

    if (mvmExtraSegment) {
        auto buffer = mvmExtraSegment->toBuffer();
        std::copy(buffer.cbegin(), buffer.cend(), std::back_inserter(rxSegmentBuffer));
        modularFrameHeader.numRxSegments++;
    }

    if (sdrExtraSegment) {
        auto buffer = sdrExtraSegment->toBuffer();
        std::copy(buffer.cbegin(), buffer.cend(), std::back_inserter(rxSegmentBuffer));
        modularFrameHeader.numRxSegments++;
    }

    if (pilotCSISegment) {
        auto pilotCSIBuffer = pilotCSISegment->toBuffer();
        std::copy(pilotCSIBuffer.cbegin(), pilotCSIBuffer.cend(), std::back_inserter(rxSegmentBuffer));
        modularFrameHeader.numRxSegments++;
    }

    if (legacyCSISegment) {
        auto legacyCSIBuffer = legacyCSISegment->toBuffer();
        std::copy(legacyCSIBuffer.cbegin(), legacyCSIBuffer.cend(), std::back_inserter(rxSegmentBuffer));
        modularFrameHeader.numRxSegments++;
    }

    if (basebandSignalSegment) {
        auto segmentBuffer = basebandSignalSegment->toBuffer();
        std::copy(segmentBuffer.cbegin(), segmentBuffer.cend(), std::back_inserter(rxSegmentBuffer));
        modularFrameHeader.numRxSegments++;
    }

    if (preEQSymbolsSegment) {
        auto segmentBuffer = preEQSymbolsSegment->toBuffer();
        std::copy(segmentBuffer.cbegin(), segmentBuffer.cend(), std::back_inserter(rxSegmentBuffer));
        modularFrameHeader.numRxSegments++;
    }

    // Assembly the full buffer
    Uint8Vector frameBuffer;
    modularFrameHeader.frameLength = sizeof(modularFrameHeader) + rxSegmentBuffer.size() + mpdu.size() - 4;
    std::copy((uint8_t *) &modularFrameHeader, (uint8_t *) &modularFrameHeader + sizeof(ModularPicoScenesRxFrameHeader), std::back_inserter(frameBuffer));
    std::copy(rxSegmentBuffer.cbegin(), rxSegmentBuffer.cend(), std::back_inserter(frameBuffer));
    std::copy(mpdu.cbegin(), mpdu.cend(), std::back_inserter(frameBuffer));
    //// for in-situ validation
//    {
//        auto recovered = ModularPicoScenesRxFrame::fromBuffer(frameBuffer.data(), frameBuffer.size());
//        std::cout << "recovered:" << *recovered << std::endl;
//    }

    return frameBuffer;
}

void ModularPicoScenesRxFrame::rebuildRawBuffer() {
    rawBuffer.clear();

    if (PicoScenesHeader) {
        mpdu.clear();
        std::copy((uint8_t *) &standardHeader, (uint8_t *) &standardHeader + sizeof(standardHeader), std::back_inserter(mpdu));
        std::copy((uint8_t *) &PicoScenesHeader.value(), (uint8_t *) &PicoScenesHeader.value() + sizeof(PicoScenesFrameHeader), std::back_inserter(mpdu));

        if (txExtraInfoSegment) {
            const auto &buffer = txExtraInfoSegment->toBuffer();
            std::copy(buffer.cbegin(), buffer.cend(), std::back_inserter(mpdu));
        }
        for (const auto &payloadSegment: payloadSegments) {
            const auto &buffer = payloadSegment.toBuffer();
            std::copy(buffer.cbegin(), buffer.cend(), std::back_inserter(mpdu));
        }
        if (cargoSegment) {
            const auto &buffer = cargoSegment->toBuffer();
            std::copy(buffer.cbegin(), buffer.cend(), std::back_inserter(mpdu));
        }
        for (const auto &unknownSegment: txUnknownSegments) {
            auto buffer = unknownSegment.second.toBuffer();
            std::copy(buffer.cbegin(), buffer.cend(), std::back_inserter(mpdu));
        }
    }

    rawBuffer = toBuffer();
}

std::ostream &operator<<(std::ostream &os, const ModularPicoScenesRxFrame &rxframe) {
    os << rxframe.toString();
    return os;
}

void ModularPicoScenesTxFrame::addSegments(const std::shared_ptr<AbstractPicoScenesFrameSegment> &segment) {
    segments.emplace_back(segment);
    if (!frameHeader)
        frameHeader = PicoScenesFrameHeader();
    frameHeader->numSegments = segments.size();
}

std::shared_ptr<AbstractPicoScenesFrameSegment> ModularPicoScenesTxFrame::getSegment(const std::string &querySegmentName) {
    auto resultIt = std::find_if(segments.begin(), segments.end(), [=](const auto &eachSegment) {
        return eachSegment->segmentName == querySegmentName;
    });

    return resultIt == segments.end() ? nullptr : *resultIt;
}

uint32_t ModularPicoScenesTxFrame::totalLength() const {
    if (txParameters.NDPFrame)
        return 0;

    if (!arbitraryMPDUContent.empty())
        return arbitraryMPDUContent.size();

    uint32_t length = sizeof(ieee80211_mac_frame_header) + (frameHeader ? sizeof(PicoScenesFrameHeader) : 4); // plus 4 is to avoid NDP skip on QCA9300
    for (const auto &segment: segments) {
        length += segment->totalLength() + 4;
    }
    return length;
}

void ModularPicoScenesTxFrame::prebuildMPDU() {
    prebuiltMPDU = toBuffer();
}

Uint8Vector ModularPicoScenesTxFrame::toBuffer() const {
    if (txParameters.NDPFrame)
        return {};

    auto bufferLength = totalLength();
    Uint8Vector buffer(bufferLength);
    auto copiedLength = toBuffer(&buffer[0], bufferLength);
    assert(bufferLength == copiedLength);
    return buffer;
}

int ModularPicoScenesTxFrame::toBuffer(uint8_t *buffer, uint32_t bufferLength) const {
    if (bufferLength < totalLength())
        throw std::overflow_error("Buffer not long enough for TX frame dumping...");

    if (!arbitraryMPDUContent.empty()) {
        std::copy(arbitraryMPDUContent.cbegin(), arbitraryMPDUContent.cend(), buffer);
        return arbitraryMPDUContent.size();
    }

    memset(buffer, 0, bufferLength);
    memcpy(buffer, &standardHeader, sizeof(ieee80211_mac_frame_header));
    uint32_t pos = sizeof(ieee80211_mac_frame_header);
    if (frameHeader) {
        if (frameHeader->numSegments != segments.size())
            throw std::overflow_error("ModularPicoScenesTxFrame toBuffer method segment number in-consistent!");

        memcpy(buffer + sizeof(ieee80211_mac_frame_header), &frameHeader, sizeof(PicoScenesFrameHeader));
        pos += sizeof(PicoScenesFrameHeader);
        for (const auto &segment: segments) {
            segment->toBuffer(buffer + pos);
            pos += segment->totalLength() + 4;
        }
    }

    return pos;
}

std::vector<ModularPicoScenesTxFrame> ModularPicoScenesTxFrame::autoSplit(int16_t maxSegmentBuffersLength) const {
    if (!frameHeader)
        return std::vector<ModularPicoScenesTxFrame>{1, *this};

    auto segmentsLength = 0;
    for (const auto &segment: segments) {
        segmentsLength += segment->totalLength() + 4;
    }

    if (segmentsLength < maxSegmentBuffersLength)
        return std::vector<ModularPicoScenesTxFrame>{1, *this};

    auto pos = 0;
    Uint8Vector allSegmentBuffer(segmentsLength);
    for (const auto &segment: segments) {
        segment->toBuffer(allSegmentBuffer.data() + pos);
        pos += segment->totalLength() + 4;
    }

    pos = 0;
    uint8_t sequence = 0;
    uint8_t numCargos = std::ceil(1.0 * segmentsLength / maxSegmentBuffersLength);
    auto cargos = std::vector<PayloadCargo>();
    while (pos < segmentsLength) {
        auto stepLength = pos + maxSegmentBuffersLength < segmentsLength ? maxSegmentBuffersLength : segmentsLength - pos;
        cargos.emplace_back(PayloadCargo{
                .taskId = frameHeader->taskId,
                .numSegments = frameHeader->numSegments,
                .sequence = sequence++,
                .totalParts = numCargos,
                .payloadData = Uint8Vector(allSegmentBuffer.data() + pos, allSegmentBuffer.data() + pos + stepLength),
        });
        pos += stepLength;
    }

    auto cargoFrames = std::vector<ModularPicoScenesTxFrame>();
    for (auto i = 0; i < cargos.size(); i++) {
        const auto &cargo = cargos[i];
        auto txframe = *this;
        txframe.standardHeader.seq = i;
        txframe.segments.clear();
        txframe.addSegments(std::make_shared<CargoSegment>(cargo));
        cargoFrames.emplace_back(txframe);
    }
    return cargoFrames;
}

void ModularPicoScenesTxFrame::reset() {
    standardHeader = ieee80211_mac_frame_header();
    frameHeader = PicoScenesFrameHeader();
    txParameters = PicoScenesFrameTxParameters();
    arbitraryMPDUContent.clear();
    AMPDUFrames.clear();
    prebuiltCS8Signals.clear();
    prebuiltCS16Signals.clear();
    segments.clear();
}

ModularPicoScenesTxFrame &ModularPicoScenesTxFrame::setMoreFrags() {
    standardHeader.fc.moreFrags = 1;
    return *this;
}

ModularPicoScenesTxFrame &ModularPicoScenesTxFrame::setFragNumber(uint8_t fragNumber) {
    standardHeader.frag = fragNumber;
    return *this;
}

ModularPicoScenesTxFrame &ModularPicoScenesTxFrame::setRetry() {
    standardHeader.fc.retry = 1;
    return *this;
}

ModularPicoScenesTxFrame &ModularPicoScenesTxFrame::setDeviceType(PicoScenesDeviceType deviceType) {
    if (!frameHeader)
        frameHeader = PicoScenesFrameHeader();
    frameHeader->deviceType = deviceType;
    return *this;
}

ModularPicoScenesTxFrame &ModularPicoScenesTxFrame::setTaskId(uint16_t taskId) {
    if (!frameHeader)
        frameHeader = PicoScenesFrameHeader();
    frameHeader->taskId = taskId;
    return *this;
}

ModularPicoScenesTxFrame &ModularPicoScenesTxFrame::setTxId(uint16_t txId) {
    if (!frameHeader)
        frameHeader = PicoScenesFrameHeader();
    frameHeader->txId = txId;
    return *this;
}

ModularPicoScenesTxFrame &ModularPicoScenesTxFrame::setRandomTaskId() {
    static std::random_device r;
    static std::default_random_engine randomEngine(r());
    static std::uniform_int_distribution<uint16_t> randomGenerator(10000, UINT16_MAX);
    auto newValue = randomGenerator(randomEngine);
    setTaskId(newValue);
    return *this;
}

ModularPicoScenesTxFrame &ModularPicoScenesTxFrame::setPicoScenesFrameType(uint8_t frameType) {
    if (!frameHeader)
        frameHeader = PicoScenesFrameHeader();
    frameHeader->frameType = frameType;
    return *this;
}

[[maybe_unused]] ModularPicoScenesTxFrame &ModularPicoScenesTxFrame::setFrameFormat(PacketFormatEnum format) {
    txParameters.frameType = format;
    return *this;
}

ModularPicoScenesTxFrame &ModularPicoScenesTxFrame::setChannelBandwidth(const ChannelBandwidthEnum &cbw) {
    txParameters.cbw = cbw;
    return *this;
}

ModularPicoScenesTxFrame &ModularPicoScenesTxFrame::setGuardInterval(GuardIntervalEnum guardInterval) {
    txParameters.guardInterval = guardInterval;
    return *this;
}

ModularPicoScenesTxFrame &ModularPicoScenesTxFrame::setMCS(uint8_t mcs) {
    txParameters.mcs.clear();
    txParameters.mcs.emplace_back(mcs);
    return *this;
}

ModularPicoScenesTxFrame &ModularPicoScenesTxFrame::setNumSTS(uint8_t numSTS) {
    txParameters.numSTS.clear();
    txParameters.numSTS.emplace_back(numSTS);
    return *this;
}

ModularPicoScenesTxFrame &ModularPicoScenesTxFrame::setNumSTS(const std::vector<uint8_t> &numSTSs) {
    txParameters.numSTS = numSTSs;
    return *this;
}


[[maybe_unused]] ModularPicoScenesTxFrame &ModularPicoScenesTxFrame::setMCS(const std::vector<uint8_t> &mcs) {
    txParameters.mcs = mcs;
    return *this;
}

ModularPicoScenesTxFrame &ModularPicoScenesTxFrame::setDestinationAddress(const uint8_t macAddr[6]) {
    std::copy(macAddr, macAddr + 6, standardHeader.addr1.begin());
    return *this;
}

ModularPicoScenesTxFrame &ModularPicoScenesTxFrame::setSourceAddress(const uint8_t macAddr[6]) {
    std::copy(macAddr, macAddr + 6, standardHeader.addr2.begin());
    return *this;
}

ModularPicoScenesTxFrame &ModularPicoScenesTxFrame::set3rdAddress(const uint8_t macAddr[6]) {
    std::copy(macAddr, macAddr + 6, standardHeader.addr3.begin());
    return *this;
}

ModularPicoScenesTxFrame &ModularPicoScenesTxFrame::setForceSounding(bool forceSounding) {
    txParameters.forceSounding = forceSounding;
    return *this;
}

ModularPicoScenesTxFrame &ModularPicoScenesTxFrame::setNumExtraSounding(uint8_t numExtraSounding) {
    txParameters.numExtraSounding = numExtraSounding;
    return *this;
}

ModularPicoScenesTxFrame &ModularPicoScenesTxFrame::setChannelCoding(ChannelCodingEnum codings) {
    txParameters.coding.clear();
    txParameters.coding.emplace_back(codings);
    return *this;
}

ModularPicoScenesTxFrame &ModularPicoScenesTxFrame::setChannelCoding(const std::vector<ChannelCodingEnum> &channelCoding) {
    txParameters.coding = channelCoding;
    return *this;
}

ModularPicoScenesTxFrame &ModularPicoScenesTxFrame::setTxHEExtendedRange(bool txHEExtendedRange) {
    txParameters.txHEExtendedRange = txHEExtendedRange;
    return *this;
}

ModularPicoScenesTxFrame &ModularPicoScenesTxFrame::setHEHighDoppler(bool heHighDoppler) {
    txParameters.heHighDoppler = heHighDoppler;
    return *this;
}

ModularPicoScenesTxFrame &ModularPicoScenesTxFrame::setHEMidamblePeriodicity(double heMidamblePeriodicity) {
    txParameters.heMidamblePeriodicity = heMidamblePeriodicity;
    return *this;
}

std::string ModularPicoScenesTxFrame::toString() const {
    std::stringstream ss;
    if (!txParameters.NDPFrame) {
        ss << "TxFrame:{" << standardHeader;
        if (frameHeader)
            ss << ", " << *frameHeader;
    } else {
        ss << "TxFrame:{NDP frame";
    }
    ss << ", " << txParameters;

    if (!arbitraryMPDUContent.empty()) {
        ss << ", ArbitraryMPDU:" << std::to_string(arbitraryMPDUContent.size()) << "B}";
        return ss.str();
    }

    if (!segments.empty()) {
        std::stringstream segss;
        segss << "Segments:(";
        for (const auto &segment: segments) {
            segss << *segment << ", ";
        }
        auto temp = segss.str();
        temp.erase(temp.end() - 2, temp.end());
        temp.append(")");
        ss << ", " << temp;
    }
    ss << "}";
    return ss.str();
}

void ModularPicoScenesTxFrame::appendAMPDUFrames(const ModularPicoScenesTxFrame &frame) {
    AMPDUFrames.emplace_back(frame);
}

std::ostream &operator<<(std::ostream &os, const ModularPicoScenesTxFrame &txframe) {
    os << txframe.toString();
    return os;
}