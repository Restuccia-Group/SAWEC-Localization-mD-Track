//
// Created by Zhiping Jiang on 10/4/22.
//

#ifndef PICOSCENES_PLATFORM_DYNAMICFIELDINTERPRETATION_HXX
#define PICOSCENES_PLATFORM_DYNAMICFIELDINTERPRETATION_HXX

#include <map>
#include <vector>
#include <memory>
#include <optional>
#include <string>

class DynamicContentTypeDictionary;

enum class DynamicContentFieldPrimitiveType : uint8_t {
    Int8 = 0,
    Uint8,
    Int16,
    Uint16,
    Int32,
    Uint32,
    Int64,
    Uint64,
    Single,
    Double
};

struct DynamicContentField {
    std::string fieldName;
    DynamicContentFieldPrimitiveType fieldType;
    int32_t fieldOffset;
    int32_t arraySize{1};

    bool operator==(const DynamicContentField &rhs) const {
        return std::tie(fieldName, fieldType, fieldOffset) == std::tie(rhs.fieldName, rhs.fieldType, rhs.fieldOffset);
    }

    bool operator<(const DynamicContentField &rhs) const {
        return fieldOffset < rhs.fieldOffset;
    }
};

class DynamicContentType {
public:
    std::string name{};
    uint16_t version{0};
    std::vector<DynamicContentField> fields;

    DynamicContentType(const std::string &name, uint16_t version, const std::vector<DynamicContentField> &fields);

    std::optional<DynamicContentField> queryFieldOffset(const std::string &fieldName);

    bool operator==(const DynamicContentType &rhs) const {
        return std::tie(name, version) == std::tie(rhs.name, rhs.version);
    }

private:
    friend DynamicContentTypeDictionary;

    DynamicContentType() = default;

    std::map<std::string, DynamicContentField> quickQueryMap;
};

class DynamicContentTypeDictionary {
public:
    static std::shared_ptr<DynamicContentTypeDictionary> getInstance();

    void registerType(const DynamicContentType &type);

    std::shared_ptr<DynamicContentType> queryType(const std::string &name, uint16_t version);

private:
    DynamicContentTypeDictionary() = default;

    std::map<std::string, std::shared_ptr<DynamicContentType>> dictionary;
};

class DynamicFieldInterpreter {
public:
    DynamicFieldInterpreter() = default;

    DynamicFieldInterpreter(const std::string &typeName, uint16_t version, const uint8_t *content) : typeName(typeName), version(version), content(content) {}

    std::optional<DynamicContentField> queryField(const std::string &fieldName);

    std::optional<DynamicContentField> queryField(const std::string &fieldName) const;

    template<typename OutputType>
    const OutputType *getFieldPointer(const std::string &fieldName) const {
        if (auto queryResult = queryField(fieldName)) {
            return (OutputType *) (content + queryResult->fieldOffset);
        } else
            throw std::invalid_argument("Field not existent: " + fieldName);
    }

    template<typename OutputType>
    OutputType getField(const std::string &fieldName) const {
        return *getFieldPointer<OutputType>(fieldName);
    }

    template<typename OutputType>
    std::vector<OutputType> getArray(const std::string &fieldName) const {
        if (auto queryResult = queryField(fieldName)) {
            auto numElement = queryResult->arraySize;
            return std::vector<OutputType>((OutputType *) (content + queryResult->fieldOffset), (OutputType *) (content + queryResult->fieldOffset) + numElement);
        } else
            throw std::invalid_argument("Field not existent: " + fieldName);
    }

private:
    std::string typeName;
    uint16_t version{0};
    const uint8_t *content;

    std::shared_ptr<DynamicContentType> typeCache{};
};


#endif //PICOSCENES_PLATFORM_DYNAMICFIELDINTERPRETATION_HXX
