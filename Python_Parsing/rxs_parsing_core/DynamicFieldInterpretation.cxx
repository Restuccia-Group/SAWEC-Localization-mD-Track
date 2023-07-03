//
// Created by Zhiping Jiang on 10/4/22.
//

#include "DynamicFieldInterpretation.hxx"

std::optional<DynamicContentField> DynamicContentType::queryFieldOffset(const std::string &fieldName) {
    if (quickQueryMap.find(fieldName) != quickQueryMap.cend()) {
        return quickQueryMap[fieldName];
    }

    if (auto result = std::find_if(fields.cbegin(), fields.cend(), [=](const DynamicContentField &currentField) {
            return currentField.fieldName == fieldName;
        }); result != fields.cend()) {
        quickQueryMap[fieldName] = *result;
        return *result;
    }
    return std::nullopt;
}

DynamicContentType::DynamicContentType(const std::string &name, uint16_t version, const std::vector<DynamicContentField> &fields) : name(name), version(version), fields(fields) {}

std::shared_ptr<DynamicContentTypeDictionary> DynamicContentTypeDictionary::getInstance() {
    static auto instance = std::shared_ptr<DynamicContentTypeDictionary>(new DynamicContentTypeDictionary());
    return instance;
}

std::shared_ptr<DynamicContentType> DynamicContentTypeDictionary::queryType(const std::string &name, uint16_t version) {
    auto queryString = name + std::to_string(version);
    return dictionary.find(queryString) != dictionary.cend() ? dictionary[queryString] : nullptr;
}

void DynamicContentTypeDictionary::registerType(const DynamicContentType &type) {
    auto queryString = type.name + std::to_string(type.version);
    dictionary[queryString] = std::make_shared<DynamicContentType>(type);
}

std::optional<DynamicContentField> DynamicFieldInterpreter::queryField(const std::string &fieldName) {
    if (!typeCache) {
        if (typeCache = DynamicContentTypeDictionary::getInstance()->queryType(typeName, version); !typeCache)
            throw std::invalid_argument("Field Type not existent: " + typeName);
    }

    return typeCache->queryFieldOffset(fieldName);
}

std::optional<DynamicContentField> DynamicFieldInterpreter::queryField(const std::string &fieldName) const {
    if (auto typeQuery = DynamicContentTypeDictionary::getInstance()->queryType(typeName, version)) {
        if (auto fieldQuery = typeQuery->queryFieldOffset(fieldName))
            return fieldQuery;
    }

    return std::nullopt;
}
