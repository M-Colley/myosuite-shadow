#pragma once

#include <openxr/openxr.h>
#include <memory>
#include <string>

// Extensions must be enabled on the existing application's OpenXR instance.
class QuestBodySender {
public:
    QuestBodySender(XrInstance instance, XrSystemId system, XrSession session,
                    XrSpace stageSpace, bool fullBodyExtensionEnabled,
                    bool fidelityExtensionEnabled, const std::string& receiverIpv4,
                    const std::string& personId, const std::string& device,
                    const std::string& sdkVersion, unsigned short port = 9763,
                    unsigned short clockPort = 9764);
    ~QuestBodySender();
    QuestBodySender(const QuestBodySender&) = delete;
    QuestBodySender& operator=(const QuestBodySender&) = delete;
    // Call every app update, even while no valid body is available.
    void Send();
    // Forward this event from the application's existing xrPollEvent loop.
    void ReferenceSpaceChanged(const XrEventDataReferenceSpaceChangePending& event);
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
