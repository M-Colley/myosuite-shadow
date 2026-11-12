// OpenXR 1.1 extension types verified against Khronos OpenXR-SDK headers.
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <unknwn.h>
#define XR_USE_PLATFORM_WIN32
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <time.h>
#endif
#define XR_USE_TIMESPEC
#include <openxr/openxr_platform.h>
#include "QuestBodySender.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
struct ScopeExit {
    std::function<void()> cleanup;
    ~ScopeExit() { if (cleanup) cleanup(); }
};
#ifdef _WIN32
using Socket = SOCKET;
constexpr Socket invalidSocket = INVALID_SOCKET;
void closeSocket(Socket s) { closesocket(s); }
const char* clockId = "win32_qpc_seconds";
#else
using Socket = int;
constexpr Socket invalidSocket = -1;
void closeSocket(Socket s) { close(s); }
const char* clockId = "android_monotonic";
#endif
void check(XrResult result) {
    if (XR_FAILED(result)) throw std::runtime_error("OpenXR body call failed: " +
                                                  std::to_string(result));
}
template <class T> T proc(XrInstance instance, const char* name) {
    PFN_xrVoidFunction fn = nullptr;
    check(xrGetInstanceProcAddr(instance, name, &fn));
    return reinterpret_cast<T>(fn);
}
std::string quoted(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4)
                              << std::setfill('0') << int(c) << std::dec;
        else out << c;
    }
    return out.str() + '"';
}
}

struct QuestBodySender::Impl {
    XrInstance instance;
    XrBodyTrackerFB tracker = XR_NULL_HANDLE;
    XrSpace space;
    PFN_xrLocateBodyJointsFB locate;
    PFN_xrDestroyBodyTrackerFB destroy;
#ifdef _WIN32
    PFN_xrConvertWin32PerformanceCounterToTimeKHR fromNative;
    PFN_xrConvertTimeToWin32PerformanceCounterKHR toNative;
    double frequency;
#else
    PFN_xrConvertTimespecTimeToTimeKHR fromNative;
    PFN_xrConvertTimeToTimespecTimeKHR toNative;
#endif
    Socket socket = invalidSocket;
#ifdef _WIN32
    bool winsockStarted = false;
#endif
    sockaddr_in destination{};
    bool fullBody, fidelity;
    std::string person, device, sdk, runtime;
    uint64_t runtimeVersion = 0, sequence = 0;
    unsigned int revision = 0;
    std::vector<XrTime> changes;

    Impl(XrInstance i, XrSystemId system, XrSession session, XrSpace stage,
         bool full, bool fid, const std::string& ip, const std::string& p,
         const std::string& d, const std::string& s, unsigned short port,
         unsigned short clockPort)
        : instance(i), space(stage), fullBody(false), fidelity(fid),
          person(p), device(d), sdk(s) {
        ScopeExit onError{[this]() { cleanup(); }};
        if (p.empty() || d.empty() || s.empty())
            throw std::invalid_argument("person, device and SDK version are required");
        auto create = proc<PFN_xrCreateBodyTrackerFB>(i, "xrCreateBodyTrackerFB");
        locate = proc<PFN_xrLocateBodyJointsFB>(i, "xrLocateBodyJointsFB");
        destroy = proc<PFN_xrDestroyBodyTrackerFB>(i, "xrDestroyBodyTrackerFB");
        if (full) {
            XrSystemPropertiesBodyTrackingFullBodyMETA capability{
                XR_TYPE_SYSTEM_PROPERTIES_BODY_TRACKING_FULL_BODY_META};
            XrSystemProperties properties{XR_TYPE_SYSTEM_PROPERTIES};
            properties.next = &capability;
            check(xrGetSystemProperties(i, system, &properties));
            fullBody = capability.supportsFullBodyTracking == XR_TRUE;
        }
        XrBodyTrackerCreateInfoFB info{XR_TYPE_BODY_TRACKER_CREATE_INFO_FB};
        info.bodyJointSet = fullBody ? XR_BODY_JOINT_SET_FULL_BODY_META
                                    : XR_BODY_JOINT_SET_DEFAULT_FB;
        check(create(session, &info, &tracker));
        if (fidelity) {
            auto request = proc<PFN_xrRequestBodyTrackingFidelityMETA>(
                i, "xrRequestBodyTrackingFidelityMETA");
            check(request(tracker, XR_BODY_TRACKING_FIDELITY_HIGH_META));
        }
        XrInstanceProperties properties{XR_TYPE_INSTANCE_PROPERTIES};
        check(xrGetInstanceProperties(i, &properties));
        runtime = properties.runtimeName;
        runtimeVersion = properties.runtimeVersion;
#ifdef _WIN32
        fromNative = proc<PFN_xrConvertWin32PerformanceCounterToTimeKHR>(
            i, "xrConvertWin32PerformanceCounterToTimeKHR");
        toNative = proc<PFN_xrConvertTimeToWin32PerformanceCounterKHR>(
            i, "xrConvertTimeToWin32PerformanceCounterKHR");
        LARGE_INTEGER freq;
        QueryPerformanceFrequency(&freq);
        frequency = double(freq.QuadPart);
        WSADATA wsa;
        if (WSAStartup(MAKEWORD(2, 2), &wsa)) throw std::runtime_error("WSAStartup failed");
        winsockStarted = true;
#else
        fromNative = proc<PFN_xrConvertTimespecTimeToTimeKHR>(
            i, "xrConvertTimespecTimeToTimeKHR");
        toNative = proc<PFN_xrConvertTimeToTimespecTimeKHR>(
            i, "xrConvertTimeToTimespecTimeKHR");
#endif
        socket = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (socket == invalidSocket) throw std::runtime_error("UDP socket failed");
        sockaddr_in local{};
        local.sin_family = AF_INET;
        local.sin_addr.s_addr = htonl(INADDR_ANY);
        local.sin_port = htons(clockPort);
        if (bind(socket, reinterpret_cast<sockaddr*>(&local), sizeof(local)))
            throw std::runtime_error("clock UDP bind failed");
#ifdef _WIN32
        u_long nonblocking = 1;
        ioctlsocket(socket, FIONBIO, &nonblocking);
#else
        fcntl(socket, F_SETFL, O_NONBLOCK);
#endif
        destination.sin_family = AF_INET;
        destination.sin_port = htons(port);
        if (inet_pton(AF_INET, ip.c_str(), &destination.sin_addr) != 1)
            throw std::invalid_argument("receiver must be an IPv4 address");
        onError.cleanup = nullptr;
    }
    void cleanup() {
        if (tracker != XR_NULL_HANDLE) destroy(tracker);
        if (socket != invalidSocket) closeSocket(socket);
#ifdef _WIN32
        if (winsockStarted) WSACleanup();
#endif
    }
    ~Impl() { cleanup(); }
    XrTime nativeNow() {
        XrTime result;
#ifdef _WIN32
        LARGE_INTEGER counter;
        QueryPerformanceCounter(&counter);
        check(fromNative(instance, &counter, &result));
#else
        timespec counter;
        clock_gettime(CLOCK_MONOTONIC, &counter);
        check(fromNative(instance, &counter, &result));
#endif
        return result;
    }
    double seconds(XrTime time) {
#ifdef _WIN32
        LARGE_INTEGER counter;
        check(toNative(instance, time, &counter));
        return double(counter.QuadPart) / frequency;
#else
        timespec counter;
        check(toNative(instance, time, &counter));
        return double(counter.tv_sec) + double(counter.tv_nsec) * 1e-9;
#endif
    }
    void clockReplies() {
        char buffer[128];
        sockaddr_in peer{};
#ifdef _WIN32
        int size = sizeof(peer);
#else
        socklen_t size = sizeof(peer);
#endif
        for (int count = 0; count < 16; ++count) {
            int length = int(recvfrom(socket, buffer, sizeof(buffer) - 1, 0,
                reinterpret_cast<sockaddr*>(&peer), &size));
            if (length < 0) break;
            buffer[length] = 0;
            std::istringstream input(std::string(buffer, length));
            std::string tag, extra;
            uint64_t nonce;
            if (!(input >> tag >> nonce) || tag != "shadow-clock-v1" || input >> extra)
                continue;
            std::ostringstream reply;
            reply << std::setprecision(17) << "{\"clock_id\":" << quoted(clockId)
                  << ",\"nonce\":" << nonce << ",\"source_time_s\":"
                  << seconds(nativeNow()) << '}';
            auto text = reply.str();
            sendto(socket, text.data(), int(text.size()), 0,
                   reinterpret_cast<sockaddr*>(&peer), size);
        }
    }
    void send() {
        clockReplies();
        XrTime requested = nativeNow();
        XrBodyJointsLocateInfoFB info{XR_TYPE_BODY_JOINTS_LOCATE_INFO_FB};
        info.baseSpace = space;
        info.time = requested;
        std::vector<XrBodyJointLocationFB> joints(fullBody ? 84 : 70);
        XrBodyTrackingFidelityStatusMETA actual{XR_TYPE_BODY_TRACKING_FIDELITY_STATUS_META};
        XrBodyJointLocationsFB locations{XR_TYPE_BODY_JOINT_LOCATIONS_FB};
        locations.next = fidelity ? &actual : nullptr;
        locations.jointCount = uint32_t(joints.size());
        locations.jointLocations = joints.data();
        check(locate(tracker, &info, &locations));
        if (locations.time <= 0) locations.time = requested;
        for (auto change = changes.begin(); change != changes.end();) {
            if (locations.time >= *change) { ++revision; change = changes.erase(change); }
            else ++change;
        }
        const double sampleTime = seconds(locations.time);
        const double prediction = std::max(0.0, sampleTime - seconds(nativeNow()));
        const char* actualName = !fidelity ? "unknown" :
            actual.fidelity == XR_BODY_TRACKING_FIDELITY_HIGH_META ? "high" : "low";
        std::ostringstream out;
        out << std::setprecision(17)
            << "{\"schema_version\":1,\"kind\":\"quest_body\",\"source_id\":\"quest\","
            << "\"person_id\":" << quoted(person) << ",\"sequence\":" << sequence++
            << ",\"clock_id\":" << quoted(clockId) << ",\"time_s\":" << sampleTime
            << ",\"requested_time_s\":" << seconds(requested)
            << ",\"time_kind\":" << quoted(prediction > 0 ? "predicted" : "estimate")
            << ",\"prediction_s\":" << prediction << ",\"base_space\":\"stage\","
            << "\"space_revision\":" << revision
            << ",\"skeleton_revision\":" << locations.skeletonChangedCount
            << ",\"joint_set\":" << quoted(fullBody ? "full_body" : "upper_body")
            << ",\"actual_fidelity\":" << quoted(actualName)
            << ",\"capabilities\":{\"body_tracking\":true,\"full_body\":"
            << (fullBody ? "true" : "false") << ",\"fidelity\":"
            << (fidelity ? "true" : "false") << '}'
            << ",\"device\":" << quoted(device) << ",\"sdk_version\":" << quoted(sdk)
            << ",\"runtime\":" << quoted(runtime) << ",\"runtime_version\":" << runtimeVersion
            << ",\"is_active\":" << (locations.isActive ? "true" : "false")
            << ",\"confidence\":" << (locations.isActive ? locations.confidence : 0)
            << ",\"joints\":[";
        for (size_t j = 0; j < joints.size(); ++j) {
            if (j) out << ',';
            auto flags = locations.isActive ? joints[j].locationFlags : 0;
            XrPosef pose{};
            pose.orientation.w = 1;
            if (flags & XR_SPACE_LOCATION_POSITION_VALID_BIT) pose.position = joints[j].pose.position;
            if (flags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT) pose.orientation = joints[j].pose.orientation;
            out << "{\"id\":" << j << ",\"location_flags\":" << flags
                << ",\"position\":[" << pose.position.x << ',' << pose.position.y << ',' << pose.position.z
                << "],\"orientation\":[" << pose.orientation.x << ',' << pose.orientation.y
                << ',' << pose.orientation.z << ',' << pose.orientation.w << "]}";
        }
        out << "]}";
        auto text = out.str();
        if (text.size() > 60000) throw std::runtime_error("body packet too large");
        if (sendto(socket, text.data(), int(text.size()), 0,
                   reinterpret_cast<sockaddr*>(&destination), sizeof(destination)) < 0)
            throw std::runtime_error("body UDP send failed");
    }
};

QuestBodySender::QuestBodySender(XrInstance instance, XrSystemId system, XrSession session,
    XrSpace stageSpace, bool full, bool fidelity, const std::string& ip,
    const std::string& person, const std::string& device, const std::string& sdk,
    unsigned short port, unsigned short clockPort)
    : impl(std::make_unique<Impl>(instance, system, session, stageSpace, full, fidelity,
                                ip, person, device, sdk, port, clockPort)) {}
QuestBodySender::~QuestBodySender() = default;
void QuestBodySender::Send() { impl->send(); }
void QuestBodySender::ReferenceSpaceChanged(const XrEventDataReferenceSpaceChangePending& e) {
    if (e.referenceSpaceType == XR_REFERENCE_SPACE_TYPE_STAGE)
        impl->changes.push_back(e.changeTime);
}
