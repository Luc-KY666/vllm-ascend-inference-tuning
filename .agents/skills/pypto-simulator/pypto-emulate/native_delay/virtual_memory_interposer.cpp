#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <mutex>
#include <sys/file.h>
#include <sys/types.h>
#include <unistd.h>
#include <unordered_map>

namespace {

constexpr uint64_t kMagic = 0x5059544f564d454dULL;
constexpr uint64_t kVersion = 1;
constexpr uint64_t kPending = 1;
constexpr uint64_t kConsumed = 2;
constexpr uint64_t kFailed = 3;
constexpr uint64_t kPageBytes = 2ULL * 1024ULL * 1024ULL;

struct ControlRecord {
    uint64_t magic;
    uint64_t version;
    uint64_t state;
    uint64_t request_id;
    uint64_t expected_bytes;
    uint64_t virtual_address;
    uint64_t reserved_bytes;
    int64_t consumer_pid;
    int64_t forbidden_pid;
    int64_t error_code;
    char operation[32];
};
static_assert(sizeof(ControlRecord) == 112);

using RtMallocFn = int (*)(void **, uint64_t, int, uint16_t);
using RtFreeFn = int (*)(void *);
using ReserveFn = int (*)(void **, size_t, size_t, void *, uint64_t);
using ReleaseFn = int (*)(void *);

std::mutex g_virtual_mu;
std::unordered_map<void *, uint64_t> g_virtual_ranges;
thread_local bool g_inside_interposer = false;

template <typename Fn>
Fn next_symbol(const char *name) {
    return reinterpret_cast<Fn>(dlsym(RTLD_NEXT, name));
}

uint64_t align_up(uint64_t value, uint64_t alignment) {
    if (value == 0 || value > UINT64_MAX - (alignment - 1)) return 0;
    return ((value + alignment - 1) / alignment) * alignment;
}

bool load_control(int fd, ControlRecord *record) {
    return pread(fd, record, sizeof(*record), 0) == static_cast<ssize_t>(sizeof(*record)) &&
           record->magic == kMagic && record->version == kVersion;
}

void store_control(int fd, const ControlRecord &record) {
    const ssize_t written = pwrite(fd, &record, sizeof(record), 0);
    if (written == static_cast<ssize_t>(sizeof(record))) (void)fsync(fd);
}

// Returns true only when this call owns a pending virtual-allocation token.
bool consume_virtual_token(uint64_t size, void **out, int *error) {
    const char *path = std::getenv("PYPTO_NATIVE_DELAY_VMEM_CONTROL");
    if (path == nullptr || path[0] == '\0') return false;
    const int fd = open(path, O_RDWR | O_CLOEXEC);
    if (fd < 0) return false;
    if (flock(fd, LOCK_EX) != 0) {
        close(fd);
        return false;
    }

    ControlRecord record{};
    const bool valid = load_control(fd, &record);
    if (!valid || record.state != kPending || record.forbidden_pid == static_cast<int64_t>(getpid())) {
        (void)flock(fd, LOCK_UN);
        close(fd);
        return false;
    }

    record.consumer_pid = static_cast<int64_t>(getpid());
    if (record.expected_bytes != size) {
        record.state = kFailed;
        record.error_code = -2;
        store_control(fd, record);
        (void)flock(fd, LOCK_UN);
        close(fd);
        *error = 1;
        return true;
    }

    const uint64_t reserved = align_up(size, kPageBytes);
    ReserveFn reserve = next_symbol<ReserveFn>("aclrtReserveMemAddress");
    void *address = nullptr;
    const int rc = reserved == 0 || reserve == nullptr
                       ? 1
                       : reserve(&address, static_cast<size_t>(reserved), 0, nullptr, 0);
    if (rc != 0 || address == nullptr) {
        record.state = kFailed;
        record.error_code = rc == 0 ? -3 : rc;
        store_control(fd, record);
        (void)flock(fd, LOCK_UN);
        close(fd);
        *error = rc == 0 ? 1 : rc;
        return true;
    }

    {
        std::scoped_lock lock(g_virtual_mu);
        g_virtual_ranges[address] = reserved;
    }
    record.state = kConsumed;
    record.virtual_address = reinterpret_cast<uint64_t>(address);
    record.reserved_bytes = reserved;
    record.error_code = 0;
    store_control(fd, record);
    (void)flock(fd, LOCK_UN);
    close(fd);
    *out = address;
    *error = 0;
    return true;
}

}  // namespace

extern "C" int rtMalloc(void **dev_ptr, uint64_t size, int type, const uint16_t module_id) {
    RtMallocFn real_malloc = next_symbol<RtMallocFn>("rtMalloc");
    if (dev_ptr == nullptr || real_malloc == nullptr) return 1;
    if (g_inside_interposer) return real_malloc(dev_ptr, size, type, module_id);
    g_inside_interposer = true;
    int error = 0;
    const bool consumed = consume_virtual_token(size, dev_ptr, &error);
    g_inside_interposer = false;
    return consumed ? error : real_malloc(dev_ptr, size, type, module_id);
}

extern "C" int rtFree(void *dev_ptr) {
    RtFreeFn real_free = next_symbol<RtFreeFn>("rtFree");
    if (dev_ptr == nullptr) return 0;
    bool is_virtual = false;
    {
        std::scoped_lock lock(g_virtual_mu);
        const auto it = g_virtual_ranges.find(dev_ptr);
        if (it != g_virtual_ranges.end()) {
            g_virtual_ranges.erase(it);
            is_virtual = true;
        }
    }
    if (!is_virtual) return real_free == nullptr ? 1 : real_free(dev_ptr);
    ReleaseFn release = next_symbol<ReleaseFn>("aclrtReleaseMemAddress");
    return release == nullptr ? 1 : release(dev_ptr);
}
