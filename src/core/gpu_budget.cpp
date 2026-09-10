#include "../internal/gpu_budget.hpp"

#include <cctype>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace rfdetr::gpu_budget {

namespace {

struct State {
    std::mutex m;
    std::size_t in_use{0};
    std::size_t peak{0};
    std::size_t limit{0};
    bool limit_read{false};
    std::string limit_error;
};

// Intentionally survives static destruction so late detector teardown can
// return reservations safely.
State& state() {
    static State* value = new State();
    return *value;
}

}  // namespace

std::string human(std::size_t bytes) {
    std::ostringstream os;
    os.setf(std::ios::fixed);
    os.precision(1);

    if (bytes >= (1ull << 30)) {
        os << static_cast<double>(bytes) / (1ull << 30) << " GiB";
    } else if (bytes >= (1ull << 20)) {
        os << static_cast<double>(bytes) / (1ull << 20) << " MiB";
    } else if (bytes >= (1ull << 10)) {
        os << static_cast<double>(bytes) / (1ull << 10) << " KiB";
    } else {
        os << bytes << " B";
    }
    return os.str();
}

std::size_t parse_byte_size(const std::string& text) {
    const std::size_t begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return 0;
    const std::size_t end = text.find_last_not_of(" \t\r\n");
    const std::string value_text = text.substr(begin, end - begin + 1);

    std::size_t pos = 0;
    std::size_t value = 0;
    while (pos < value_text.size() &&
           std::isdigit(static_cast<unsigned char>(value_text[pos]))) {
        const std::size_t digit = static_cast<std::size_t>(value_text[pos] - '0');
        if (value > (std::numeric_limits<std::size_t>::max() - digit) / 10) {
            throw std::runtime_error("rfdetr: byte size out of range: '" + value_text + "'");
        }
        value = value * 10 + digit;
        ++pos;
    }
    if (pos == 0) {
        throw std::runtime_error("rfdetr: byte size must start with a digit: '" +
                                 value_text + "'");
    }

    std::string suffix = value_text.substr(pos);
    for (char& c : suffix) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    std::size_t multiplier = 1;
    if (suffix.empty() || suffix == "b") {
        multiplier = 1;
    } else if (suffix == "k" || suffix == "kb" || suffix == "kib") {
        multiplier = 1ull << 10;
    } else if (suffix == "m" || suffix == "mb" || suffix == "mib") {
        multiplier = 1ull << 20;
    } else if (suffix == "g" || suffix == "gb" || suffix == "gib") {
        multiplier = 1ull << 30;
    } else {
        throw std::runtime_error("rfdetr: unknown byte size suffix in '" + value_text +
                                 "' (expected K, M, or G)");
    }

    if (multiplier != 1 &&
        value > std::numeric_limits<std::size_t>::max() / multiplier) {
        throw std::runtime_error("rfdetr: byte size out of range: '" + value_text + "'");
    }
    return value * multiplier;
}

std::size_t limit() {
    State& value = state();
    std::lock_guard<std::mutex> lock(value.m);
    if (!value.limit_read) {
        value.limit_read = true;
        if (const char* environment = std::getenv("RFDETR_GPU_MEM_LIMIT")) {
            try {
                value.limit = parse_byte_size(environment);
            } catch (const std::exception& error) {
                // Preserve the failure. A later detector-creation retry must not
                // silently turn a malformed safety limit into unlimited memory.
                value.limit_error = std::string("RFDETR_GPU_MEM_LIMIT: ") + error.what();
            }
        }
    }
    if (!value.limit_error.empty()) {
        throw std::runtime_error(value.limit_error);
    }
    return value.limit;
}

void reserve(std::size_t bytes, const char* what) {
    if (bytes == 0) return;
    const std::size_t cap = limit();
    State& value = state();
    std::lock_guard<std::mutex> lock(value.m);

    if (cap != 0 && (value.in_use > cap || bytes > cap - value.in_use)) {
        const std::size_t remaining = value.in_use < cap ? cap - value.in_use : 0;
        throw std::runtime_error(
            std::string("rfdetr: GPU memory limit exceeded: ") + what + " wants " +
            human(bytes) + ", " + human(value.in_use) + " already in use, limit " +
            human(cap) + ", " + human(remaining) + " remaining.\n" +
            "Raise RFDETR_GPU_MEM_LIMIT or use a smaller engine.");
    }

    if (bytes > std::numeric_limits<std::size_t>::max() - value.in_use) {
        throw std::runtime_error("rfdetr: GPU memory accounting overflow");
    }
    value.in_use += bytes;
    if (value.in_use > value.peak) value.peak = value.in_use;
}

void release(std::size_t bytes) noexcept {
    if (bytes == 0) return;
    State& value = state();
    std::lock_guard<std::mutex> lock(value.m);
    value.in_use = bytes > value.in_use ? 0 : value.in_use - bytes;
}

std::size_t in_use() noexcept {
    State& value = state();
    std::lock_guard<std::mutex> lock(value.m);
    return value.in_use;
}

std::size_t peak() noexcept {
    State& value = state();
    std::lock_guard<std::mutex> lock(value.m);
    return value.peak;
}

void reset_for_testing(std::size_t limit_bytes) noexcept {
    State& value = state();
    std::lock_guard<std::mutex> lock(value.m);
    value.in_use = 0;
    value.peak = 0;
    value.limit = limit_bytes;
    value.limit_read = true;
    value.limit_error.clear();
}

void reset_from_environment_for_testing() noexcept {
    State& value = state();
    std::lock_guard<std::mutex> lock(value.m);
    value.in_use = 0;
    value.peak = 0;
    value.limit = 0;
    value.limit_read = false;
    value.limit_error.clear();
}

}  // namespace rfdetr::gpu_budget
