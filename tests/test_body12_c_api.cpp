#undef NDEBUG
#include <cassert>
#include <cstring>

#include "rfdetr/c_api.h"

int main() {
    assert(rfdetr_body12_create(nullptr) == nullptr);
    assert(std::strstr(rfdetr_last_error(), "path") != nullptr);
    assert(rfdetr_body12_infer(nullptr, nullptr, 0, nullptr, 0, nullptr, 0,
                             nullptr, 0) == 0);
    assert(std::strstr(rfdetr_last_error(), "buffers") != nullptr);
    rfdetr_body12_destroy(nullptr);
}
