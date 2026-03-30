#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdint>
#include <iostream>
#include <functional>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./tiny_llama <path_to_gguf>" << std::endl;
        return 1;
    }

    // 打開檔案
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) {
        std::cerr << "Failed to open file" << std::endl;
        return 1;
    }

    // 取得檔案大小
    struct stat st;
    fstat(fd, &st);
    size_t file_size = st.st_size;
    std::cout << "File size: " << file_size << " bytes" << std::endl;

    // mmap
    void* data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) {
        std::cerr << "mmap failed" << std::endl;
        close(fd);
        return 1;
    }

    // 讀 magic number
    uint8_t* ptr = (uint8_t*)data;
    std::cout << "Magic: "
              << (char)ptr[0] << (char)ptr[1]
              << (char)ptr[2] << (char)ptr[3]
              << std::endl;
    
    uint32_t version = *(uint32_t*)(ptr + 4);
    std::cout << "Version: " << version << std::endl;

    uint64_t tensor_count = *(uint64_t*)(ptr + 8);
    std::cout << "Tensor Count: " << tensor_count << std::endl;

    uint64_t kv_count = *(uint64_t*)(ptr + 16);
    std::cout << "Metadata KV Count: " << kv_count << std::endl;

    size_t offset = 24;

    // 定義 GGUF value types
    enum GGUFType : uint32_t {
        UINT8   = 0,
        INT8    = 1,
        UINT16  = 2,
        INT16   = 3,
        UINT32  = 4,
        INT32   = 5,
        FLOAT32 = 6,
        BOOL    = 7,
        STRING  = 8,
        ARRAY   = 9,
        UINT64  = 10,
        INT64   = 11,
        FLOAT64 = 12,
    };

    // 讀一個 string 的 lambda
    auto read_string = [&]() -> std::string {
        uint64_t len = *(uint64_t*)(ptr + offset);
        offset += 8;
        std::string s((char*)(ptr + offset), len);
        offset += len;
        return s;
    };

    // 跳過一個 value 的 lambda（先只跳過，不印）
    std::function<void(uint32_t)> skip_value = [&](uint32_t type) {
        switch (type) {
            case UINT8: case INT8: case BOOL: offset += 1; break;
            case UINT16: case INT16:          offset += 2; break;
            case UINT32: case INT32: case FLOAT32: offset += 4; break;
            case UINT64: case INT64: case FLOAT64: offset += 8; break;
            case STRING: read_string(); break;
            case ARRAY: {
                uint32_t elem_type = *(uint32_t*)(ptr + offset); offset += 4;
                uint64_t count     = *(uint64_t*)(ptr + offset); offset += 8;
                for (uint64_t i = 0; i < count; i++)
                    skip_value(elem_type);
                break;
            }
        }
    };

    // 逐一解析 metadata，只印 key
    for (uint64_t i = 0; i < kv_count; i++) {
        std::string key = read_string();
        uint32_t type   = *(uint32_t*)(ptr + offset); offset += 4;
        std::cout << "  [" << i << "] " << key << std::endl;
        skip_value(type);
    }

    std::cout << "Metadata end offset: " << offset << std::endl;

    // 讀每個 tensor info
    std::cout << "\n=== Tensor Info ===" << std::endl;
    for (uint64_t i = 0; i < tensor_count; i++) {
        std::string name = read_string();

        uint32_t n_dims = *(uint32_t*)(ptr + offset); offset += 4;

        std::string shape_str = "[";
        for (uint32_t d = 0; d < n_dims; d++) {
            uint64_t dim = *(uint64_t*)(ptr + offset); offset += 8;
            shape_str += std::to_string(dim);
            if (d < n_dims - 1) shape_str += ", ";
        }
        shape_str += "]";

        uint32_t type   = *(uint32_t*)(ptr + offset); offset += 4;
        uint64_t toffset = *(uint64_t*)(ptr + offset); offset += 8;

        std::cout << "  " << name
                << "  shape=" << shape_str
                << "  type=" << type
                << "  offset=" << toffset
                << std::endl;
    }

    // 清理
    munmap(data, file_size);
    close(fd);
    return 0;
}