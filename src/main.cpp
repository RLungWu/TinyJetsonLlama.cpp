#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdint>
#include <iostream>

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

    // 清理
    munmap(data, file_size);
    close(fd);
    return 0;
}