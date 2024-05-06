#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

class DATACOLLECTOR {
   public:
    static std::ofstream cache_access_log;
    static std::ofstream prefetch_log;
    static std::string run_name;

    // DATACOLLECTOR() {
    //     if (!fs::exists("collector_output")) {
    //         fs::create_directory("collector_output");
    //     }
    // }

    // ~DATACOLLECTOR() {
    //     if (cache_access_log.is_open()) {
    //         cache_access_log.close();
    //     }

    //     if (prefetch_log.is_open()) {
    //         prefetch_log.close();
    //     }
    // }

    static void setName(const std::string& name) {
        if (!fs::exists("collector_output")) {
            fs::create_directory("collector_output");
        }

        run_name = name;

        cache_access_log.open("collector_output/cache_accesses_" + run_name + ".csv", std::ios::out);
        prefetch_log.open("collector_output/prefetches_" + run_name + ".csv", std::ios::out);

        cache_access_log << "triggering_cpu"
                         << ","
                         << "set"
                         << ","
                         << "way"
                         << ","
                         << "full_addr"
                         << ","
                         << "ip"
                         << ","
                         << "victim_addr"
                         << ","
                         << "type"
                         << ","
                         << "hit"
                         << ","
                         << "timestamp"
                         << "\n";
        prefetch_log << "addr"
                     << ","
                     << "ip"
                     << ","
                     << "cache_hit"
                     << ","
                     << "useful_prefetch"
                     << ","
                     << "type"
                     << ","
                     << "timestamp"
                     << "\n";
    }

    static uint64_t timeSinceEpochMicrosec() {
        return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }

    static void log_cache_event(uint32_t triggering_cpu, uint32_t set, uint32_t way, uint64_t full_addr, uint64_t ip, uint64_t victim_addr, uint32_t type, uint8_t hit) {
        uint64_t time = timeSinceEpochMicrosec();
        if (cache_access_log.is_open()) {
            cache_access_log << triggering_cpu << "," << set << "," << way << ","
                             << full_addr << "," << ip << "," << victim_addr << "," << type << ","
                             << static_cast<unsigned int>(hit) << "," << time << "\n";
        } else {
            std::cout << "Cache access log is not open\n";
        }
    }

    static void log_prefetch_event(uint64_t addr, uint64_t ip, uint8_t cache_hit, bool useful_prefetch, uint8_t type) {
        uint64_t time = timeSinceEpochMicrosec();
        if (prefetch_log.is_open()) {
            prefetch_log << addr << "," << ip << ","
                         << static_cast<unsigned int>(cache_hit) << ","
                         << useful_prefetch << "," << static_cast<unsigned int>(type) << ","
                         << time << "\n";
        }
    }
};
