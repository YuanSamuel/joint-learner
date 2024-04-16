#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
namespace fs = std::filesystem;

class DATACOLLECTOR {
    std::ofstream cache_access_log;
    std::ofstream prefetch_log;

   public:
    DATACOLLECTOR() {
        if (!fs::exists("collector_output")) {
            fs::create_directory("collector_output");
        }

        cache_access_log.open("collector_output/cache_accesses.csv", std::ios::out);
        prefetch_log.open("collector_output/prefetches.csv", std::ios::out);

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
                     << "\n";
    }

    ~DATACOLLECTOR() {
        if (cache_access_log.is_open()) {
            cache_access_log.close();
        }

        if (prefetch_log.is_open()) {
            prefetch_log.close();
        }
    }

    void log_cache_event(uint32_t triggering_cpu, uint32_t set, uint32_t way, uint64_t full_addr, uint64_t ip, uint64_t victim_addr, uint32_t type, uint8_t hit) {
        if (cache_access_log.is_open()) {
            cache_access_log << triggering_cpu << "," << set << "," << way << "," << full_addr << "," << ip << "," << victim_addr << "," << type << "," << static_cast<unsigned int>(hit)
                             << "\n";
        } else {
            std::cout << "Cache access log is not open\n";
        }
    }

    void log_prefetch_event(uint64_t addr, uint64_t ip, uint8_t cache_hit, bool useful_prefetch, uint8_t type) {
        if (prefetch_log.is_open()) {
            prefetch_log << addr << "," << ip << "," << static_cast<unsigned int>(cache_hit) << "," << useful_prefetch << "," << static_cast<unsigned int>(type) << "\n";
        }
    }
};
