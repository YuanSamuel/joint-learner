#include <fstream>
#include <string>

class DATACOLLECTOR {
    std::ofstream cache_access_log;
    std::ofstream prefetch_log;

public:
    DATACOLLECTOR() {
        cache_access_log.open("cache_accesses.csv", std::ios::out);
        prefetch_log.open("prefetches.csv", std::ios::out);

        cache_access_log << "ip" << "," << "full_addr" << "," << "hit" << "\n";
        prefetch_log << "ip" << "," << "full_addr" << "," << "hit" << "\n";
    }

    ~DATACOLLECTOR() {
        if (cache_access_log.is_open()) {
            cache_access_log.close();
        }

        if (prefetch_log.is_open()) {
            prefetch_log.close();
        }
    }

    void log_cache_event(uint64_t ip, uint64_t full_addr, uint8_t hit) {
        if (cache_access_log.is_open()) {
            cache_access_log << ip << "," << full_addr << "," << (hit != 1 ? "HIT" : "MISS") << "\n";
        }
    }

    void log_prefetch_event(uint64_t ip, uint64_t full_addr, uint8_t hit, uint8_t useful_prefetch) {
        if (cache_access_log.is_open()) {
            cache_access_log << ip << "," << full_addr << "," << (hit != 1 ? "HIT" : "MISS") << "," << (useful_prefetch != 1 ? "true" : "false") << "\n";
        }
    }
};
