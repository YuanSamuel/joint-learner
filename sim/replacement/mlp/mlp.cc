#include <algorithm>
#include <cassert>
#include <map>
#include <vector>

#include "cache.h"
#include "datacollector.h"
#include "mlp_replacement.h"


MLP_REPLACEMENT* predictor;

namespace {
std::map<CACHE*, std::vector<uint64_t>> last_used_cycles;
}

void CACHE::initialize_replacement() {
    ::last_used_cycles[this] = std::vector<uint64_t>(NUM_SET * NUM_WAY);
    predictor = new MLP_REPLACEMENT("models/cache_repl_w5_d12_pb100m_2.pth_traced.pt");
}

uint32_t CACHE::find_victim(uint32_t triggering_cpu, uint64_t instr_id, uint32_t set, const BLOCK* current_set, uint64_t ip, uint64_t full_addr, uint32_t type) {
    auto begin = std::next(std::begin(::last_used_cycles[this]), set * NUM_WAY);
    auto end = std::next(begin, NUM_WAY);

    // Find the way whose last use cycle is most distant
    auto victim = std::min_element(begin, end);
    assert(begin <= victim);
    assert(victim < end);
    return static_cast<uint32_t>(std::distance(begin, victim));  // cast protected by prior asserts
}

void CACHE::update_replacement_state(uint32_t triggering_cpu, uint32_t set, uint32_t way, uint64_t full_addr, uint64_t ip, uint64_t victim_addr, uint32_t type,
                                     uint8_t hit) {

    std::vector<uint64_t> vect(0, 5);
    int output = predictor->forward(ip, vect);
    std::cout << "Output: " << output << std::endl;
    // Mark the way as being used on the current cycle
    if (!hit || access_type{type} != access_type::WRITE)  // Skip this for writeback hits
        ::last_used_cycles[this].at(set * NUM_WAY + way) = current_cycle;

    // cacheCollector.log_cache_event(triggering_cpu, set, way, full_addr, ip, victim_addr, type, hit);
}

void CACHE::replacement_final_stats() {}
