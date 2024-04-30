#include <assert.h>
#include <math.h>

#include <iostream>
#include <map>

#include "cache.h"
#include "channel.h"
#include "glider_predictor.h"
#include "optgen.h"

// Num cores in the system
#define NUM_CORE 1
// Total num LLC sets
#define LLC_SETS NUM_CORE * 2048
// Num ways in each set
#define LLC_WAYS 16

// RRIP -
#define MAXRRIP 7
#define MIDRRIP 2

// Store RRIP value for each way for each set
uint32_t rrpv[LLC_SETS][LLC_WAYS];

#define OPTGEN_VECTOR_SIZE 128

struct rrpv_structure {
    int way;
    int rrpv_val;
    bool flag;
};

struct rrpv_structure rrpv_str[LLC_WAYS];

// Glider predictor
Glider_Predictor* demand_predictor;
Glider_Predictor* prefetch_predictor;

// Per-set timers; we only use 64 of these
// Budget = 64 sets * 1 timer per set * 10 bits per timer = 80 bytes
#define TIMER_SIZE 1024
uint64_t perset_mytimer[LLC_SETS];

// Signatures for sampled sets; we only use 64 of these
// Budget = 64 sets * 16 ways * 12-bit signature per line = 1.5B
uint64_t signatures[LLC_SETS][LLC_WAYS];
bool prefetched[LLC_SETS][LLC_WAYS];

// #define OPTGEN_VECTOR_SIZE 128

OPTgen perset_optgen[LLC_SETS];  // per-set occupancy vectors; we only use 64 of these

#define bitmask(l) (((l) == 64) ? (unsigned long long)(-1LL) : ((1LL << (l)) - 1LL))
// Extract l-length bits from i to x
#define bits(x, i, l) (((x) >> (i)) & bitmask(l))
// Sample 64 ests per core, using bit ops to determine whether a set has been sampled
#define SAMPLED_SET(set) (bits(set, 0, 6) == bits(set, ((unsigned long long)log2(LLC_SETS) - 6), 6))

// Sampler to track 8x cache history for sampled sets
// 2800 entris * 4 bytes per entry = 11.2KB
#define SAMPLED_CACHE_SIZE 2800
#define SAMPLER_WAYS 8
#define SAMPLER_SETS SAMPLED_CACHE_SIZE / SAMPLER_WAYS
vector<map<uint64_t, ADDR_INFO>> addr_history;  // Sampler

uint8_t PREFETCH = static_cast<std::underlying_type_t<access_type>>(access_type::PREFETCH);

void CACHE::initialize_replacement() {
    cout << "Initialize Glider replacement policy state" << endl;

    for (int i = 0; i < LLC_SETS; i++) {
        for (int j = 0; j < LLC_WAYS; j++) {
            rrpv[i][j] = MAXRRIP;
            signatures[i][j] = 0;
            prefetched[i][j] = false;
        }
        // Initialize timestamps in sample all to 0
        perset_mytimer[i] = 0;
        perset_optgen[i].init(LLC_WAYS - 2);
    }

    addr_history.resize(SAMPLER_SETS);
    for (int i = 0; i < SAMPLER_SETS; i++) {
        addr_history[i].clear();
    }

    demand_predictor = new Glider_Predictor();
    prefetch_predictor = new Glider_Predictor();

    cout << "Finished initializing Glider replacement policy state" << endl;
}

// This function is called when a tag is checked in the cache. The parameters passed are:
// * triggering_cpu: the core index that initiated this fill
// * instr_id: an instruction count that can be used to examine the program order of requests.
// * set: the set that the fill occurred in.
// * current_set: a pointer to the beginning of the set being accessed.
// * ip: the address of the instruction that initiated the demand. If the packet is a prefetch from another level, this value will be 0.
// * addr: the address of the packet. If this is the first-level cache, the offset bits are included. Otherwise, the offset bits are zero. If the cache was
// configured with `"virtual_prefetch": true`, this address will be a virtual address. Otherwise, this is a physical address.
// * type: the result of `static_cast<std::underlying_type_t<access_type>>(v)`
// The function should return the way index that should be evicted, or `this->NUM_WAY` to indicate that a bypass should occur.
uint32_t CACHE::find_victim(uint32_t triggering_cpu, uint64_t instr_id, uint32_t set, const BLOCK* current_set, uint64_t ip, uint64_t full_addr, uint32_t type) {
    // Find victim from low RRPV value (7)
    for (uint32_t i = 0; i < LLC_WAYS; i++) {
        if (rrpv[set][i] == MAXRRIP) {
            return i;
        }
    }

    // If we cannot find a cache-averse line, we evict the oldest cache-friendly line
    uint32_t max_rrip = 0;
    int32_t lru_victim = -1;
    for (uint32_t i = 0; i < LLC_WAYS; i++) {
        if (rrpv[set][i] >= max_rrip) {
            max_rrip = rrpv[set][i];
            lru_victim = i;
        }
    }

    assert(lru_victim != -1);

    // The predictor is trained negatively on LRU evictions
    if (SAMPLED_SET(set)) {
        // if(prefetched[set][lru_victim])
        //     prefetch_predictor->decrement(signatures[set][lru_victim]);
        // else
        //     demand_predictor->decrement(signatures[set][lru_victim]);
        demand_predictor->decrement(signatures[set][lru_victim]);

        // Debug
        // demand_predictor->print_all_weights();
    }

    return lru_victim;
}

void replace_addr_history_element(unsigned int sampler_set) {
    uint64_t lru_addr = 0;

    for (map<uint64_t, ADDR_INFO>::iterator it = addr_history[sampler_set].begin(); it != addr_history[sampler_set].end(); it++) {
        //     uint64_t timer = (it->second).last_quanta;

        if ((it->second).lru == (SAMPLER_WAYS - 1)) {
            // lru_time =  (it->second).last_quanta;
            lru_addr = it->first;
            break;
        }
    }

    addr_history[sampler_set].erase(lru_addr);
}

// Updates cache history with current value for the given sample set
// Increments LRU for all cache history entries whose LRU value is less than currentVal
void update_addr_history_lru(unsigned int sampler_set, unsigned int curr_lru) {
    // Traverse cache history in given sample collection
    for (map<uint64_t, ADDR_INFO>::iterator it = addr_history[sampler_set].begin(); it != addr_history[sampler_set].end(); it++) {
        if ((it->second).lru < curr_lru) {
            (it->second).lru++;
            assert((it->second).lru < SAMPLER_WAYS);
        }
    }
}


uint32_t sampled = 0;
uint32_t inced = 0;
uint32_t deced = 0;
uint32_t newval = 0;

// This function is called when a hit occurs or a miss is filled in the cache. The parameters passed are:
// * triggering_cpu: the core index that initiated this fill
// * set: the set that the fill occurred in.
// * way: the way that the fill occurred in.
// * full_addr: the address of the packet. If this is the first-level cache, the offset bits are included. Otherwise, the offset bits are zero. If the cache was
// configured with `"virtual_prefetch": true`, this address will be a virtual address. Otherwise, this is a physical address.
// * ip: the address of the instruction that initiated the demand. If the packet is a prefetch from another level, this value will be 0.
// * victim_addr: the address of the evicted block, if this is a miss. If this is a hit, the value is 0.
// * type: the result of `static_cast<std::underlying_type_t<access_type>>(v)
// * hit: indicates whether the access resulted in a cache hit
// The function should return metadata that will be stored alongside the block.
void CACHE::update_replacement_state(uint32_t triggering_cpu, uint32_t set, uint32_t way, uint64_t full_addr, uint64_t ip, uint64_t victim_addr, uint32_t type,
                                     uint8_t hit) {
    // Align address to cache line boundaries
    full_addr = (full_addr >> 6) << 6;

    if (type == PREFETCH) {
        if (!hit)
            prefetched[set][way] = true;
    } else
        prefetched[set][way] = false;

    // Ignore writebacks
    if (type == static_cast<std::underlying_type_t<access_type>>(access_type::WRITE))
        return;

    // If OPT selects current cache set for sampling
    if (SAMPLED_SET(set)) {
        // Get timestamp of current cache set modulo OPTGEN_VECTOR_SIZE
        uint64_t curr_quanta = perset_mytimer[set] % OPTGEN_VECTOR_SIZE;

        // Get CRC hash modulo 256 to get sample tag
        uint32_t sampler_set = (full_addr >> 6) % SAMPLER_SETS;
        // Get sample set by ignoring cache line info then modulo sampler_sets
        uint64_t sampler_tag = CRC(full_addr >> 12) % 256;

        assert(sampler_set < SAMPLER_SETS);

        sampled++;

        // This line has been used before. Since the right end of a usage interval is always
        // a demand, ignore prefetches
        if ((addr_history[sampler_set].find(sampler_tag) != addr_history[sampler_set].end()) && (type != PREFETCH)) {
            unsigned int curr_timer = perset_mytimer[set];
            // Account for time wraparound
            if (curr_timer < addr_history[sampler_set][sampler_tag].last_quanta)
                curr_timer = curr_timer + TIMER_SIZE;

            bool wrap = ((curr_timer - addr_history[sampler_set][sampler_tag].last_quanta) > OPTGEN_VECTOR_SIZE);
            uint64_t last_quanta = addr_history[sampler_set][sampler_tag].last_quanta % OPTGEN_VECTOR_SIZE;

            // and for prefetch hits, we train the last prefetch trigger PC
            if (!wrap && perset_optgen[set].should_cache(curr_quanta, last_quanta)) {
                // if(addr_history[sampler_set][sampler_tag].prefetched)
                //     prefetch_predictor->increment(addr_history[sampler_set][sampler_tag].PC);
                // else
                //     demand_predictor->increment(addr_history[sampler_set][sampler_tag].PC);
                inced++;
                demand_predictor->increment(addr_history[sampler_set][sampler_tag].PC);
            } else {
                // Train the predictor negatively because OPT would not have cached this line
                //  if(addr_history[sampler_set][sampler_tag].prefetched)
                //      prefetch_predictor->decrement(addr_history[sampler_set][sampler_tag].PC);
                //  else
                //      demand_predictor->decrement(addr_history[sampler_set][sampler_tag].PC);
                deced++;
                demand_predictor->decrement(addr_history[sampler_set][sampler_tag].PC);
            }

            // predictor_demand->print_all_weights();

            // Some maintenance operations for OPTgen
            perset_optgen[set].add_access(curr_quanta);
            update_addr_history_lru(sampler_set, addr_history[sampler_set][sampler_tag].lru);

            // Since this was a demand access, mark the prefetched bit as false
            addr_history[sampler_set][sampler_tag].prefetched = false;
        }
        // This is the first time we are seeing this line (could be demand or prefetch)
        else if (addr_history[sampler_set].find(sampler_tag) == addr_history[sampler_set].end()) {

            newval++;
            // Find a victim from the sampled cache if we are sampling
            if (addr_history[sampler_set].size() == SAMPLER_WAYS)
                replace_addr_history_element(sampler_set);

            assert(addr_history[sampler_set].size() < SAMPLER_WAYS);

            // Initialize a new entry in the sampler
            addr_history[sampler_set][sampler_tag].init(curr_quanta);
            // If it's a prefetch, mark the prefetched bit;
            if (type == PREFETCH) {
                addr_history[sampler_set][sampler_tag].mark_prefetch();
                perset_optgen[set].add_prefetch(curr_quanta);
            } else
                perset_optgen[set].add_access(curr_quanta);
            update_addr_history_lru(sampler_set, SAMPLER_WAYS - 1);
        }
        // This line is a prefetch
        else {
            assert(addr_history[sampler_set].find(sampler_tag) != addr_history[sampler_set].end());
            // if(hit && prefetched[set][way])
            uint64_t last_quanta = addr_history[sampler_set][sampler_tag].last_quanta % OPTGEN_VECTOR_SIZE;
            if (perset_mytimer[set] - addr_history[sampler_set][sampler_tag].last_quanta < 5 * NUM_CORE) {
                if (perset_optgen[set].should_cache(curr_quanta, last_quanta)) {
                    if (addr_history[sampler_set][sampler_tag].prefetched)
                        prefetch_predictor->increment(addr_history[sampler_set][sampler_tag].PC);
                    else
                        demand_predictor->increment(addr_history[sampler_set][sampler_tag].PC);
                }
            }

            // Mark the prefetched bit
            addr_history[sampler_set][sampler_tag].mark_prefetch();
            // Some maintenance operations for OPTgen
            perset_optgen[set].add_prefetch(curr_quanta);
            update_addr_history_lru(sampler_set, addr_history[sampler_set][sampler_tag].lru);
        }

        cout << "Sampled: " << sampled << " Inced: " << inced << " Deced: " << deced << " new: " << newval << endl;

        // Get Hawkeye's prediction for this line
        Prediction new_prediction = demand_predictor->get_prediction(ip);
        if (type == PREFETCH)
            new_prediction = prefetch_predictor->get_prediction(ip);
        // Update the sampler with the timestamp, PC and our prediction
        // For prefetches, the PC will represent the trigger PC
        addr_history[sampler_set][sampler_tag].update(perset_mytimer[set], ip);
        addr_history[sampler_set][sampler_tag].lru = 0;
        // Increment the set timer
        perset_mytimer[set] = (perset_mytimer[set] + 1) % TIMER_SIZE;
    }

    Prediction new_prediction = demand_predictor->get_prediction(ip);
    if (type == PREFETCH)
        new_prediction = prefetch_predictor->get_prediction(ip);

    signatures[set][way] = ip;

    // Update RRIP value for the current set and way
    if (new_prediction == Prediction::Low) {
        rrpv[set][way] = MAXRRIP;
    } else if (new_prediction == Prediction::Medium) {
        rrpv[set][way] = MIDRRIP;
    }
    // High Priority
    else {
        rrpv[set][way] = 0;
        // Access is a miss, check RRPV values of rows are sturated
        if (!hit) {
            bool isMaxVal = false;
            for (uint32_t i = 0; i < LLC_WAYS; i++) {
                if (rrpv[set][i] == MIDRRIP - 1) {
                    isMaxVal = true;
                }
            }

            // Age cache lines
            for (uint32_t i = 0; i < LLC_WAYS; i++) {
                if (!isMaxVal && rrpv[set][i] < MIDRRIP - 1) {
                    rrpv[set][i]++;
                }
            }
        }
        // Set RRIP of current way to 0 (recently used)
        rrpv[set][way] = 0;
    }
}

void CACHE::replacement_final_stats() {}
