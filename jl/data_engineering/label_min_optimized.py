def preprocess_future_indices(accesses):
    next_access = {}
    last_occurrence = {}
    for index in reversed(range(len(accesses))):
        access = accesses[index]
        if access in last_occurrence:
            next_access[index] = last_occurrence[access]
        else:
            next_access[index] = float("inf")
        last_occurrence[access] = index
    return next_access


def get_beladys(accesses, cache_size):
    future_indices = preprocess_future_indices(accesses)
    cache = {}
    decisions = []

    for i, access in enumerate(accesses):
        if i % 1000000 == 0:
            print(f"Processing access {i} of {len(accesses)}")
        next_use = future_indices[i]

        # Update or set the next use in the cache dictionary
        if access in cache:
            cache[access] = next_use
            decisions.append("In Cache")
        elif len(cache) < cache_size:
            # There's space in the cache
            cache[access] = next_use
            decisions.append("Cached")
        else:
            # Cache is full, need to potentially evict the farthest item
            farthest_item = max(cache, key=cache.get)
            if cache[farthest_item] > next_use:
                # Evict the farthest item
                del cache[farthest_item]
                cache[access] = next_use
                decisions.append("Cached")
            else:
                decisions.append("Not Cached")

    return decisions


if __name__ == "__main__":
    # Example usage
    access_sequence = [5, 1, 3, 5, 2, 4, 1, 5, 5, 1, 3, 5, 2, 3, 4, 5]
    cache_size = 3
    result = get_beladys(access_sequence, cache_size)
    for res in result:
        print(res)
