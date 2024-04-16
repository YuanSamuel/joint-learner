def beladys_decision_algorithm(accesses, cache_size):
    cache = []
    
    decisions = []

    for i, access in enumerate(accesses):
        future_accesses = accesses[i+1:]
        if access in cache:
            # Calculate future indices for all items currently in the cache
            future_indices = {item: (future_accesses.index(item) if item in future_accesses else float('inf')) for item in cache}
            # If the current access has a sooner use than any other in the cache or it's not the farthest, keep it cached
            if all(future_indices[access] <= idx for item, idx in future_indices.items()):
                decisions.append((access, 'Cached'))
                continue
            else:
                # If there are better candidates for caching, remove this one and decide later if it should be re-cached
                cache.remove(access)


        # For items not in the cache or removed from the cache, decide if they should be added
        if len(cache) < cache_size:
            cache.append(access)
            decisions.append((access, 'Cached'))
        else:
            # Find the item with the farthest next use
            future_indices = {item: (future_accesses.index(item) if item in future_accesses else float('inf')) for item in cache}
            item_to_evict = max(future_indices, key=future_indices.get)
            access_future_index = future_accesses.index(access) if access in future_accesses else float('inf')
            
            # Replace the item with the farthest next use if the current access is sooner
            if future_indices[item_to_evict] > access_future_index:
                cache.remove(item_to_evict)
                cache.append(access)
                decisions.append((access, 'Cached'))
            else:
                decisions.append((access, 'Not Cached'))

    return decisions

# Example usage
access_sequence = [5, 1, 3, 5, 2, 4, 1, 5, 5, 1, 3, 5, 2, 3, 4, 5]
cache_size = 2
result = beladys_decision_algorithm(access_sequence, cache_size)
for res in result:
    print(res)