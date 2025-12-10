from queue import PriorityQueue 
graph = { 
    'S': [('A', 4), ('B', 3), ('C', 5)], 
    'A': [('G', 2)], 
    'B': [('G', 2)], 
    'C': [('G', 1)], 
} 
 
heuristic = { 
    'S': 3, 
    'A': 0, 
    'B': 2, 
    'C': 1, 
    'G': 0 
} 
 
MEMORY_LIMIT = 3 
def sma_star_limited_memory(start, goal): 
    open_set = PriorityQueue() 
    memory = {} 
     
    f_start = heuristic[start] 
    open_set.put((f_start, [start], 0)) 
    memory[start] = f_start  
 
    while not open_set.empty(): 
        f, path, g = open_set.get() 
        current = path[-1] 
 
        if current == goal: 
            print("Total cost:", g)  
            return path 
 
        for neighbor, cost in graph.get(current, []): 
            if neighbor not in memory: 
                if len(memory) >= MEMORY_LIMIT: 
                    # Remove node with highest f that is NOT in current path 
                    candidates = [k for k in memory if k not in path] 
                    if candidates: 
                        worst_node = max(candidates, key=lambda k: memory[k]) 
                        del memory[worst_node] 
                    else: 
                        # If no safe node to evict, skip adding this neighbor 
                        continue 
 
            new_g = g + cost 
            new_f = new_g + heuristic[neighbor] 
            new_path = path + [neighbor] 
            open_set.put((new_f, new_path, new_g)) 
            memory[neighbor] = new_f             
    return None 
 
# Run the algorithm 
result = sma_star_limited_memory('S', 'G') 
print("Path found (SMA* with memory=3):", result)