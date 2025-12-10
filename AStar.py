from queue import PriorityQueue
graph = {
    'A': [('B', 2), ('C', 1)],
    'B': [('D', 7), ('E', 3)],
    'C': [('F', 6)],
    'D': [('G', 5)],
    'E': [('H', 3)],
    'F': [('H', 3)],
    'H': [('G', 1)],
    'G': []
}
heuristic = {
    'A': 9,
    'B': 5,
    'C': 4,
    'D': 6,
    'E': 3,
    'F': 7,
    'H': 1,
    'G': 0
}
def a_star(graph, start, goal, heuristic):
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0
    while not open_set.empty():
        current = open_set.get()[1]
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        for (neighbor, cost) in graph[current]:
            tentative_g = g_score[current] + cost
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic[neighbor]
                open_set.put((f_score, neighbor))
    return None
path = a_star(graph, 'A', 'G', heuristic)
print("Shortest path using A*:", path)
