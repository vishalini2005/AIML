from collections import deque
class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
    def addEdge(self, u, v):
        self.graph[u].append(v)
    def BFS(self, s):
        visited = set()
        queue = []
        queue.append(s)
        visited.add(s)
        while queue:
            s = queue.pop(0)
            print(s, end=" ")

            for v in self.graph[s]:
                if v not in visited:
                    queue.append(v)
                    visited.add(v)
g = Graph()
g.addEdge('a', 'b')
g.addEdge('b', 'd')
g.addEdge('a', 'c')
print("Following is Breadth First traversal (starting from vertex 'a'):")
g.BFS('a')
