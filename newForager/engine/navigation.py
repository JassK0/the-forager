"""Route finding around water, trees and rock.

Animals normally head straight for their goal (with a little steering noise
from movement.py). When the straight line is blocked they plan a route with
A* over the walkable cells: 8-connected, never cutting a blocked corner, with
the octile distance as the heuristic, so the route found is the shortest one.
Searches are capped, so a goal that is unreachable, or only reachable by a
long detour, is reported as unreachable rather than costing unbounded time.
"""

import heapq
import math

SQRT2 = math.sqrt(2.0)
_STEPS = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
          (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2))


def find_path(walk, width: int, height: int, sx: float, sy: float, gx: float, gy: float,
              max_cost: float = math.inf, max_nodes: int = 4000):
    """Shortest route from (sx, sy) to (gx, gy).

    Returns ``(waypoints, length)`` where waypoints are cell centres leading to
    the goal (the last one is the goal itself), or None if there is no route
    no longer than ``max_cost`` within the search budget.
    """
    start = int(sy) * width + int(sx)
    goal = int(gy) * width + int(gx)
    if not (0 <= start < width * height and 0 <= goal < width * height) or not walk[goal]:
        return None
    if start == goal:
        return [(gx, gy)], math.hypot(gx - sx, gy - sy)
    tx, ty = goal % width, goal // width

    def h(i: int) -> float:
        dx, dy = abs(i % width - tx), abs(i // width - ty)
        return dx + dy + (SQRT2 - 2.0) * min(dx, dy)

    best = {start: 0.0}
    came = {}
    frontier = [(h(start), 0.0, start)]
    expanded = 0
    while frontier:
        f, g, i = heapq.heappop(frontier)
        if i == goal:
            break
        if g > best.get(i, math.inf):
            continue
        if f > max_cost:
            return None
        expanded += 1
        if expanded > max_nodes:
            return None
        x, y = i % width, i // width
        for dx, dy, cost in _STEPS:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            j = ny * width + nx
            if not walk[j]:
                continue
            if dx and dy and not (walk[y * width + nx] and walk[ny * width + x]):
                continue
            ng = g + cost
            if ng < best.get(j, math.inf):
                best[j] = ng
                came[j] = i
                heapq.heappush(frontier, (ng + h(j), ng, j))
    else:
        return None
    if goal not in came:
        return None
    cells = []
    i = goal
    while i != start:
        cells.append(i)
        i = came[i]
    cells.reverse()
    waypoints = [(c % width + 0.5, c // width + 0.5) for c in cells]
    waypoints[-1] = (gx, gy)
    return waypoints, best[goal]
