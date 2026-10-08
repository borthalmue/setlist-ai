"""
Ordering a crate for the smoothest run of transitions is an open-path
traveling-salesman problem (minimize total adjacent-pair cost, start and end
don't connect back to each other), which is NP-hard for an exact solution, but
nearest-neighbor plus a 2-opt cleanup pass gets close enough for a set of the
size an actual DJ set is built from, and runs instantly.
"""


def nearest_neighbor_order(cost_matrix, start=0):
    n = len(cost_matrix)
    unvisited = set(range(n))
    unvisited.discard(start)
    order = [start]

    current = start
    while unvisited:
        nxt = min(unvisited, key=lambda j: cost_matrix[current][j])
        order.append(nxt)
        unvisited.discard(nxt)
        current = nxt

    return order


def path_cost(order, cost_matrix):
    return sum(cost_matrix[order[i]][order[i + 1]] for i in range(len(order) - 1))


def two_opt(order, cost_matrix, max_passes=50):
    """Repeatedly try reversing a segment of the route; keep the reversal if
    it shortens the total cost. Classic TSP local search, stops once a full
    pass makes no improvement."""
    order = list(order)
    n = len(order)
    improved = True
    passes = 0

    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                a, b = order[i], order[i + 1]
                c, d = order[j], order[(j + 1) % n] if j + 1 < n else None

                before = cost_matrix[a][b] + (cost_matrix[c][d] if d is not None else 0)
                after = cost_matrix[a][c] + (cost_matrix[b][d] if d is not None else 0)

                if after < before - 1e-9:
                    order[i + 1:j + 1] = reversed(order[i + 1:j + 1])
                    improved = True

    return order


def plan_set(tracks, cost_matrix, start=0):
    """Returns (track order as indices, total cost before 2-opt, total cost after)."""
    nn_order = nearest_neighbor_order(cost_matrix, start=start)
    nn_cost = path_cost(nn_order, cost_matrix)

    refined = two_opt(nn_order, cost_matrix)
    refined_cost = path_cost(refined, cost_matrix)

    return refined, nn_cost, refined_cost


if __name__ == "__main__":
    import random

    random.seed(0)
    n = 8
    m = [[0 if i == j else round(random.uniform(0.1, 2.0), 2) for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            m[j][i] = m[i][j]  # symmetric for this smoke test

    order, nn_cost, refined_cost = plan_set(list(range(n)), m)
    print("nearest-neighbor cost:", round(nn_cost, 2))
    print("after 2-opt:          ", round(refined_cost, 2))
    print("order:", order)
    assert refined_cost <= nn_cost + 1e-9, "2-opt should never make the route worse"
    print("ok: 2-opt did not regress the route")
