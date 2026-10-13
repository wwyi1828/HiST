from functools import lru_cache
import heapq

import numpy as np
import torch

def get_sorted_indices(pos):

    x = pos[:, 0]
    y = pos[:, 1]
    multiplier = y.max() + 1
    key = x * multiplier + y
    sorted_indices = torch.argsort(key)
    return sorted_indices

def map_to_integer_grid(coords, initial_radius=10, expansion_factor=1.1, max_attempts=1000):
    """Snap points to distinct cells in the original greedy processing order.

    Most points retain their nearest cell. Only collisions need a sequential
    search; dense occupancy arrays let those searches inspect a whole ring at
    once. Unusually sparse coordinate ranges use the original implementation
    so the optimization cannot allocate an enormous grid.
    """
    def fallback():
        return _map_to_integer_grid_sparse(
            coords, initial_radius, expansion_factor, max_attempts
        )
    if (len(coords) == 0 or not isinstance(initial_radius, (int, np.integer))
            or initial_radius < 0 or not isinstance(max_attempts, (int, np.integer))
            or max_attempts <= 0):
        return fallback()

    # Preserve the torch sort (including ties) in the input dtype and device.
    distances = torch.sum((coords - torch.round(coords))**2, dim=1)
    order = torch.argsort(distances).detach().cpu().numpy()
    points = coords.detach().to(device="cpu", dtype=torch.float64).numpy()
    # Outside this range converting integer coordinates to float64, or adding
    # candidate offsets, could lose precision relative to Python's integers.
    if not np.isfinite(points).all() or np.abs(points).max() >= 2**52:
        return fallback()
    nearest = np.rint(points).astype(np.int64)
    result = nearest.copy()
    count = len(coords)
    max_cells = min(4_000_000, max(65_536, 32 * count))
    margin = int(initial_radius) + 2
    lower = [int(nearest[:, axis].min()) - margin for axis in range(2)]
    upper = [int(nearest[:, axis].max()) + margin for axis in range(2)]
    shape = tuple(upper[axis] - lower[axis] + 1 for axis in range(2))
    if shape[0] * shape[1] > max_cells:
        return fallback()

    first = np.full(shape, count, dtype=np.int64)
    taken = np.zeros(shape, dtype=np.bool_)
    nearest_local = nearest - np.asarray(lower, dtype=np.int64)
    np.minimum.at(first, (nearest_local[order, 0], nearest_local[order, 1]),
                  np.arange(count, dtype=np.int64))
    earliest = first[nearest_local[order, 0], nearest_local[order, 1]]
    pending = np.flatnonzero(earliest < np.arange(count)).tolist()
    queued = np.zeros(count, dtype=np.bool_)
    queued[pending] = True
    heapq.heapify(pending)

    def ensure_bounds(minimum, maximum):
        nonlocal first, taken, lower, upper
        if all(lower[axis] <= minimum[axis] and maximum[axis] <= upper[axis]
               for axis in range(2)):
            return True
        new_lower = [min(lower[axis], int(minimum[axis])) for axis in range(2)]
        new_upper = [max(upper[axis], int(maximum[axis])) for axis in range(2)]
        new_shape = tuple(new_upper[axis] - new_lower[axis] + 1 for axis in range(2))
        if new_shape[0] * new_shape[1] > max_cells:
            return False
        new_first = np.full(new_shape, count, dtype=np.int64)
        new_taken = np.zeros(new_shape, dtype=np.bool_)
        old_region = tuple(slice(lower[axis] - new_lower[axis],
                                 upper[axis] - new_lower[axis] + 1)
                           for axis in range(2))
        new_first[old_region] = first
        new_taken[old_region] = taken
        first, taken, lower, upper = new_first, new_taken, new_lower, new_upper
        return True

    while pending:
        position = heapq.heappop(pending)
        index = int(order[position])
        cell = nearest[index]
        search_radius = int(initial_radius)
        selected = None
        for _ in range(max_attempts):
            if (2 * search_radius + 1)**2 > max_cells:
                return fallback()
            if not ensure_bounds(cell - search_radius, cell + search_radius):
                return fallback()
            candidates = cell + _grid_ring_offsets(search_radius)
            local = candidates - np.asarray(lower, dtype=np.int64)
            free = ((first[local[:, 0], local[:, 1]] >= position)
                    & ~taken[local[:, 0], local[:, 1]])
            candidates = candidates[free]
            if len(candidates):
                delta = points[index] - candidates
                candidate_distances = delta[:, 0]**2 + delta[:, 1]**2
                best = int(np.argmin(candidate_distances))
                # Python float **2 and NumPy's square can round differently.
                # Re-evaluate close minima with the original arithmetic and
                # order, so exact ties still pick the first visited cell.
                tolerance = 8 * np.spacing(max(1.0, float(candidate_distances[best])))
                close = np.flatnonzero(
                    candidate_distances <= candidate_distances[best] + tolerance
                )
                if len(close) > 1:
                    x, y = points[index].tolist()
                    best = min(close, key=lambda candidate: (
                        (x - int(candidates[candidate, 0]))**2
                        + (y - int(candidates[candidate, 1]))**2
                    ))
                selected = candidates[best]
                break
            search_radius = int(search_radius * expansion_factor)

        if selected is None:
            # Keep the original max_attempts fallback, including its x/y
            # scan order and its final cell when the whole box is occupied.
            minimum = [int(points[:, axis].min()) - 10 for axis in range(2)]
            maximum = [int(points[:, axis].max()) + 10 for axis in range(2)]
            if not ensure_bounds(minimum, [maximum[0] + 1, maximum[1]]):
                return fallback()
            region = tuple(slice(minimum[axis] - lower[axis],
                                 maximum[axis] - lower[axis] + 1)
                           for axis in range(2))
            free = (first[region] >= position) & ~taken[region]
            available = np.flatnonzero(free)
            if len(available):
                local_x, local_y = np.unravel_index(int(available[0]), free.shape)
                selected = np.asarray([minimum[0] + local_x, minimum[1] + local_y])
            else:
                selected = np.asarray([maximum[0] + 1, minimum[1]])

        result[index] = selected
        local_x, local_y = selected - np.asarray(lower, dtype=np.int64)
        taken[local_x, local_y] = True
        displaced = int(first[local_x, local_y])
        if position < displaced < count and not queued[displaced]:
            queued[displaced] = True
            heapq.heappush(pending, displaced)

    integer_coords = torch.as_tensor(result, dtype=torch.int64, device=coords.device)
    integer_coords[:, 0] -= torch.min(integer_coords[:, 0])
    integer_coords[:, 1] -= torch.min(integer_coords[:, 1])
    return integer_coords


@lru_cache(maxsize=4)
def _grid_ring_offsets(search_radius):
    """Offsets in the same visitation order as the original nested loops."""
    rings = []
    for radius in range(1, search_radius + 1):
        across = np.arange(-radius, radius + 1, dtype=np.int64)
        sides = np.arange(-radius + 1, radius, dtype=np.int64)
        rings.append(np.column_stack((np.repeat(across, 2),
                                      np.tile([-radius, radius], len(across)))))
        rings.append(np.column_stack((np.tile([-radius, radius], len(sides)),
                                      np.repeat(sides, 2))))
    return np.concatenate(rings) if rings else np.empty((0, 2), dtype=np.int64)


def _map_to_integer_grid_sparse(coords, initial_radius=10, expansion_factor=1.1, max_attempts=1000):
    """Use sparse occupancy when a dense grid exceeds memory or precision limits."""

    normalized_coords = coords
    distances = torch.sum((normalized_coords - torch.round(normalized_coords))**2, dim=1)

    sorted_indices = torch.argsort(distances)

    integer_coords = torch.zeros((len(coords), 2), dtype=torch.int64, device=coords.device)
    used_coords = set()

    for idx in sorted_indices:
        idx = idx.item()
        norm_coord = normalized_coords[idx]
        nearest_grid = (int(round(norm_coord[0].item())), int(round(norm_coord[1].item())))

        if nearest_grid not in used_coords:
            integer_coords[idx, 0] = nearest_grid[0]
            integer_coords[idx, 1] = nearest_grid[1]
            used_coords.add(nearest_grid)
        else:
            search_radius = initial_radius

            for attempt in range(max_attempts):
                candidates = []

                for radius in range(1, search_radius + 1):
                    for dx in range(-radius, radius + 1):
                        for dy in [-radius, radius]:
                            test_coord = (nearest_grid[0] + dx, nearest_grid[1] + dy)
                            if test_coord not in used_coords:
                                dist = (norm_coord[0].item() - test_coord[0])**2 + (norm_coord[1].item() - test_coord[1])**2
                                candidates.append((test_coord, dist))

                    for dy in range(-radius + 1, radius):
                        for dx in [-radius, radius]:
                            test_coord = (nearest_grid[0] + dx, nearest_grid[1] + dy)
                            if test_coord not in used_coords:
                                dist = (norm_coord[0].item() - test_coord[0])**2 + (norm_coord[1].item() - test_coord[1])**2
                                candidates.append((test_coord, dist))

                if candidates:
                    candidates.sort(key=lambda x: x[1])
                    best_coord = candidates[0][0]
                    integer_coords[idx, 0] = best_coord[0]
                    integer_coords[idx, 1] = best_coord[1]
                    used_coords.add(best_coord)
                    break
                else:
                    search_radius = int(search_radius * expansion_factor)

            if attempt == max_attempts - 1 and not candidates:
                min_x = int(torch.min(normalized_coords[:, 0]).item()) - 10
                max_x = int(torch.max(normalized_coords[:, 0]).item()) + 10
                min_y = int(torch.min(normalized_coords[:, 1]).item()) - 10
                max_y = int(torch.max(normalized_coords[:, 1]).item()) + 10

                found_coord = False
                for x in range(min_x, max_x + 1):
                    for y in range(min_y, max_y + 1):
                        if (x, y) not in used_coords:
                            integer_coords[idx, 0] = x
                            integer_coords[idx, 1] = y
                            used_coords.add((x, y))
                            found_coord = True
                            break
                    if found_coord:
                        break

                if not found_coord:
                    new_x, new_y = max_x + 1, min_y
                    integer_coords[idx, 0] = new_x
                    integer_coords[idx, 1] = new_y
                    used_coords.add((new_x, new_y))

    min_grid_x = torch.min(integer_coords[:, 0])
    min_grid_y = torch.min(integer_coords[:, 1])
    integer_coords[:, 0] -= min_grid_x
    integer_coords[:, 1] -= min_grid_y

    return integer_coords

def reshape_coords(sparse_coords):

    int_coords = sparse_coords.int()
    offsets = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], device=sparse_coords.device)
    expanded_coords = int_coords.unsqueeze(1) + offsets.unsqueeze(0)
    flat_coords = expanded_coords.view(-1, 2)
    unique_coords = torch.unique(flat_coords, dim=0, sorted=False)
    return unique_coords.float()

def nearest_neighbor_interpolation(sparse_coords, sparse_features, grid_coords, k, batch_size=4096):

    interpolated_features_list = []
    valid_grid_coords_list = []

    for i in range(0, len(grid_coords), batch_size):
        batch_grid_coords = grid_coords[i:i + batch_size]
        dists = torch.cdist(batch_grid_coords, sparse_coords)
        min_dists, indices = torch.min(dists, dim=1)
        mask = min_dists <= k
        valid_batch_grid_coords = batch_grid_coords[mask]

        if len(valid_batch_grid_coords) > 0:
            valid_grid_coords_list.append(valid_batch_grid_coords)
            interpolated_features_list.append(sparse_features[indices[mask]])

    if not valid_grid_coords_list:
        return torch.empty((0, sparse_features.size(1))), torch.empty((0, 2))

    interpolated_features = torch.cat(interpolated_features_list, dim=0)
    valid_grid_coords = torch.cat(valid_grid_coords_list, dim=0)

    return interpolated_features, valid_grid_coords
