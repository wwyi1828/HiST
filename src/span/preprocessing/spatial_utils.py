import torch

def get_sorted_indices(pos):

    x = pos[:, 0]
    y = pos[:, 1]
    multiplier = y.max() + 1
    key = x * multiplier + y
    sorted_indices = torch.argsort(key)
    return sorted_indices

def map_to_integer_grid(coords, initial_radius=10, expansion_factor=1.1, max_attempts=1000):

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
