import torch
import math
from .spatial_utils import map_to_integer_grid

class RandomTransform:

    def __init__(self, p=1.0):

        self.base_p = p
        self.p = p

    def set_multiplier(self, multiplier):
        self.p = self.base_p * multiplier

    def __call__(self, transform_matrix):
        raise NotImplementedError("Each transform must implement the __call__ method.")

class RandomDrop:
    def __init__(self, p=0.0):
        self.base_p = p
        self.p = p
        
    def set_multiplier(self, multiplier):
        self.p = self.base_p * multiplier

    def __call__(self, num_points, device):
        if self.p <= 0:
            return None
        mask = torch.rand(num_points, device=device) > self.p
        return mask

class RandomMirror(RandomTransform):

    def __call__(self, transform_matrix):

        if torch.rand(1).item() > self.p:
            return transform_matrix
        mirror_x = (torch.rand(1) > 0.5).item()
        mirror_y = (torch.rand(1) > 0.5).item()
        mirror_matrix = torch.tensor([
            [-1 if mirror_x else 1, 0],
            [0, -1 if mirror_y else 1]
        ], dtype=torch.float32, device=transform_matrix.device)
        transform_matrix = torch.mm(transform_matrix, mirror_matrix)
        return transform_matrix

class RandomShear(RandomTransform):

    def __init__(self, max_angle=15, p=1.0):

        super().__init__(p)
        self.shear_range = math.tan(math.radians(max_angle))

    def __call__(self, transform_matrix):

        if torch.rand(1) > self.p:
            return transform_matrix

        shear_factor_x = shear_factor_y = torch.tensor(0, device=transform_matrix.device)
        if torch.rand(1) > 0.5:
            shear_factor_x = (torch.rand(1, device=transform_matrix.device) - 0.5) * 2 * self.shear_range
        else:
            shear_factor_y = (torch.rand(1, device=transform_matrix.device) - 0.5) * 2 * self.shear_range

        shear_matrix = torch.tensor([
            [1, shear_factor_x],
            [shear_factor_y, 1]
        ], dtype=torch.float32, device=transform_matrix.device)

        transform_matrix = torch.mm(transform_matrix, shear_matrix)
        return transform_matrix

class RandomRotate(RandomTransform):

    def __init__(self, p=1, fixed_angle=False, max_angle=15):

        super().__init__(p)
        self.fixed_angle = fixed_angle
        self.max_angle = max_angle

        self.fixed_rotation_matrices = torch.tensor([
            [[1, 0],
             [0, 1]],

            [[0, -1],
             [1, 0]],

            [[-1, 0],
             [0, -1]],

            [[0, 1],
             [-1, 0]]
        ], dtype=torch.float32)

    def __call__(self, transform_matrix):

        if torch.rand(1).item() > self.p:
            return transform_matrix

        if self.fixed_angle:
            idx = torch.randint(0, 4, (1,)).item()
            rotation_matrix = self.fixed_rotation_matrices[idx].to(transform_matrix.device)
        else:
            base_angles = torch.tensor([0, 90, 180, 270], dtype=torch.float32, device=transform_matrix.device)
            base_angle = base_angles[torch.randint(0, 4, (1,))]
            deviation = (torch.rand(1, device=transform_matrix.device) * 2 - 1) * self.max_angle
            total_angle = base_angle + deviation
            theta = torch.deg2rad(total_angle)

            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)

            rotation_matrix = torch.tensor([
                [cos_theta, -sin_theta],
                [sin_theta, cos_theta]
            ], dtype=torch.float32, device=transform_matrix.device)

        transform_matrix = torch.mm(transform_matrix, rotation_matrix)

        return transform_matrix

class ComposeTransforms:

    def __init__(self, transforms):

        self.transforms = transforms

    def __call__(self, pos):

        pos = pos.float()
        center = torch.round(torch.mean(pos, dim=0))
        centered_pos = pos - center
        transform_matrix = torch.eye(2, device=pos.device)

        drop_mask = None

        for transform in self.transforms:
            if isinstance(transform, RandomDrop):
                mask = transform(len(pos), pos.device)
                if mask is not None:
                    if drop_mask is None:
                        drop_mask = mask
                    else:
                        drop_mask = drop_mask & mask
            else:
                transform_matrix = transform(transform_matrix)

        transformed_pos = torch.matmul(centered_pos, transform_matrix)
        
        if drop_mask is not None:
            transformed_pos = transformed_pos[drop_mask]

        min_pos = transformed_pos.min(dim=0, keepdim=True).values
        transformed_pos = transformed_pos - min_pos

        transformed_pos = map_to_integer_grid(transformed_pos)

        return transformed_pos, drop_mask
