import torch

# Example tensors
alpha = torch.tensor([32, 43, 76])  # [N]
alpha_domain = torch.tensor([0, 15, 30, 45, 60, 75, 90])  # [M]


# Step 1: Compute the absolute differences
diffs = torch.abs(alpha.unsqueeze(1) - alpha_domain.unsqueeze(0))  # [N, M]

# Step 2: Find the index of the minimum difference for each value in alpha
indices = torch.argmin(diffs, dim=1)  # [N]

print(indices)  # Output: tensor([2, 3, 5])
