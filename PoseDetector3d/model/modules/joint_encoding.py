import torch


JOINT_GROUP1 = [[0], [1, 2, 3], [4, 5, 6], [7, 8, 9, 10], [11, 12, 13], [14, 15, 16]]
JOINT_GROUP2 = [[0], [1, 4, 7, 11, 14], [2, 5, 8, 12, 15], [3, 6, 9, 13, 16], [10]]


class PositionalEncoder:
    def __init__(self, temporal_dim, joint_dim, joint_groups1=None, joint_groups2=None):
        self.temporal_dim = temporal_dim
        self.joint_dim = joint_dim
        if joint_groups1:
            self.joint_groups1 = joint_groups1
        else:
            self.joint_groups1 = JOINT_GROUP1
        if joint_groups2:
            self.joint_groups2 = joint_groups2
        else:
            self.joint_groups2 = JOINT_GROUP2

    def sinusoidal_embedding(self, num_positions, dim):

        pe = torch.FloatTensor([[pos / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for pos in range(num_positions)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)

    def apply_temporal_encoding(self, input_tensor):

        batch_size, num_frames, num_joints, feature_dim = input_tensor.shape

        temporal_encoding = self.sinusoidal_embedding(num_frames, feature_dim).to(
            input_tensor.device)
        temporal_encoding = temporal_encoding.unsqueeze(2).repeat(batch_size, 1, num_joints,
                                                                  1)
        return input_tensor + temporal_encoding

    def apply_joint_encoding(self, input_tensor, joint_groups):

        batch_size, num_frames, num_joints, feature_dim = input_tensor.shape
        num_groups = len(joint_groups)

        joint_encoding = self.sinusoidal_embedding(num_groups, self.joint_dim).to(
            input_tensor.device)
        joint_encoded_tensor = input_tensor.clone()

        for group_idx, joint_indices in enumerate(joint_groups):
            for joint_idx in joint_indices:

                joint_encoded_tensor[:, :, joint_idx, :] += joint_encoding[:, group_idx, :].unsqueeze(1).repeat(
                    batch_size, num_frames, 1)

        return joint_encoded_tensor

    def apply_positional_encoding(self, input_tensor):

        temporal_encoded_tensor = self.apply_temporal_encoding(input_tensor)

        joint_encoded_tensor1 = self.apply_joint_encoding(temporal_encoded_tensor, self.joint_groups1)

        joint_encoded_tensor2 = self.apply_joint_encoding(temporal_encoded_tensor, self.joint_groups2)
        return temporal_encoded_tensor + joint_encoded_tensor1 + joint_encoded_tensor2


def find_duplicate_indices_per_batch(tensor):
    batch_size, num_frames, num_joints, feature_dim = tensor.shape
    batch_duplicate_indices = []

    for batch_idx in range(batch_size):
        batch_tensor = tensor[batch_idx]
        unique_values, counts = torch.unique(batch_tensor, return_counts=True)
        duplicate_values = unique_values[counts > 1]

        duplicate_indices = {}
        for value in duplicate_values:
            indices = (batch_tensor == value).nonzero(as_tuple=False)
            duplicate_indices[value.item()] = indices

        batch_duplicate_indices.append(duplicate_indices)

    return batch_duplicate_indices
