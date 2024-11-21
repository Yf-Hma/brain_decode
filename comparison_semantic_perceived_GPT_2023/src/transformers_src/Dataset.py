import torch
from torch.utils.data import Dataset
import random


class MyDataset(Dataset):
    def __init__(self, tensor_list, token_ids_list, sos_token_id, eos_token_id, max_size):
        self.tensor_list = tensor_list
        self.token_ids_list = token_ids_list
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        self.max_size = max_size
        self._shuffle_data()

    def _shuffle_data(self):
        num_groups = len(self.tensor_list) // 5
        tensor_groups = []
        token_groups = []
        for j in range(num_groups):
            i = j * 5
            tensor_group = self.tensor_list[i:i + 5]
            tensor_groups.append(tensor_group)
            token_group = self.token_ids_list[i:i + 5]
            token_groups.append(token_group)
        grouped_data = list(zip(tensor_groups, token_groups))
        random.seed(42)
        random.shuffle(grouped_data)
        t_list = []
        id_list = []
        i = 0
        r = 0
        for tensor_group, token_group in grouped_data:
            for t in tensor_group:
                t_list.append(t)
            for to in token_group:
                id_list.append(to)
        self.tensor_list, self.token_ids_list = t_list, id_list

    def __len__(self):
        return len(self.tensor_list)

    def __getitem__(self, index):
        tensors = self.tensor_list[index]
        token_ids = self.token_ids_list[index]
        
#         sos_token = torch.full((1, 274), self.sos_token_id)
#         eos_token = torch.full((1, 274), self.eos_token_id)
#         src = []
#         for t in tensors:
#             src_l = torch.concat([sos_token, t, eos_token], dim=0)
#             src.append(src_l)
        src = torch.stack(tensors)
        padded_tensors = []
        for tensor in token_ids:
            #tensor = tensor[1:]
            if tensor.size(-1) < self.max_size:
                # Pad the tensor with zeros to match the size of the largest tensor
                padded_tensor = torch.nn.functional.pad(tensor, (0, self.max_size - tensor.size(-1)), mode='constant',
                                                        value=0)
            else:
                # Resize the tensor to match the size of the largest tensor
                padded_tensor = tensor[:self.max_size]
            padded_tensors.append(padded_tensor)
        trg = torch.stack(padded_tensors)
        return src, trg
    def slice(self, start, end):
        sliced_tensor_list = self.tensor_list[start:end]
        sliced_token_ids_list = self.token_ids_list[start:end]

        # Create a new MyDataset object with the sliced data and return it
        return MyDataset(sliced_tensor_list, sliced_token_ids_list, self.sos_token_id, self.eos_token_id, self.max_size)