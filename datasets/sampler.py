import numpy as np
from torch.utils.data.sampler import Sampler
import copy


class StatefulSampler(Sampler):
    def __init__(self, data_source_length, use_random=True):
        self.use_random = use_random
        self.data_source_length = data_source_length
        self.continue_flag = False

    def __len__(self):
        return self.data_source_length

    def __iter__(self):
        if self.continue_flag == True:
            self.continue_flag = False
        else:
            if self.use_random:
                self.indices = list(np.random.permutation(self.data_source_length))
            else:
                self.indices = list(range(self.data_source_length))

        self.indices_record = copy.deepcopy(self.indices)

        for idx in self.indices:
            self.indices_record.pop(0)
            yield idx

    def load_state_dict(self, indices):
        self.indices = list(indices)
        self.continue_flag = True

    def state_dict(self):
        return np.array(self.indices_record)

