from torch.utils.data import DataLoader


class FixationDataLoader(DataLoader):
    def __init__(self,
                 data: any,
                 batch_size: int,
                 sampler=None):
        super().__init__(data, batch_size=batch_size, num_workers=2, pin_memory=True, sampler=sampler)
