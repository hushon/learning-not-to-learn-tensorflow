import numpy as np

def ichunks(x, n, drop_remainder=False):
    '''Creates n-sized chunks from a list-like object. Used to make minibatch of training dataset.
    Args:
        x (list, tuple, or any iterable type): input list-like object
        n (int): size of each chunk
        drop_remainder (bool): if True, drop the final chunk if its size is smaller then N
    Returns: 
        A generator that yields each chunk
    Example:
        >>> items = ichunks([1, 2, 3, 4, 5, 6], 2)
        >>> for x in items: print(x)
            [1, 2]
            [3, 4]
            [5, 6]
    '''
    assert n > 0

    x = iter(x)
    while True:
        try:
            chunk = []
            for i in range(n):
                chunk.append(next(x))
        except StopIteration:
            if drop_remainder:
                break
            else:
                if len(chunk)>0:
                    yield chunk
                    break
                else:
                    break
        else:
            yield chunk

class Dataloader:
    def __init__(self):
        self.__dataset = None
        self.__index = None

    def __call__(self):
        pass

    def from_dict(self, dataset: dict):
        # check if size of dataset matches
        size = 0
        for key, value in dataset.items():
            if size:
                assert size == len(value)
            else:
                size = len(value)

        self.__dataset = dataset
        self.__index = list(range(size))
        return self

    def shuffle(self, seed: int):
        np.random.seed(seed)
        np.random.shuffle(self.__index)
        return self

    def iter(self, batch_size: int, drop_remainder: bool):
        for batch_index in ichunks(self.__index, batch_size, drop_remainder):
            yield {k: v[batch_index] for k, v in self.__dataset.items()}

    def head(self, batch_size: int):
        batch_index = next(ichunks(self.__index, batch_size, False))
        return {k: v[batch_index] for k, v in self.__dataset.items()}