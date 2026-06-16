from torch.utils.data import DataLoader
from torch.utils.data import _utils
from typing import Generic, TypeVar


_T_co = TypeVar("_T_co", covariant=True)

class TxtDataLoader(Generic[_T_co], DataLoader):

    def _next_data(self):
        index = self._next_index()  # may raise StopIteration
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        if self._pin_memory:
            data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)
        return data