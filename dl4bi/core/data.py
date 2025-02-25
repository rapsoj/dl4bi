from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class Batch(Mapping):
    def update(self, **kwargs):
        """Returns a new batch with updated attributes."""
        return replace(self, **kwargs)

    def __getitem__(self, key):
        return asdict(self)[key]

    def __iter__(self):
        """Allows you to use **batch to expand as kwargs."""
        return iter(asdict(self))

    def __len__(self):
        return len(asdict(self))
