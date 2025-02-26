from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace

import jax


@dataclass(frozen=True)
class ElementSelectorMixin(Mapping):
    def element(self, i: int):
        d = {}
        for k, v in self.items():
            d[k] = v
            if isinstance(v, jax.Array):
                d[k] = v[i]
        return d


@dataclass(frozen=True)
class Data(Mapping):
    def update(self, **kwargs):
        """Returns a new batch with updated attributes."""
        return replace(self, **kwargs)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return all(
            getattr(self, f.name) == getattr(other, f.name) for f in fields(self)
        )

    def __getitem__(self, key):
        return asdict(self)[key]

    def __iter__(self):
        """Allows you to use **batch to expand as kwargs."""
        return iter(asdict(self))

    def __len__(self):
        return len(asdict(self))


@dataclass(frozen=True)
class Batch(Mapping):
    def update(self, **kwargs):
        """Returns a new batch with updated attributes."""
        return replace(self, **kwargs)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return all(
            getattr(self, f.name) == getattr(other, f.name) for f in fields(self)
        )

    def __getitem__(self, key):
        return asdict(self)[key]

    def __iter__(self):
        """Allows you to use **batch to expand as kwargs."""
        return iter(asdict(self))

    def __len__(self):
        return len(asdict(self))
