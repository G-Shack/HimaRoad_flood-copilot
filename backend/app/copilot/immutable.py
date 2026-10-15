"""Read-only JSON containers for server-owned, already verified snapshots."""

from typing import Any, NoReturn


def _immutable(self, *args: Any, **kwargs: Any) -> NoReturn:
    raise TypeError("verified snapshot is immutable")


class FrozenDict(dict[str, Any]):
    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable


class FrozenList(list[Any]):
    __setitem__ = __delitem__ = append = clear = extend = insert = pop = remove = _immutable
    reverse = sort = __iadd__ = __imul__ = _immutable


def freeze(value: Any) -> Any:
    if isinstance(value, (FrozenDict, FrozenList)):
        return value
    if isinstance(value, dict):
        return FrozenDict({key: freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return FrozenList(freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(freeze(item) for item in value)
    return value
