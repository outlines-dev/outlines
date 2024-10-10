import asyncio
import contextlib
import functools
import os
from typing import Callable, Optional

import cloudpickle
from diskcache import Cache, Disk
from diskcache.core import ENOVAL, UNKNOWN, args_to_key, full_name

_caching_enabled = True


class CloudpickleDisk(Disk):
    def __init__(self, directory, compress_level=1, **kwargs):
        self.compress_level = compress_level
        super().__init__(directory, **kwargs)

    def put(self, key):
        data = cloudpickle.dumps(key)
        return super().put(data)

    def get(self, key, raw):
        data = super().get(key, raw)
        return cloudpickle.loads(data)

    def store(self, value, read, key=UNKNOWN):
        if not read:
            value = cloudpickle.dumps(value)
        return super().store(value, read, key=key)

    def fetch(self, mode, filename, value, read):
        data = super().fetch(mode, filename, value, read)
        if not read:
            data = cloudpickle.loads(data)
        return data


@functools.lru_cache(1)
def get_cache():
    """Get the context object that contains previously-computed return values.

    The cache is used to avoid unnecessary computations and API calls, which can
    be long and expensive for large models.

    The cache directory defaults to `HOMEDIR/.cache/outlines`, but this choice
    can be overridden by the user by setting the value of the `OUTLINES_CACHE_DIR`
    environment variable.

    """
    from outlines._version import __version__ as outlines_version  # type: ignore

    home_dir = os.path.expanduser("~")
    cache_dir = os.environ.get("OUTLINES_CACHE_DIR", f"{home_dir}/.cache/outlines")
    memory = Cache(
        cache_dir,
        eviction_policy="none",
        cull_limit=0,
        disk=CloudpickleDisk,
    )

    # ensure if version upgrade occurs, old cache is pruned
    if outlines_version != memory.get("__version__"):
        memory.clear()
    memory["__version__"] = outlines_version

    return memory


def cache(expire: Optional[float] = None, typed=False, ignore=()):
    """Caching decorator for memoizing function calls.

    The cache key is created based on the values returned by the key_function callable
    if provided or based on the arguments of the decorated function directly otherwise

    This is based on `diskcache`'s `memoize`.

    Parameters
    ----------
    expire
        Seconds until arguments expire.
    typed
        Cache different types separately.
    ignore
        Positional or keyword arguments to ignore.

    Returns
    -------
        A decorator function that can be applied to other functions.
    """

    def decorator(cached_function: Callable):
        memory = get_cache()

        base = (full_name(cached_function),)

        if asyncio.iscoroutinefunction(cached_function):

            async def wrapper(*args, **kwargs):
                if not _caching_enabled:
                    return await cached_function(*args, **kwargs)

                cache_key = wrapper.__cache_key__(*args, **kwargs)
                result = wrapper.__memory__.get(cache_key, default=ENOVAL, retry=True)

                if result is ENOVAL:
                    result = await cached_function(*args, **kwargs)
                    wrapper.__memory__.set(cache_key, result, expire, retry=True)

                return result

        else:

            def wrapper(*args, **kwargs):
                if not _caching_enabled:
                    return cached_function(*args, **kwargs)

                cache_key = wrapper.__cache_key__(*args, **kwargs)
                result = wrapper.__memory__.get(cache_key, default=ENOVAL, retry=True)

                if result is ENOVAL:
                    result = cached_function(*args, **kwargs)
                    wrapper.__memory__.set(cache_key, result, expire, retry=True)

                return result

        def __cache_key__(*args, **kwargs):
            """Make key for cache given function arguments."""
            return str(args_to_key(base, args, kwargs, typed, ignore))

        wrapper.__cache_key__ = __cache_key__  # type: ignore
        wrapper.__memory__ = memory  # type: ignore
        wrapper.__wrapped__ = cached_function  # type: ignore

        return wrapper

    return decorator


def disable_cache():
    """Disable the cache for this session.

    Generative models output different results each time they are called when
    sampling. This can be a desirable property for some workflows, in which case
    one can call `outlines.call.disable` to disable the cache for the session.

    This function does not delete the cache, call `outlines.cache.clear`
    instead. It also does not overwrite the cache with the values returned
    during the session.

    Example
    -------

    `outlines.cache.disable` should be called right after importing outlines:

    >>> import outlines.caching as cache
    >>> cache.disable_cache()

    """
    global _caching_enabled
    _caching_enabled = False


def clear_cache():
    """Erase the cache completely."""
    memory = get_cache()
    memory.clear()


@contextlib.contextmanager
def cache_disabled():
    # outlines.caching._caching_enabled
    global _caching_enabled
    original_state = _caching_enabled
    _caching_enabled = False
    try:
        yield
    finally:
        _caching_enabled = original_state
