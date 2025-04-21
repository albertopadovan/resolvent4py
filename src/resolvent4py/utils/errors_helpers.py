from .. import functools


def raise_not_implemented_error(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        raise NotImplementedError(
            f"The linear operator '{self.__class__.__name__}' provides no "
            f"implementation for '{method.__name__}'"
        )

    return wrapper
