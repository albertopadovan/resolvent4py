__all__ = ["raise_not_implemented_error"]

import functools


def raise_not_implemented_error(method):
    r"""
    Decorator that turns a method body into a ``NotImplementedError`` raiser.

    Used on the opt-in methods of :class:`.LinearOperator`
    (:meth:`.LinearOperator.apply_hermitian_transpose`,
    :meth:`.LinearOperator.solve`, ...) so that subclasses which do not
    override them fail loudly at call time with a message naming both the
    concrete class and the missing method.

    :param method: the wrapped method (typically an empty method body on
        the base class)

    :return: a wrapper that raises ``NotImplementedError`` when invoked
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        raise NotImplementedError(
            f"The linear operator '{self.__class__.__name__}' provides no "
            f"implementation for '{method.__name__}'"
        )

    return wrapper
