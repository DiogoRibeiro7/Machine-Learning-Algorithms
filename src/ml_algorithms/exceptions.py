"""Package-specific exceptions."""


class NotFittedError(RuntimeError):
    """Raised when an estimator is used before fitting."""


__all__ = ["NotFittedError"]
