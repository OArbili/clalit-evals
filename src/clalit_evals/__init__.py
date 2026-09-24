"""clalit-evals: a minimal package with a single hello-world function."""

__all__ = ["hello_world"]


def hello_world() -> None:
    """Print ``Hello, World!`` to standard output."""
    print("Hello, World!")
