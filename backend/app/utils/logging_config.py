from pathlib import Path


def configure_logging() -> None:
    """Configure a small, useful default log format for the API process."""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


PROJECT_ROOT = Path(__file__).resolve().parents[3]
