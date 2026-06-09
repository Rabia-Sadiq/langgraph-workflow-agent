import logging
import colorlog


def get_logger(name: str = "langgraph-agent") -> logging.Logger:
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)s%(reset)s | %(message)s",
            log_colors={
                "DEBUG":    "cyan",
                "INFO":     "green",
                "WARNING":  "yellow",
                "ERROR":    "red",
                "CRITICAL": "bold_red",
            },
        )
    )

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = get_logger()


def print_trace(trace: list[str]) -> None:
    """
    Prints the full workflow trace after a run completes.
    """
    print("\n" + "=" * 60)
    print("  WORKFLOW TRACE")
    print("=" * 60)
    for step in trace:
        print(step)
    print("=" * 60 + "\n")