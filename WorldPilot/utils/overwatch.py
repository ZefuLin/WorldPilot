import logging
from contextlib import nullcontext
from logging import LoggerAdapter
from typing import Any, Callable, ClassVar, Dict, MutableMapping, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


class ContextAdapter(LoggerAdapter):
    CTX_PREFIXES: ClassVar[Dict[int, str]] = {
        0: "",
        1: "  ",
        2: "    ",
        3: "      ",
    }

    def process(self, msg: str, kwargs: MutableMapping[str, Any]) -> Tuple[str, MutableMapping[str, Any]]:
        ctx_level = kwargs.pop("ctx_level", 0)
        return f"{self.CTX_PREFIXES.get(ctx_level, '')}{msg}", kwargs


class PureOverwatch:
    def __init__(self, name: str) -> None:
        self.logger = ContextAdapter(logging.getLogger(name), extra={})
        self.debug = self.logger.debug
        self.info = self.logger.info
        self.warning = self.logger.warning
        self.error = self.logger.error
        self.critical = self.logger.critical

    def log(self, msg: str, *args, **kwargs) -> None:
        self.info(msg, *args, **kwargs)

    @staticmethod
    def _identity(fn: Callable[..., Any]) -> Callable[..., Any]:
        return fn

    @property
    def rank_zero_only(self) -> Callable[..., Any]:
        return self._identity

    @property
    def local_zero_only(self) -> Callable[..., Any]:
        return self._identity

    @property
    def rank_zero_first(self):
        return nullcontext

    @property
    def local_zero_first(self):
        return nullcontext

    @staticmethod
    def is_rank_zero() -> bool:
        return True

    @staticmethod
    def rank() -> int:
        return 0

    @staticmethod
    def world_size() -> int:
        return 1


def initialize_overwatch(name: str) -> PureOverwatch:
    return PureOverwatch(name)
