import os
from dotenv import load_dotenv

load_dotenv()


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


class Settings:
    _show_plots: bool = False
    _database_dir: str = ""

    def __init__(self):
        self._show_plots = _to_bool(os.getenv("SHOW_PLOTS"), default=False)
        self._database_dir = os.environ["DATABASE_DIR"]

    @property
    def show_plots(self) -> bool:
        return self._show_plots

    @property
    def database_dir(self) -> str:
        return self._database_dir


settings = Settings()
