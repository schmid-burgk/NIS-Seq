from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Optional

import pandas as pd
import typer
import yaml

from nisseq.config import ROOT_DIR


def load_config(config_file: Optional[Path] = None) -> dict[str, Any]:
    if config_file is None:
        config_file = ROOT_DIR / "config.yaml"

    with config_file.open("r") as f:
        config = yaml.safe_load(f)
        config["config_file"] = config_file
        return config


def wrap_command_with_config(
    function: Callable[[dict[str, Any]], None],
) -> Callable[[Annotated[Optional[Path], typer.Option]], None]:
    def wrapper(
        config_file: Annotated[Optional[Path], typer.Option(help="Path to the yaml configuration file.")] = None,
    ) -> None:
        config = load_config(config_file)
        function(config)

    return wrapper


def load_csv_or_parquet(file: Path) -> pd.DataFrame:
    if file.suffix == ".csv":
        return pd.read_csv(file)
    if file.suffix == ".parquet":
        return pd.read_parquet(file)
    raise ValueError(f"Unsupported file format: {file.suffix}")


def save_csv_or_parquet(data: pd.DataFrame, file: Path, index: bool) -> None:
    if file.suffix == ".csv":
        data.to_csv(file, index=index)
    elif file.suffix == ".parquet":
        data.to_parquet(file, index=index)
    else:
        raise ValueError(f"Unsupported file format: {file.suffix}")
