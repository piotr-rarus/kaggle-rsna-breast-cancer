from pathlib import Path

import matplotlib
import pandas as pd
from pandas_profiling import ProfileReport

from src.util.logs import get_logger

matplotlib.use("Agg")
logger = get_logger(__name__)


def profile(
    data: pd.DataFrame,
    name: str,
    dump_dir: Path,
    minimal: bool = False,
    explorative: bool = True,
    progress_bar: bool = False,
) -> None:
    dump_dir.mkdir(exist_ok=True, parents=True)

    logger.debug(f"Generating report: {name}")
    profile_report = ProfileReport(
        data,
        title=name,
        minimal=minimal,
        explorative=explorative,
        progress_bar=progress_bar,
    )
    output_path = dump_dir / f"{name}.html"
    profile_report.to_file(output_path)
    logger.info(f"Saved report as: {output_path}")
