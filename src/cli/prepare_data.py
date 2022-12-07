import shutil
from pathlib import Path

import click

from src.preparation import prepare_data
from src.util import get_logger

logger = get_logger(__name__)


@click.command()
@click.option(
    "--data-dir",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=False,
        readable=True,
        path_type=Path,
    ),
    required=True,
)
@click.option(
    "--output-dir",
    type=click.Path(
        exists=False,
        file_okay=False,
        dir_okay=True,
        writable=True,
        readable=True,
        path_type=Path,
    ),
    required=True,
)
@click.option("--resolution", type=int)
@click.option("--n-jobs", type=int)
def main(data_dir: Path, output_dir: Path, resolution: int, n_jobs: int) -> None:
    logger.debug(f"Initializing output data structure: {output_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
        logger.debug("Found existing dir. Removing existing files.")
    output_dir.mkdir(exist_ok=True, parents=True)

    prepare_data(data_dir, output_dir, resolution, n_jobs)


if __name__ == "__main__":
    main()
