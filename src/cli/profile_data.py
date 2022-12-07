from pathlib import Path

import click
import pandas as pd

from src.lib.eda import profile


@click.command()
@click.option(
    "--data-path",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
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
def main(data_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True, parents=True)
    data = pd.read_csv(data_path)
    profile(data=data, name=data_path.stem, dump_dir=output_dir, explorative=True)


if __name__ == "__main__":
    main()
