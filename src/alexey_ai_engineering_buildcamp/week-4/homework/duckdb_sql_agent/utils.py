import sys
from pathlib import Path

import duckdb
import httpx
from tqdm.auto import tqdm


def _log(message: str) -> None:
    """
    Log to stderr.
    """
    print(message, file=sys.stderr)


PARQUET_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
)
DB_DIR = Path(__file__).resolve().parent / "db"
PARQUET_PATH = DB_DIR / "yellow_tripdata_2024-01.parquet"
DB_PATH = DB_DIR / "taxi.duckdb"


def download_parquet(
    url: str = PARQUET_URL,
    save_path: str | Path = PARQUET_PATH,
) -> str:
    """Download a parquet file from a URL to a local file.

    Args:
        url (str): The URL of the parquet file to download.
        save_path (str | Path): The local path to save the file to.

    Returns:
        str: The path of the downloaded file.
    """

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.exists():
        _log(f"Parquet file already exists at '{save_path}', skipping download")
        return str(save_path)

    _log(f"Downloading parquet file from '{url}' to '{save_path}'")
    with httpx.stream("GET", url, follow_redirects=True) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with (
            open(save_path, "wb") as f,
            tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc="Downloading parquet",
            ) as progress,
        ):
            for chunk in response.iter_bytes():
                f.write(chunk)
                progress.update(len(chunk))

    _log(f"Downloaded parquet file to '{save_path}'")
    return str(save_path)


def create_database(db_path: str | Path = DB_PATH) -> str:
    """Create a duckdb database named taxi.

    Args:
        db_path (str | Path): The path to the duckdb database file.

    Returns:
        str: The path of the created database.
    """

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists():
        _log(f"Duckdb database already exists at '{db_path}', skipping creation")
        return str(db_path)

    _log(f"Creating duckdb database '{db_path}'")
    con = duckdb.connect(str(db_path))
    con.close()

    _log(f"Created duckdb database '{db_path}'")
    return str(db_path)


def setup_database(
    parquet_path: str | Path = PARQUET_PATH,
    db_path: str | Path = DB_PATH,
) -> int:
    """Connect to the parquet file and test it by counting the number of trips.

    Args:
        parquet_path (str | Path): The path to the downloaded parquet file.
        db_path (str | Path): The path to the duckdb database.

    Returns:
        int: The number of trips available in the parquet file.
    """

    parquet_path = Path(parquet_path)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    _log(f"Connecting to database '{db_path}' and parquet file '{parquet_path}'")
    con = duckdb.connect(str(db_path))

    con.execute(f"""
        CREATE TABLE IF NOT EXISTS trips AS
        SELECT * FROM '{parquet_path}'
    """)
    stmt = con.execute(f"SELECT count(*) FROM read_parquet('{parquet_path}')")
    row = stmt.fetchone()
    trip_count = row[0] if row else 0
    con.close()

    _log(f"Number of trips available: {trip_count}")
    return trip_count


def init_database() -> str:
    """Ensure the parquet file and database are ready, then return the db path."""

    download_parquet()
    setup_database()
    return str(DB_PATH)
