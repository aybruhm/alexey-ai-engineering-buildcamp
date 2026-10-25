from pathlib import Path

import duckdb

DB_PATH = Path(__file__).resolve().parent / "db" / "taxi.duckdb"


class SQLTools:
    def __init__(
        self,
        max_rows: int = 50,
        db_path: str | Path = DB_PATH,
    ) -> None:
        """
        Initialize the SQLTools instance.

        Args:
            max_rows (int): Maximum number of rows returned by `run_sql`.
            db_path (str | Path): Path to the DuckDB database containing the `trips` table.
        """
        self.max_rows = max_rows
        self.connection = duckdb.connect(str(db_path))

    def get_schema(self) -> str:
        """
        Return the column names and types of the `trips` table.

        Returns:
            str: One line per column, formatted as "<column>: <type>".
        """
        try:
            rows = self.connection.execute("DESCRIBE trips").fetchall()
        except duckdb.Error as exc:
            return f"SQL error: {exc}"

        return "\n".join(f"{name}: {column_type}" for name, column_type, *_ in rows)

    def run_sql(self, query: str) -> str:
        """
        Execute a SQL query and return the results as text.

        Args:
            query (str): The SQL query to execute.

        Returns:
            str: Column headers followed by up to `max_rows` data rows, or an error message.
        """
        try:
            result = self.connection.execute(query)
            if result.description is None:
                return "Query executed successfully (no rows returned)."

            columns = [description[0] for description in result.description]
            rows = result.fetchmany(self.max_rows + 1)
        except duckdb.Error as exc:
            return f"SQL error: {exc}"

        truncated = len(rows) > self.max_rows
        rows = rows[: self.max_rows]

        lines = [" | ".join(columns)]
        lines.extend(
            " | ".join("NULL" if value is None else str(value) for value in row)
            for row in rows
        )

        if not rows:
            lines.append("(no rows)")
        elif truncated:
            lines.append(f"(showing first {self.max_rows} rows)")

        return "\n".join(lines)
