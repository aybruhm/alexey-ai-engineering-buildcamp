import io
import sys
import zipfile
from pathlib import Path
from urllib.parse import quote

import httpx
from tqdm.auto import tqdm


def _log(message: str) -> None:
    """Log to stderr.

    This module runs inside an MCP stdio server, where stdout carries the
    JSON-RPC protocol. Diagnostics must go to stderr so they don't corrupt it.
    """
    print(message, file=sys.stderr)


def get_or_download_documentation(
    repo_owner: str,
    repo_name: str,
    branch_name: str,
    docs_path: str = "docs",
    save_dir: str | None = None,
) -> bytes:
    """
    Return the Markdown documentation as a zip archive, reusing a local copy if one already exists, and downloading it otherwise.

    Args:
        repo_owner (str): The owner of the GitHub repository.
        repo_name (str): The name of the GitHub repository.
        branch_name (str): The branch to download from.
        docs_path (str): Path of the documentation folder inside the repository.
        save_dir (str | None): Local directory to reuse/save the files in. If
            None, the files are only returned as an in-memory zip.

    Returns:
        bytes: A zip archive containing only the Markdown files under `docs_path`.
    """

    def _get(doc_dir: str | None = None) -> bytes | None:
        """Build a zip from an already-downloaded local directory.

        Args:
            doc_dir (str | None): Directory that may already contain the files.

        Returns:
            bytes | None: A zip archive of the local files, or None if the
            directory is missing or empty.
        """
        if doc_dir is None:
            return None

        out_dir = Path(doc_dir)
        existing_files = sorted(out_dir.rglob("*")) if out_dir.exists() else []
        existing_files = [f for f in existing_files if f.is_file()]
        if not existing_files:
            return None

        _log(
            f"Using already-downloaded documentation in '{doc_dir}' ({len(existing_files)} files)"
        )

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for file in existing_files:
                archive.write(file, file.relative_to(out_dir).as_posix())
        return buffer.getvalue()

    def _download() -> bytes:
        """Download only the Markdown documentation from GitHub.

        Returns:
            bytes: A zip archive containing only the Markdown files under
            `docs_path`.
        """
        api = "https://api.github.com"
        raw = "https://raw.githubusercontent.com"

        with httpx.Client(follow_redirects=True) as client:
            # Resolve the branch to its root tree SHA
            branch = client.get(
                f"{api}/repos/{repo_owner}/{repo_name}/branches/{branch_name}"
            )
            branch.raise_for_status()
            root_tree_sha = branch.json()["commit"]["commit"]["tree"]["sha"]
            _log(f"Retrieved root tree SHA: {root_tree_sha}")

            # Locate the docs subtree inside the root tree
            root_tree = client.get(
                f"{api}/repos/{repo_owner}/{repo_name}/git/trees/{root_tree_sha}"
            )
            root_tree.raise_for_status()
            docs_entry = next(
                (
                    item
                    for item in root_tree.json()["tree"]
                    if item["path"] == docs_path
                ),
                None,
            )
            if docs_entry is None:
                raise ValueError(
                    f"Path '{docs_path}' not found in {repo_owner}/{repo_name}@{branch_name}"
                )
            _log(f"Retrieved docs subtree '{docs_path}' (SHA: {docs_entry['sha']})")

            # Recursively list every file under docs/
            docs_tree = client.get(
                f"{api}/repos/{repo_owner}/{repo_name}/git/trees/{docs_entry['sha']}?recursive=1"
            )
            docs_tree.raise_for_status()
            _log(f"Retrieved recursive tree for '{docs_path}'")

            # Collect the Markdown files to download
            markdown_files = [
                item["path"]
                for item in docs_tree.json().get("tree", [])
                if item.get("type") == "blob" and item["path"].endswith((".md", ".mdx"))
            ]
            _log(f"Found {len(markdown_files)} Markdown files to download")

            # Download each Markdown file and package it into an in-memory zip
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as archive:
                for path in tqdm(markdown_files, desc="Downloading docs", unit="file"):
                    file_url = f"{raw}/{repo_owner}/{repo_name}/{branch_name}/{quote(f'{docs_path}/{path}')}"
                    response = client.get(file_url)
                    response.raise_for_status()
                    archive.writestr(path, response.content)

        return buffer.getvalue()

    # Reuse the local copy if it already exists
    data = _get(save_dir)
    if data is not None:
        return data

    # Otherwise download and (optionally) save to disk
    data = _download()

    if save_dir is not None:
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            num_files = len(zf.namelist())
            zf.extractall(out_dir)
        _log(f"Extracted {num_files} files to '{save_dir}'")

    return data


def retrieve_documentation(
    repo_owner: str,
    repo_name: str,
    branch_name: str,
    docs_path: str = "docs",
    save_dir: str | None = None,
):
    response_contents = get_or_download_documentation(
        repo_owner=repo_owner,
        repo_name=repo_name,
        branch_name=branch_name,
        docs_path=docs_path,
        save_dir=save_dir,
    )
    documents = []
    with zipfile.ZipFile(io.BytesIO(response_contents)) as zip_ref:
        for file_path in zip_ref.namelist():
            with zip_ref.open(file_path) as file:
                content = file.read().decode("utf-8")
                doc = {"content": content, "filename": file_path}
                documents.append(doc)
    return documents
