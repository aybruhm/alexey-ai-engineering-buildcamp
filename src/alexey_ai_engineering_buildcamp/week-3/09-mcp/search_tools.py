from typing import Any

from minsearch import AppendableIndex


class SearchTools:
    def __init__(self, index: AppendableIndex, file_index: dict[str, str]):
        self.index = index
        self.file_index = file_index

    def search(self, query: str) -> list[dict[str, Any]]:
        """
        Search the documentation for entries matching the given query.

        Args:
            query (str): The search query to look up in the index.

        Returns:
            list[dict[str, Any]]: A list of search result entries, each containing relevant metadata.
        """
        boost = {"question": 3.0, "section": 0.5}
        results = self.index.search(
            query=query,
            boost_dict=boost,
            num_results=5,
        )

        return results

    def add_entry(self, question: str, answer: str) -> None:
        """
        Add a new entry to the documentation database.

        Args:
            question (str): The question to be added to the documentation database.
            answer (str): The corresponding answer to the question.
        """
        doc = {
            "question": question,
            "text": answer,
            "section": "user added",
            "course": "data-engineering-zoomcamp",
        }
        self.index.append(doc)

    def get_file(self, filename: str) -> str:
        """
        Retrieve a file's contents by filename.

        Args:
            filename (str): The filename of the file to retrieve.

        Returns:
            str: The file contents if found, otherwise an error message.
        """

        if filename in self.file_index:
            return self.file_index[filename]
        return f"file {filename} does not exist"
