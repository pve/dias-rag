"""Tests for CLI module."""

import subprocess
import tempfile
import os
from pathlib import Path

from click.testing import CliRunner

from src.cli import cli
from src.indexer import Document, index_documents


class TestCLI:
    """Test CLI commands."""

    def test_cli_help(self):
        """Test --help flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "Dias-RAG" in result.output
        assert "index" in result.output
        assert "search" in result.output

    def test_cli_version(self):
        """Test --version flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert "0.1.0" in result.output


class TestIndexCommand:
    """Test index command."""

    def test_index_command_help(self):
        """Test index --help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["index", "--help"])

        assert result.exit_code == 0
        assert "CONTENT_DIRECTORY" in result.output

    def test_index_command_success(self):
        """Test successful indexing."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            # Create test markdown files
            with open("test.md", "w") as f:
                f.write("# Test\n\nContent here.")

            result = runner.invoke(cli, ["index", "."])

            assert result.exit_code == 0
            assert "Indexing completed successfully" in result.output

    def test_index_command_invalid_directory(self):
        """Test indexing non-existent directory."""
        runner = CliRunner()
        result = runner.invoke(cli, ["index", "/nonexistent/path"])

        assert result.exit_code != 0
        # Click will catch the invalid path before our code

    def test_index_command_no_markdown(self):
        """Test indexing directory with no markdown files."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["index", "."])

            assert result.exit_code != 0
            assert "Error" in result.output

    def test_index_command_custom_data_dir(self):
        """Test using custom data directory."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            with open("test.md", "w") as f:
                f.write("# Test\n\nContent.")

            result = runner.invoke(cli, ["index", ".", "--data-dir", "./custom-data"])

            assert result.exit_code == 0
            assert "custom-data" in result.output


class TestSearchCommand:
    """Test search command."""

    def test_search_command_help(self):
        """Test search --help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["search", "--help"])

        assert result.exit_code == 0
        assert "QUERY" in result.output
        assert "--limit" in result.output

    def test_search_command_no_index(self):
        """Test search without index."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["search", "test query"])

            assert result.exit_code != 0
            assert "Error" in result.output

    def test_search_command_empty_query(self):
        """Test search with empty query."""
        runner = CliRunner()
        result = runner.invoke(cli, ["search", ""])

        assert result.exit_code != 0
        assert "Error" in result.output

    def test_search_command_success(self):
        """Test successful search."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            # Create and index a document
            with open("test.md", "w") as f:
                f.write("# Test\n\nThis is about authentication patterns.")

            # Index first
            result = runner.invoke(cli, ["index", "."])
            assert result.exit_code == 0

            # Then search
            result = runner.invoke(cli, ["search", "authentication"])

            assert result.exit_code == 0
            assert "Found" in result.output or "results" in result.output

    def test_search_command_with_limit(self):
        """Test search with custom limit."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            # Create test files
            for i in range(5):
                with open(f"test{i}.md", "w") as f:
                    f.write(f"# Test {i}\n\nDocument {i} content.")

            # Index
            runner.invoke(cli, ["index", "."])

            # Search with limit
            result = runner.invoke(cli, ["search", "document", "--limit", "2"])

            assert result.exit_code == 0

    def test_search_command_custom_data_dir(self):
        """Test search with custom data directory."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            with open("test.md", "w") as f:
                f.write("# Test\n\nContent.")

            # Index with custom data dir
            runner.invoke(cli, ["index", ".", "--data-dir", "./custom-data"])

            # Search with same data dir
            result = runner.invoke(cli, ["search", "test", "--data-dir", "./custom-data"])

            assert result.exit_code == 0


class TestCLIIntegration:
    """Integration tests that run the actual installed command.

    These tests use subprocess to run 'uv run dias-rag' which tests:
    - The package is correctly installed
    - The entry point script can import the src module
    - The command works as users would actually run it
    """

    def test_package_is_installed(self):
        """Test that the dias-rag package is installed (catches 'No module named src')."""
        result = subprocess.run(
            ["uv", "run", "python", "-c", "import src.cli; print('OK')"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, (
            f"Package not installed properly. stderr: {result.stderr}\n"
            f"If you see 'ModuleNotFoundError: No module named src', "
            f"try: rm -rf .venv && uv sync"
        )
        assert "OK" in result.stdout

    def test_installed_command_help(self):
        """Test that 'uv run dias-rag --help' works (catches packaging issues)."""
        result = subprocess.run(
            ["uv", "run", "dias-rag", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, (
            f"Command failed. stderr: {result.stderr}\n"
            f"stdout: {result.stdout}\n"
            f"If you see 'ModuleNotFoundError: No module named src', "
            f"try: rm -rf .venv && uv sync"
        )
        assert "Dias-RAG" in result.stdout
        assert "index" in result.stdout
        assert "search" in result.stdout

    def test_installed_command_version(self):
        """Test that 'uv run dias-rag --version' works."""
        result = subprocess.run(
            ["uv", "run", "dias-rag", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0
        assert "0.1.0" in result.stdout

    def test_installed_command_search_integration(self):
        """Test the full workflow: index then search using 'uv run dias-rag'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test markdown file
            content_file = Path(tmpdir) / "test.md"
            content_file.write_text("# Test\n\nThis is about authentication patterns.")

            data_dir = Path(tmpdir) / "data"

            # Index the content
            result = subprocess.run(
                ["uv", "run", "dias-rag", "index", tmpdir, "--data-dir", str(data_dir)],
                capture_output=True,
                text=True,
                timeout=60,
            )

            assert result.returncode == 0, f"Index failed: {result.stderr}"
            assert "Indexing completed successfully" in result.stdout

            # Search the indexed content
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "dias-rag",
                    "search",
                    "authentication patterns",
                    "--data-dir",
                    str(data_dir),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            assert result.returncode == 0, f"Search failed: {result.stderr}"
            assert "test.md" in result.stdout or "Test" in result.stdout
