"""Command-line interface for Dias-RAG."""

import sys
import time

import click

from src.indexer import rebuild_index
from src.search import format_results, semantic_search


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Dias-RAG: Local semantic search for markdown files.

    \b
    Examples:
        # Index markdown files
        $ dias-rag index /path/to/content

        # Search indexed documents
        $ dias-rag search "authentication patterns"

        # Limit number of results
        $ dias-rag search "security" --limit 5
    """
    pass


@cli.command()
@click.argument("content_directory", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option(
    "--data-dir",
    default="./data",
    help="Directory for vector database storage",
    type=click.Path(),
)
def index(content_directory: str, data_dir: str):
    """Index all markdown files in CONTENT_DIRECTORY.

    This command scans the specified directory for .md files, parses their content,
    generates vector embeddings, and stores them in a local ChromaDB database.

    \b
    Args:
        CONTENT_DIRECTORY: Path to directory containing markdown files

    \b
    Examples:
        $ dias-rag index /path/to/content
        $ dias-rag index ./docs --data-dir ./my-data
    """
    try:
        click.echo(f"Indexing markdown files from: {content_directory}")
        click.echo(f"Using data directory: {data_dir}")
        click.echo()

        # Rebuild the index
        rebuild_index(directory=content_directory, data_dir=data_dir)

        click.echo()
        click.secho("✓ Indexing completed successfully!", fg="green", bold=True)

    except ValueError as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)
    except Exception as e:
        click.secho(f"Unexpected error: {e}", fg="red", err=True)
        sys.exit(1)


@cli.command()
@click.argument("query")
@click.option(
    "--limit",
    "-n",
    default=10,
    help="Number of results to return",
    type=click.IntRange(1, 100),
)
@click.option(
    "--data-dir",
    default="./data",
    help="Directory containing vector database",
    type=click.Path(),
)
def search(query: str, limit: int, data_dir: str):
    """Search indexed documents using semantic similarity.

    Performs vector similarity search on previously indexed documents,
    returning the most relevant results based on semantic meaning.

    \b
    Args:
        QUERY: Search query text

    \b
    Examples:
        $ dias-rag search "authentication patterns"
        $ dias-rag search "API design" --limit 5
        $ dias-rag search "security" --data-dir ./my-data
    """
    try:
        # Validate query
        if not query.strip():
            click.secho("Error: Query cannot be empty", fg="red", err=True)
            sys.exit(1)

        # Perform search
        start_time = time.time()
        results = semantic_search(query, limit=limit, data_dir=data_dir)
        search_time = time.time() - start_time

        # Format and display results
        output = format_results(results, query=query, search_time=search_time)
        click.echo(output)

    except ValueError as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        click.echo()
        click.echo("Hint: Have you run 'dias-rag index' yet?")
        sys.exit(1)
    except Exception as e:
        click.secho(f"Unexpected error: {e}", fg="red", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
