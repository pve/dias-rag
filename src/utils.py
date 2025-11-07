"""Utility functions for markdown parsing and content chunking."""

import re
from typing import Dict, List, Tuple

import frontmatter


def parse_frontmatter(content: str) -> Tuple[Dict, str]:
    """
    Extract YAML frontmatter and content from markdown.

    Args:
        content: Raw markdown content with optional YAML frontmatter

    Returns:
        Tuple of (frontmatter_dict, markdown_content)
        - frontmatter_dict: Parsed YAML as dictionary (empty dict if missing/invalid)
        - markdown_content: Markdown content without frontmatter
    """
    try:
        # Use python-frontmatter to parse
        post = frontmatter.loads(content)
        return (dict(post.metadata), post.content)
    except Exception:
        # If parsing fails, return empty dict and original content
        return ({}, content)


def chunk_content(content: str, max_chunk_size: int = 512) -> List[str]:
    """
    Split content into semantic chunks using sentence boundaries.

    Algorithm:
    1. Split on paragraph breaks (double newlines)
    2. Within paragraphs, split on sentence boundaries (., !, ?)
    3. Accumulate sentences until max_chunk_size would be exceeded
    4. If single sentence > max_chunk_size, split at clause boundaries (; , :)
    5. If still too large, split at word boundaries

    Args:
        content: Text content to chunk
        max_chunk_size: Maximum characters per chunk (default: 512)

    Returns:
        List of text chunks, each ≤ max_chunk_size characters
        No overlap between chunks in MVP
    """
    if not content or not content.strip():
        return []

    chunks = []

    # Step 1: Split on paragraph breaks (double newlines)
    paragraphs = re.split(r'\n\s*\n', content)

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # If paragraph fits in chunk size, add it directly
        if len(paragraph) <= max_chunk_size:
            chunks.append(paragraph)
            continue

        # Step 2: Split paragraph into sentences
        sentences = _split_sentences(paragraph)

        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Step 3: If sentence is too long, split it further
            if len(sentence) > max_chunk_size:
                # First, add any accumulated sentences as a chunk
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0

                # Step 4: Split long sentence at clause boundaries
                sub_chunks = _split_at_clauses(sentence, max_chunk_size)
                chunks.extend(sub_chunks)
            else:
                # Check if adding this sentence would exceed limit
                sentence_length = len(sentence)
                # Account for space between sentences
                needed_length = sentence_length + (1 if current_chunk else 0)

                if current_length + needed_length > max_chunk_size:
                    # Save current chunk and start new one
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    # Add to current chunk
                    current_chunk.append(sentence)
                    current_length += needed_length

        # Add any remaining sentences
        if current_chunk:
            chunks.append(' '.join(current_chunk))

    return chunks


def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using common sentence boundaries.

    Args:
        text: Text to split

    Returns:
        List of sentences
    """
    # Split on .!? followed by whitespace and capital letter or end of string
    # This is a simple heuristic and won't be perfect for all cases
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$'
    sentences = re.split(sentence_pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def _split_at_clauses(text: str, max_size: int) -> List[str]:
    """
    Split text at clause boundaries (semicolon, comma, colon).

    Args:
        text: Text to split
        max_size: Maximum chunk size

    Returns:
        List of text chunks
    """
    chunks = []

    # Try splitting at clause boundaries: ; : ,
    # Priority: semicolon > colon > comma
    for delimiter in [';', ':', ',']:
        if delimiter in text:
            parts = text.split(delimiter)
            current = []
            current_length = 0

            for i, part in enumerate(parts):
                part = part.strip()
                if not part:
                    continue

                # Add delimiter back except for last part
                if i < len(parts) - 1:
                    part = part + delimiter

                part_length = len(part)

                # If single part is still too large, need word-level split
                if part_length > max_size:
                    if current:
                        chunks.append(' '.join(current))
                        current = []
                        current_length = 0
                    # Step 5: Split at word boundaries
                    word_chunks = _split_at_words(part, max_size)
                    chunks.extend(word_chunks)
                    continue

                needed_length = part_length + (1 if current else 0)

                if current_length + needed_length > max_size:
                    if current:
                        chunks.append(' '.join(current))
                    current = [part]
                    current_length = part_length
                else:
                    current.append(part)
                    current_length += needed_length

            if current:
                chunks.append(' '.join(current))

            return chunks

    # If no delimiters found, fall back to word splitting
    return _split_at_words(text, max_size)


def _split_at_words(text: str, max_size: int) -> List[str]:
    """
    Split text at word boundaries as last resort.

    Args:
        text: Text to split
        max_size: Maximum chunk size

    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    current = []
    current_length = 0

    for word in words:
        word_length = len(word)
        # Account for space between words
        needed_length = word_length + (1 if current else 0)

        if word_length > max_size:
            # Single word is too large, split it
            if current:
                chunks.append(' '.join(current))
                current = []
                current_length = 0

            # Split the word itself
            for i in range(0, len(word), max_size):
                chunks.append(word[i:i + max_size])
        elif current_length + needed_length > max_size:
            # Adding this word would exceed limit
            if current:
                chunks.append(' '.join(current))
            current = [word]
            current_length = word_length
        else:
            current.append(word)
            current_length += needed_length

    if current:
        chunks.append(' '.join(current))

    return chunks
