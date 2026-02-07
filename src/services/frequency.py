import logging
import re
from collections import Counter

from src.models.schemas import WordFrequency, WordFrequencyResponse

logger = logging.getLogger(__name__)


class WordFrequencyService:
    """Service for calculating word frequencies from text."""

    def calculate(
        self,
        texts: list[str],
        ignore_list: list[str] | None = None,
        percentile: int | None = None,
    ) -> WordFrequencyResponse:
        """
        Calculate word frequencies from a list of texts.

        Args:
            texts: List of text strings to analyze.
            ignore_list: Words to exclude from results.
            percentile: If provided, only include words at or above this
                       percentile in frequency ranking (0-100).

        Returns:
            WordFrequencyResponse with word frequencies.
        """
        logger.info(
            "Calculating word frequency for %d texts, ignore_list=%d words, "
            "percentile=%s",
            len(texts),
            len(ignore_list) if ignore_list else 0,
            percentile,
        )
        ignore_set = {word.lower() for word in (ignore_list or [])}

        # Tokenize and count all words
        word_counts: Counter[str] = Counter()
        for text in texts:
            words = self._tokenize(text)
            word_counts.update(words)

        logger.debug("Total words counted: %d", sum(word_counts.values()))
        logger.debug("Unique words before filtering: %d", len(word_counts))

        # Filter out ignored words
        for word in ignore_set:
            word_counts.pop(word, None)

        if not word_counts:
            logger.info("No words remaining after filtering")
            return WordFrequencyResponse({})

        total_words = sum(word_counts.values())

        # Calculate percentages
        frequencies: dict[str, WordFrequency] = {}
        for word, count in word_counts.items():
            percentage = (count / total_words) * 100
            frequencies[word] = WordFrequency(count=count, percentage=percentage)

        # Apply percentile filter if specified
        if percentile is not None:
            before_count = len(frequencies)
            frequencies = self._filter_by_percentile(frequencies, percentile)
            logger.debug(
                "Percentile filter applied: %d -> %d words",
                before_count,
                len(frequencies),
            )

        logger.info(
            "Word frequency calculation complete: %d unique words", len(frequencies)
        )
        return WordFrequencyResponse(frequencies)

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text into lowercase words.

        Removes punctuation and filters out empty strings and pure numbers.
        """
        # Remove punctuation and convert to lowercase
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)

        # Split into words and filter
        words = text.split()
        return [word for word in words if word and not word.isdigit()]

    def _filter_by_percentile(
        self,
        frequencies: dict[str, WordFrequency],
        percentile: int,
    ) -> dict[str, WordFrequency]:
        """
        Filter words to only include those at or above the given percentile.

        A percentile of 90 means only the top 10% most frequent words.
        """
        if not frequencies:
            return frequencies

        # Sort by count descending
        sorted_words = sorted(
            frequencies.items(),
            key=lambda x: x[1].count,
            reverse=True,
        )

        # Calculate cutoff index
        total_unique = len(sorted_words)
        cutoff_index = int(total_unique * (percentile / 100))

        # Keep words from start to cutoff
        top_words = sorted_words[: total_unique - cutoff_index]

        return {word: freq for word, freq in top_words}
