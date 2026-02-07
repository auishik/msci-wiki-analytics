import pytest

from src.services.frequency import WordFrequencyService


class TestWordFrequencyService:
    @pytest.fixture
    def service(self) -> WordFrequencyService:
        return WordFrequencyService()

    def test_calculate_counts_words(self, service: WordFrequencyService) -> None:
        texts = ["hello world hello"]
        result = service.calculate(texts)

        assert "hello" in result.root
        assert result.root["hello"].count == 2
        assert "world" in result.root
        assert result.root["world"].count == 1

    def test_calculate_percentages(self, service: WordFrequencyService) -> None:
        texts = ["a a a b"]  # 4 words total, 'a' is 75%, 'b' is 25%
        result = service.calculate(texts)

        assert result.root["a"].percentage == pytest.approx(75.0)
        assert result.root["b"].percentage == pytest.approx(25.0)

    def test_calculate_combines_multiple_texts(
        self, service: WordFrequencyService
    ) -> None:
        texts = ["hello world", "hello python"]
        result = service.calculate(texts)

        assert result.root["hello"].count == 2
        assert result.root["world"].count == 1
        assert result.root["python"].count == 1

    def test_calculate_with_ignore_list(self, service: WordFrequencyService) -> None:
        texts = ["hello world hello python"]
        result = service.calculate(texts, ignore_list=["hello", "world"])

        assert "hello" not in result.root
        assert "world" not in result.root
        assert "python" in result.root

    def test_calculate_ignore_list_case_insensitive(
        self, service: WordFrequencyService
    ) -> None:
        texts = ["Hello WORLD hello"]
        result = service.calculate(texts, ignore_list=["HELLO"])

        assert "hello" not in result.root
        assert "world" in result.root

    def test_calculate_recalculates_percentage_after_ignore(
        self, service: WordFrequencyService
    ) -> None:
        texts = ["a a b b c"]  # Without ignore: a=40%, b=40%, c=20%
        result = service.calculate(texts, ignore_list=["a"])

        # After ignoring 'a': b=2/3=66.67%, c=1/3=33.33%
        assert result.root["b"].percentage == pytest.approx(66.666, rel=0.01)
        assert result.root["c"].percentage == pytest.approx(33.333, rel=0.01)

    def test_calculate_with_percentile_filter(
        self, service: WordFrequencyService
    ) -> None:
        texts = ["a a a a b b b c c d"]  # 4 unique words
        result = service.calculate(texts, percentile=50)

        # Top 50% means top 2 words (a and b)
        assert len(result.root) == 2
        assert "a" in result.root
        assert "b" in result.root
        assert "c" not in result.root
        assert "d" not in result.root

    def test_calculate_with_percentile_100_returns_empty(
        self, service: WordFrequencyService
    ) -> None:
        texts = ["a b c d"]
        result = service.calculate(texts, percentile=100)

        assert len(result.root) == 0

    def test_calculate_with_percentile_0_returns_all(
        self, service: WordFrequencyService
    ) -> None:
        texts = ["a b c d"]
        result = service.calculate(texts, percentile=0)

        assert len(result.root) == 4

    def test_calculate_empty_texts(self, service: WordFrequencyService) -> None:
        result = service.calculate([])

        assert len(result.root) == 0

    def test_calculate_empty_string(self, service: WordFrequencyService) -> None:
        result = service.calculate([""])

        assert len(result.root) == 0

    def test_calculate_removes_punctuation(self, service: WordFrequencyService) -> None:
        texts = ["hello, world! hello? world."]
        result = service.calculate(texts)

        assert result.root["hello"].count == 2
        assert result.root["world"].count == 2
        assert "hello," not in result.root

    def test_calculate_ignores_numbers(self, service: WordFrequencyService) -> None:
        texts = ["hello 123 world 456"]
        result = service.calculate(texts)

        assert "123" not in result.root
        assert "456" not in result.root
        assert len(result.root) == 2

    def test_calculate_normalizes_to_lowercase(
        self, service: WordFrequencyService
    ) -> None:
        texts = ["Hello HELLO HeLLo"]
        result = service.calculate(texts)

        assert "hello" in result.root
        assert result.root["hello"].count == 3
        assert "Hello" not in result.root


class TestTokenize:
    @pytest.fixture
    def service(self) -> WordFrequencyService:
        return WordFrequencyService()

    def test_tokenize_splits_on_whitespace(self, service: WordFrequencyService) -> None:
        result = service._tokenize("hello world")
        assert result == ["hello", "world"]

    def test_tokenize_removes_punctuation(self, service: WordFrequencyService) -> None:
        result = service._tokenize("hello, world!")
        assert result == ["hello", "world"]

    def test_tokenize_lowercases(self, service: WordFrequencyService) -> None:
        result = service._tokenize("Hello WORLD")
        assert result == ["hello", "world"]

    def test_tokenize_filters_numbers(self, service: WordFrequencyService) -> None:
        result = service._tokenize("hello 123 world")
        assert result == ["hello", "world"]

    def test_tokenize_handles_multiple_spaces(
        self, service: WordFrequencyService
    ) -> None:
        result = service._tokenize("hello    world")
        assert result == ["hello", "world"]
