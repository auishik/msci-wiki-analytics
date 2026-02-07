from pydantic import BaseModel, Field, RootModel


class HealthResponse(BaseModel):
    status: str = Field(description="Health status of the service")


class WordFrequency(BaseModel):
    count: int = Field(description="Number of occurrences of the word")
    percentage: float = Field(description="Percentage frequency of the word")


class WordFrequencyResponse(RootModel[dict[str, WordFrequency]]):
    """Response containing word frequency data as a direct dictionary."""

    pass


class WordFrequencyRequest(BaseModel):
    article: str = Field(description="The title of the Wikipedia article to start from")
    depth: int = Field(
        ge=0, description="The depth of traversal within Wikipedia articles"
    )


class KeywordsRequest(BaseModel):
    article: str = Field(description="The title of the Wikipedia article")
    depth: int = Field(ge=0, description="The depth of traversal")
    ignore_list: list[str] = Field(description="List of words to ignore")
    percentile: int = Field(
        ge=0, le=100, description="Percentile threshold for word frequency"
    )
