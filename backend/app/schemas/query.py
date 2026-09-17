from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, examples=["What is overfitting?"])

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be blank.")
        return value


class PassageEvidence(BaseModel):
    chunk_id: str
    source: str
    page: int
    distance: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    passages: list[PassageEvidence] = Field(default_factory=list)
