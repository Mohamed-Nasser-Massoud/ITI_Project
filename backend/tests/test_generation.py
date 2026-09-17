from app.services.generation import GenerationService
from app.services.retrieval import RetrievedChunk


def test_ensure_citations_converts_page_markers_to_exact_source_labels() -> None:
    passages = [
        RetrievedChunk(
            chunk_id="book-p58-c0",
            document="hands_on_ml.pdf",
            page=58,
            text="Overfitting means poor generalization.",
            distance=0.2,
        ),
        RetrievedChunk(
            chunk_id="book-p57-c0",
            document="hands_on_ml.pdf",
            page=57,
            text="More training data can reduce overfitting.",
            distance=0.3,
        ),
    ]

    answer = GenerationService.ensure_citations("Overfitting is poor generalization [58].", passages)

    assert "[58]" not in answer
    assert "[hands_on_ml.pdf | page 58 | book-p58-c0]" in answer


def test_ensure_citations_converts_page_and_chunk_markers() -> None:
    passage = RetrievedChunk(
        chunk_id="book-p58-c0",
        document="hands_on_ml.pdf",
        page=58,
        text="Overfitting means poor generalization.",
        distance=0.2,
    )

    answer = GenerationService.ensure_citations("The model overfits [58 | book-p58-c0].", [passage])

    assert answer == "The model overfits [hands_on_ml.pdf | page 58 | book-p58-c0]."


def test_ensure_citations_adds_sources_when_model_omits_citations() -> None:
    passage = RetrievedChunk(
        chunk_id="book-p12-c0",
        document="hands_on_ml.pdf",
        page=12,
        text="A cited fact.",
        distance=0.1,
    )

    answer = GenerationService.ensure_citations("A factual answer.", [passage])

    assert answer.endswith("Citations: [hands_on_ml.pdf | page 12 | book-p12-c0]")
