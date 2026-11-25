from types import SimpleNamespace

from file_tagger.ui.contextual_search import (
    ContextualSearchDialog,
    ContextualSearchSubmission,
    ContextualSearchWorker,
)


def test_question_accepts_optional_markdown_context(qtbot, tmp_path) -> None:
    dialog = ContextualSearchDialog()
    qtbot.addWidget(dialog)
    dialog.question.setText("Find encounters and treasure")
    context_file = tmp_path / "outline.md"
    context_file.write_text("# Outline\nA flooded temple", encoding="utf-8")
    dialog.set_context_file(context_file)

    submission = dialog.submission()

    assert submission.question == "Find encounters and treasure"
    assert submission.context_file is not None
    assert submission.context_file.name == "outline.md"


def test_missing_readiness_disables_submission_and_shows_setup(qtbot) -> None:
    report = SimpleNamespace(
        ready=False,
        instructions=("Run ollama pull gemma4:e4b",),
        errors=("Generation model is missing",),
    )
    dialog = ContextualSearchDialog(readiness=report)
    qtbot.addWidget(dialog)

    assert dialog.submission_button.isEnabled() is False
    assert "ollama pull" in dialog.setup_guidance.text().lower()


def test_ready_dialog_accepts_query_without_context_file(qtbot) -> None:
    report = SimpleNamespace(ready=True, instructions=(), errors=())
    dialog = ContextualSearchDialog(readiness=report)
    qtbot.addWidget(dialog)
    dialog.question.setText("What is the third clue?")

    assert dialog.submission().context_file is None
    assert dialog.submission_button.isEnabled() is True


def test_submission_is_frozen_typed_value() -> None:
    submission = ContextualSearchSubmission(question="Clues")
    assert submission.context_file is None


def test_contextual_worker_streams_answer_and_evidence(qtbot) -> None:
    class Service:
        def answer_question(self, question, markdown_context="", **kwargs):
            del kwargs
            assert question == "Find the clue"
            assert markdown_context == ""
            return SimpleNamespace(
                text="The clue is in the archive.",
                citations=(SimpleNamespace(path="C:\\library\\archive.md"),),
                evidence=(
                    SimpleNamespace(path="C:\\library\\archive.md"),
                ),
            )

    submission = ContextualSearchSubmission(question="Find the clue")
    dialog = ContextualSearchDialog(rag_service=Service())
    qtbot.addWidget(dialog)
    dialog.question.setText("Find the clue")
    worker = ContextualSearchWorker(Service(), submission)

    chunks: list[str] = []
    worker.answer_chunk.connect(chunks.append)
    worker.start()
    assert worker.wait(2000)
    qtbot.waitUntil(lambda: len(chunks) > 0)

    assert "".join(chunks) == "The clue is in the archive."
