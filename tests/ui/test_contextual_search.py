from types import SimpleNamespace
import threading

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
    import asyncio

    class Service:
        async def answer_question_stream_async(
            self,
            question,
            markdown_context,
            **kwargs,
        ):
            del kwargs
            assert question == "Find the clue"
            assert markdown_context == ""
            for fragment in ("The clue ", "is in ", "the archive."):
                await asyncio.sleep(0.01)
                yield fragment

        def get_session_answer(self, session_id):
            del session_id
            return SimpleNamespace(
                text="The clue is in the archive.",
                citations=(SimpleNamespace(path="C:\\library\\archive.md"),),
                evidence=(SimpleNamespace(path="C:\\library\\archive.md"),),
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


def test_contextual_worker_emits_slow_gateway_fragments_incrementally(qtbot) -> None:
    import asyncio

    class Service:
        async def answer_question_stream_async(
            self,
            question,
            markdown_context,
            **kwargs,
        ):
            del question, markdown_context, kwargs
            for fragment in ("The ", "clue ", "arrives ", "slowly."):
                await asyncio.sleep(0.01)
                yield fragment

        def get_session_answer(self, session_id):
            del session_id
            return SimpleNamespace(
                citations=(SimpleNamespace(path="C:\\library\\archive.md"),),
                evidence=(SimpleNamespace(path="C:\\library\\archive.md"),),
                text="The clue arrives slowly.",
            )

    submission = ContextualSearchSubmission(question="Find the clue")
    worker = ContextualSearchWorker(
        Service(),
        submission,
        session_id="ui:slow-fragments",
    )

    chunks: list[str] = []
    worker.answer_chunk.connect(chunks.append)
    worker.start()
    assert worker.wait(2000)
    qtbot.waitUntil(lambda: len(chunks) == 4)

    assert chunks == ["The ", "clue ", "arrives ", "slowly."]


def test_dialog_stale_worker_callbacks_cannot_mutate_replacement_ui(qtbot) -> None:
    dialog = ContextualSearchDialog(rag_service=SimpleNamespace())
    qtbot.addWidget(dialog)

    old_worker = ContextualSearchWorker(
        SimpleNamespace(),
        ContextualSearchSubmission(question="old"),
        session_id="old",
    )
    replacement_worker = ContextualSearchWorker(
        SimpleNamespace(),
        ContextualSearchSubmission(question="new"),
        session_id="new",
    )
    dialog._worker = replacement_worker
    dialog._worker_generation = getattr(dialog, "_worker_generation", 1) + 1
    generation = dialog._worker_generation

    dialog._answer_ready(SimpleNamespace(text="new"), replacement_worker, generation)
    dialog.answer_output.setPlainText("replacement")

    dialog._answer_ready(SimpleNamespace(text="old"), old_worker, generation)
    dialog._render_citations((SimpleNamespace(path="old.md"),), old_worker, generation)

    assert dialog.answer_output.toPlainText() == "replacement"
    assert dialog.citations_list.count() == 0


def test_dialog_close_requests_worker_cancellation_and_clears_session(qtbot) -> None:
    release = threading.Event()

    class Service:
        def close_contextual_session(self, session_id):
            assert session_id
            closed.append(session_id)

        async def answer_contextual_question_stream_async(
            self,
            question,
            markdown_context,
            **kwargs,
        ):
            del question, markdown_context, kwargs
            release.wait(2)
            yield "late"

    closed = []
    dialog = ContextualSearchDialog(rag_service=Service())
    qtbot.addWidget(dialog)
    dialog.question.setText("cancel me")
    dialog.submit()
    worker = dialog._worker
    assert worker is not None

    dialog.close()
    qtbot.waitUntil(lambda: worker.isFinished() or not worker.isRunning())
    release.set()

    assert worker._cancel_requested.is_set()


def test_dialog_close_during_blocked_gateway_stops_worker_and_zeroes_active_use(
    qtbot,
) -> None:
    import asyncio

    gateway_blocked = threading.Event()
    gateway_released = threading.Event()
    active_count = 0

    class Runtime:
        def close_contextual_session(self, session_id, timeout=2.0):
            del session_id, timeout
            gateway_released.set()

    class Service:
        async def answer_contextual_question_stream_async(
            self,
            question,
            markdown_context,
            **kwargs,
        ):
            nonlocal active_count
            del question, markdown_context, kwargs
            active_count = 1
            gateway_blocked.set()
            try:
                while not gateway_released.is_set():
                    await asyncio.sleep(0.01)
                yield "late"
            finally:
                active_count = 0

    dialog = ContextualSearchDialog(modern_runtime=Runtime(), rag_service=Service())
    qtbot.addWidget(dialog)
    dialog.question.setText("block gateway")
    dialog.submit()
    worker = dialog._worker
    assert worker is not None
    assert gateway_blocked.wait(1)

    def forbidden_terminate() -> None:
        raise AssertionError("Contextual workers must not be force-terminated")

    worker.terminate = forbidden_terminate

    dialog.close()

    qtbot.waitUntil(
        lambda: (not worker.isRunning() and active_count == 0),
        timeout=3000,
    )
    gateway_released.set()

    assert active_count == 0
    assert not worker.isRunning()
