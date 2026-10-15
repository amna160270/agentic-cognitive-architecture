
"""Tests for working memory, experience memory, and knowledge retrieval."""

import pytest

from memory.working_memory import WorkingMemory
from memory.episodic_memory import EpisodicMemory, KnowledgeStore


class FakeVector(list):
    """Minimal vector object compatible with the .tolist() API."""

    def tolist(self):
        return list(self)


class FakeTokenizer:
    """Whitespace tokenizer for deterministic chunking tests."""

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(tokens)


class FakeEmbeddingModel:
    """Deterministic local substitute for fast, network-free tests."""

    tokenizer = FakeTokenizer()

    def encode(self, text):
        words = text.lower().split()
        values = [
            float(len(words) or 1),
            float(sum(len(word) for word in words) or 1),
            float(sum(ord(char) for char in text) % 997 or 1),
        ]
        return FakeVector(values)


def test_working_memory_keeps_recent_messages():
    memory = WorkingMemory(max_messages=2)
    memory.add("user", "first")
    memory.add("assistant", "second")
    memory.add("user", "third")

    messages = memory.get_messages()
    assert len(messages) == 2
    assert messages[0]["content"] == "second"
    assert messages[1]["content"] == "third"


def test_working_memory_clear():
    memory = WorkingMemory()
    memory.add("user", "hello")
    memory.clear()
    assert len(memory) == 0


def test_working_memory_rejects_invalid_limit():
    with pytest.raises(ValueError):
        WorkingMemory(max_messages=0)


def test_working_memory_counts_tokens():
    memory = WorkingMemory(token_counter=lambda text: len(text.split()))
    memory.add("user", "one two three")
    assert memory.token_counter(memory.get_context()) >= 4


def test_compaction_preserves_system_and_last_three_steps():
    summaries = []

    def fake_summarizer(text):
        summaries.append(text)
        return "Summary of earlier steps."

    memory = WorkingMemory(
        max_tokens=12,
        keep_last_steps=3,
        summarizer=fake_summarizer,
        token_counter=lambda text: len(text.split()),
    )
    memory.add("system", "Always follow the safety rules.")
    for step in range(1, 6):
        memory.add("assistant", f"Thought for step {step}", step=step)
        memory.add("assistant", f"Observation for step {step}", step=step)

    messages = memory.get_messages()
    context = memory.get_context()

    assert summaries, "Compaction should call the injected summarizer"
    assert "Always follow the safety rules." in context
    assert any(m.get("kind") == "executive_summary" for m in messages)
    assert "Thought for step 3" in context
    assert "Thought for step 4" in context
    assert "Thought for step 5" in context
    assert "Thought for step 1" not in context
    assert "Thought for step 2" not in context


def test_compaction_fallback_handles_empty_summary():
    memory = WorkingMemory(
        max_tokens=5,
        summarizer=lambda text: "",
        token_counter=lambda text: len(text.split()),
    )
    for step in range(5):
        memory.add("assistant", f"Older detail {step}", step=step)

    summaries = [
        message for message in memory.get_messages()
        if message.get("kind") == "executive_summary"
    ]
    assert summaries
    assert summaries[0]["content"]


def test_episodic_memory_add_and_search(tmp_path):
    memory = EpisodicMemory(
        persist_directory=str(tmp_path / "memory_db"),
        embedding_model=FakeEmbeddingModel(),
    )
    memory.add("The agent solved a Python calculation task.")
    results = memory.search("Python calculation", n_results=1)

    assert len(results) == 1
    assert "Python calculation" in results[0]["text"]


def test_episodic_memory_rejects_empty_text(tmp_path):
    memory = EpisodicMemory(
        persist_directory=str(tmp_path / "memory_db"),
        embedding_model=FakeEmbeddingModel(),
    )
    with pytest.raises(ValueError):
        memory.add("  ")


def test_episodic_memory_count(tmp_path):
    memory = EpisodicMemory(
        persist_directory=str(tmp_path / "memory_db"),
        embedding_model=FakeEmbeddingModel(),
    )
    memory.add("First episode")
    memory.add("Second episode")
    assert memory.count() == 2


def test_episodic_memory_empty_search(tmp_path):
    memory = EpisodicMemory(
        persist_directory=str(tmp_path / "memory_db"),
        embedding_model=FakeEmbeddingModel(),
    )
    assert memory.search("anything") == []


def test_knowledge_chunks_have_expected_overlap(tmp_path):
    knowledge = KnowledgeStore(
        persist_directory=str(tmp_path / "memory_db"),
        chunk_tokens=5,
        overlap_tokens=2,
        embedding_model=FakeEmbeddingModel(),
    )

    chunks = knowledge.chunk_text("one two three four five six seven eight")

    assert chunks == [
        "one two three four five",
        "four five six seven eight",
    ]
    assert len(chunks[0].split()) == 5
    assert len(chunks[1].split()) == 5


def test_knowledge_and_experience_stores_are_isolated(tmp_path):
    db_path = str(tmp_path / "memory_db")
    model = FakeEmbeddingModel()

    experiences = EpisodicMemory(
        persist_directory=db_path,
        embedding_model=model,
    )
    knowledge = KnowledgeStore(
        persist_directory=db_path,
        embedding_model=model,
        chunk_tokens=5,
        overlap_tokens=1,
    )

    experiences.add("A successful Python task trace.")
    knowledge.ingest_text("one two three four five six", source="notes.txt")

    assert experiences.count() == 1
    assert knowledge.count() == 2
    assert "successful Python task" in experiences.search("Python")[0]["text"]
    assert all(
        "successful Python task" not in item["text"]
        for item in knowledge.search("Python", n_results=5)
    )
