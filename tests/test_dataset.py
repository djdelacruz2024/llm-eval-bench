from llm_eval_bench.dataset import GoldenQuestion, NoRagCase, load_golden_dataset, load_no_rag_dataset


def test_load_golden_dataset_has_entries(golden_dataset_path):
    questions = load_golden_dataset(golden_dataset_path)
    assert len(questions) > 0
    assert all(isinstance(q, GoldenQuestion) for q in questions)
    assert all(q.question and q.reference_answer for q in questions)


def test_golden_dataset_ids_are_unique(golden_dataset_path):
    questions = load_golden_dataset(golden_dataset_path)
    ids = [q.id for q in questions]
    assert len(ids) == len(set(ids))


def test_golden_question_requires_rag_reflects_context():
    with_context = GoldenQuestion(id="x", category="c", question="q", reference_answer="a", context=["ctx"])
    without_context = GoldenQuestion(id="y", category="c", question="q", reference_answer="a")
    assert with_context.requires_rag is True
    assert without_context.requires_rag is False


def test_load_no_rag_dataset_has_entries(no_rag_dataset_path):
    cases = load_no_rag_dataset(no_rag_dataset_path)
    assert len(cases) > 0
    assert all(isinstance(c, NoRagCase) for c in cases)
    assert all(c.acceptable_behaviors for c in cases)


def test_no_rag_dataset_ids_are_unique(no_rag_dataset_path):
    cases = load_no_rag_dataset(no_rag_dataset_path)
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))
