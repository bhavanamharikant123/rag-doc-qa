import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document

from retrieval.retriever import reciprocal_rank_fusion


def doc(cid: str) -> Document:
    return Document(page_content=cid, metadata={"chunk_id": cid})


def test_rrf_rewards_agreement():
    # "b" is ranked #1 by BOTH lists, so it must come first
    vector = [doc("b"), doc("a"), doc("c")]
    bm25 = [doc("b"), doc("c"), doc("d")]
    merged = [d.metadata["chunk_id"] for d in reciprocal_rank_fusion([vector, bm25])]
    assert merged[0] == "b"
    assert set(merged) == {"a", "b", "c", "d"}

def test_rrf_agreement_beats_single_list_leader():
    # "x" is #1 in only one list; "y" is #2 in both lists, so "y" should win
    vector = [doc("x"), doc("y"), doc("p")]
    bm25 = [doc("q"), doc("y"), doc("r")]
    merged = [d.metadata["chunk_id"] for d in reciprocal_rank_fusion([vector, bm25])]
    assert merged[0] == "y"


def test_rrf_no_duplicates():
    merged = reciprocal_rank_fusion([[doc("a"), doc("b")], [doc("a"), doc("b")]])
    assert len(merged) == 2