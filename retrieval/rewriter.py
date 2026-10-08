"""Turn a follow-up question into a standalone one using chat history."""
from langchain_openai import ChatOpenAI

from core.config import settings
from retrieval.prompts import REWRITE_SYSTEM_PROMPT


def format_history(history: list[dict], max_turns: int = 6) -> str:
    recent = history[-max_turns * 2 :]
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent)


class QueryRewriter:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.llm_model, api_key=settings.openai_api_key, temperature=0
        )

    def rewrite(self, question: str, history: list[dict]) -> str:
        if not history:  # first question: nothing to resolve, save an API call
            return question
        messages = [
            ("system", REWRITE_SYSTEM_PROMPT),
            ("human", f"Chat history:\n{format_history(history)}\n\nLatest question: {question}"),
        ]
        rewritten = self.llm.invoke(messages).content.strip().strip('"')
        return rewritten or question