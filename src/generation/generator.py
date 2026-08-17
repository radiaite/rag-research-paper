"""LLM generation pipeline for end-to-end RAG evaluation."""

from __future__ import annotations

from src.utils.common import RetrievedDoc, get_logger

logger = get_logger(__name__)

DEFAULT_PROMPT = """Answer the following question based ONLY on the provided context.
If the answer is a number, provide just the number.
If you cannot answer from the context, say "UNANSWERABLE".

Context:
{context}

Question: {question}

Answer:"""


class Generator:
    """LLM-based answer generator for RAG pipeline."""

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        max_tokens: int = 256,
        prompt_template: str = DEFAULT_PROMPT,
    ):
        import os

        from openai import AzureOpenAI
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.getenv("AZURE_LLM_ENDPOINT"),
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_template = prompt_template

    def generate(self, question: str, contexts: list[RetrievedDoc]) -> str:
        """Generate answer from question and retrieved contexts."""
        context_text = "\n\n---\n\n".join(
            f"[Document {i+1}]\n{doc.text}" for i, doc in enumerate(contexts)
        )

        prompt = self.prompt_template.format(
            context=context_text, question=question
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content.strip()

    def generate_batch(
        self,
        questions: list[str],
        all_contexts: list[list[RetrievedDoc]],
        show_progress: bool = True,
    ) -> list[str]:
        """Generate answers for a batch of questions."""
        from tqdm import tqdm
        answers = []
        iterator = zip(questions, all_contexts)
        if show_progress:
            iterator = tqdm(list(iterator), desc="Generating answers")

        for question, contexts in iterator:
            try:
                answer = self.generate(question, contexts)
            except Exception as e:
                logger.warning(f"Generation failed for '{question[:50]}...': {e}")
                answer = "UNANSWERABLE"
            answers.append(answer)
        return answers
