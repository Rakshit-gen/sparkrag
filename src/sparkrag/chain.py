from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_exponential

from sparkrag.config import GROQ_MODEL, require_groq_key
from sparkrag.store import get_vectorstore

PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using only the context below. If the context
doesn't contain the answer, say you don't have enough information instead
of guessing. Mention which source each part of your answer came from.

Context:
{context}

Question: {question}

Answer:"""
)


def _format_docs(docs) -> str:
    return "\n\n".join(f"[{d.metadata.get('source')}]\n{d.page_content}" for d in docs)


def build_chain(k: int = 4):
    require_groq_key()
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": k})
    llm = ChatGroq(model=GROQ_MODEL, temperature=0)

    return (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def _invoke(chain, question: str) -> str:
    # Retries on any exception from invoke, not just ones that look
    # transient. Groq's client doesn't give us a clean way to tell a
    # rate-limit or network blip apart from a bad request without parsing
    # error text, so this trades a little wasted retry time on genuine
    # permanent failures (e.g. a malformed question) for not having to
    # guess at Groq's exception hierarchy. The one permanent failure we
    # know about ahead of time, a missing API key, is caught by
    # require_groq_key() in build_chain before this ever runs, so it
    # fails immediately instead of retrying 3 times first.
    return chain.invoke(question)


def ask(question: str, k: int = 4) -> str:
    chain = build_chain(k=k)
    return _invoke(chain, question)
