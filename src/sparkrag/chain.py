from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

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


def ask(question: str, k: int = 4) -> str:
    chain = build_chain(k=k)
    return chain.invoke(question)
