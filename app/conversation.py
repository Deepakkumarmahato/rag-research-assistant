from app.llm_generator import client,build_context

from app.rag_pipeline import (
    load_rag_components,
    ask_question
)

MODEL_NAME = "openai/gpt-oss-20b"
def rewrite_question(question,chat_history):
    # if there is no previous conversion,
    # the question is already standalone
    if not chat_history:
        return question

    # keep only recent conversation message
    recent_history = chat_history[-6:]
    history_text = ""
    for message in recent_history:
        role = message["role"]
        content = message["content"]
        history_text += f"{role}: {content}\n"
    prompt = f"""
you are a question rewriting assistant.
your task is to convert the lastest user question
into a standalone question using the conversation history.

conversation history:
{history_text}

Latest user question:
{question}

Rules:
1. Reslove references such as it,they,this and that
2. preserve the original meaning
3. Do not answer the question
4. if the question is already standalone keep its meaning
5. Return only the rewritten question.

Standalone question:
"""
    response = client.chat.completions.create(
        model = MODEL_NAME,
        messages = [
            {
                "role":"user",
                "content":prompt
            }
        ],
        temperature=0
    )
    rewritten_question = (
        response.choices[0].message.content.strip()
    )

    return rewritten_question


def conversational_rag(
    question,
    chat_history,
    model,
    index,
    metadata
):

    # Step 1: Rewrite the follow-up question
    standalone_question = rewrite_question(
        question,
        chat_history
    )

    print("\nRewritten question:", standalone_question)

    # Step 2: Retrieve relevant document chunks
    retrieved_chunks = ask_question(
        standalone_question,
        model,
        index,
        metadata
    )[1]

    # Step 3: Build context from retrieved chunks
    context = build_context(retrieved_chunks)

    # Step 4: Prepare conversation history
    history_text = ""

    for message in chat_history[-6:]:

        history_text += (
            f'{message["role"]}: '
            f'{message["content"]}\n'
        )

    # Step 5: Generate answer using Groq
    prompt = f"""
You are an AI research assistant.

Answer the user's question using the retrieved
document context.

You may use conversation history to understand
references and follow-up questions.

Important rules:
1. Give priority to retrieved document context.
2. Do not invent information from the documents.
3. If the answer is unavailable in the context,
   clearly say so.
4. Give a clear and understandable answer.

Conversation history:
{history_text}

Retrieved context:
{context}

User question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    answer = response.choices[0].message.content

    # Step 6: Update conversation history
    chat_history.append({
        "role": "user",
        "content": question
    })

    chat_history.append({
        "role": "assistant",
        "content": answer
    })

    return answer, retrieved_chunks, standalone_question



