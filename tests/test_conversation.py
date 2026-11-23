from app.conversation import rewrite_question

chat_history = [
    {
        "role":"user",
        "content":"what is machine learning"
    },
    {
        "role":"assistant",
        "content":(
            "Machine learning is a field of AI"
            "that learns patterns from data."
        )
    }
]

question = "what are its type"

rewritten = rewrite_question(
    question,
    chat_history
)

print("Original question:",question)
print("Rewritten question:",rewritten)