
from app.rag_pipeline import load_rag_components

from app.conversation import conversational_rag


# Load RAG components once
model, index, metadata = load_rag_components()

# Initialize conversation memory
chat_history = []


# First question
question1 = "What is machine learning?"

answer1, sources1, rewritten1 = conversational_rag(
    question1,
    chat_history,
    model,
    index,
    metadata
)

print("\nUSER:", question1)
print("\nASSISTANT:", answer1)


# Follow-up question
question2 = "What are its types?"

answer2, sources2, rewritten2 = conversational_rag(
    question2,
    chat_history,
    model,
    index,
    metadata
)

print("\nUSER:", question2)
print("\nASSISTANT:", answer2)

print("\n========== SOURCES ==========")

for source in sources2:
    print(
        source["document_name"],
        "| Page:",
        source["page_number"]
    )
