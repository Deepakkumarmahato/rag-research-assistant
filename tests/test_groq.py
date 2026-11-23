
from app.llm_generator import generate_answer

context = """
Machine Learning is a field of Artificial
Intelligence that allows computers to learn
patterns from data.
"""

question = "What is Machine Learning?"

answer = generate_answer(context, question)

print("\nGenerated Answer:")
print(answer)
