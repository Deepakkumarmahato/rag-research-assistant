
from app.rag_pipeline import ask_question


class DummyIndex:
    ntotal = 1


class DummyModel:
    pass


class DummyMetadata:
    def __len__(self):
        return 1


model = DummyModel()
index = DummyIndex()
metadata = DummyMetadata()


test_questions = [
    ("Empty question", ""),
    ("Whitespace question", "   "),
    ("Very long question", "A" * 2001)
]

print("\n========== INPUT VALIDATION TEST ==========")

for test_name, question in test_questions:

    try:

        ask_question(
            question,
            model,
            index,
            metadata
        )

        print(f"FAILED: {test_name} was accepted.")

    except ValueError as error:

        print(f"PASSED: {test_name}")
        print("Message:", error)


print("\nAll invalid-input tests completed.")
