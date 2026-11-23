
import os
import json

from app.rag_pipeline import (
    load_rag_components,
    ask_question
)


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

QUESTIONS_PATH = os.path.join(
    BASE_DIR, "data", "evaluation", "questions.json"
)

RESULTS_PATH = os.path.join(
    BASE_DIR, "data", "evaluation", "retrieval_results.json"
)

TOP_K = 3


def load_questions():

    if not os.path.exists(QUESTIONS_PATH):
        raise FileNotFoundError(
            f"Evaluation questions not found: {QUESTIONS_PATH}"
        )

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as file:
        questions = json.load(file)

    if not questions:
        raise ValueError("No evaluation questions found.")

    return questions


def evaluate_retrieval():

    questions = load_questions()

    model, index, metadata = load_rag_components()

    recall_scores = []
    reciprocal_ranks = []
    detailed_results = []

    for item in questions:

        question = item["question"]
        expected_source = item["expected_source"]

        # Retrieve document chunks
        _, retrieved_chunks = ask_question(
            question,
            model,
            index,
            metadata
        )

        # Extract retrieved document names
        retrieved_sources = [
            chunk["document_name"]
            for chunk in retrieved_chunks[:TOP_K]
        ]

        # Check whether expected source was retrieved
        relevant_found = expected_source in retrieved_sources

        recall_scores.append(
            1.0 if relevant_found else 0.0
        )

        # Find the rank of the expected source
        rank = None

        for position, source in enumerate(
            retrieved_sources,
            start=1
        ):

            if source == expected_source:
                rank = position
                break

        reciprocal_rank = (
            1.0 / rank if rank else 0.0
        )

        reciprocal_ranks.append(reciprocal_rank)

        result = {
            "question": question,
            "expected_source": expected_source,
            "retrieved_sources": retrieved_sources,
            "hit": relevant_found,
            "rank": rank,
            "reciprocal_rank": reciprocal_rank
        }

        detailed_results.append(result)

        print("\n" + "=" * 60)
        print("Question:", question)
        print("Expected source:", expected_source)
        print("Retrieved sources:", retrieved_sources)
        print("Expected source found:", relevant_found)
        print("Rank:", rank)
        print("Reciprocal rank:", round(reciprocal_rank, 4))

    # Calculate average metrics
    total_questions = len(questions)

    recall_at_3 = sum(recall_scores) / total_questions

    mrr = sum(reciprocal_ranks) / total_questions

    # Save detailed results
    os.makedirs(
        os.path.dirname(RESULTS_PATH),
        exist_ok=True
    )

    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(
            detailed_results,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n========== RETRIEVAL EVALUATION ==========")
    print("Total questions:", total_questions)
    print("Recall@3:", round(recall_at_3, 4))
    print("MRR:", round(mrr, 4))
    print("Results saved:", RESULTS_PATH)


if __name__ == "__main__":
    evaluate_retrieval()
