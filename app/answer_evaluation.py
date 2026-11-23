
import os
import json
import time

from app.llm_generator import (
    build_context,
    evaluate_faithfulness
)

from app.rag_pipeline import (
    load_rag_components,
    ask_question
)

from app.evaluation import load_questions


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RESULTS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "evaluation",
    "answer_results.json"
)


def evaluate_answers():

    questions = load_questions()

    model, index, metadata = load_rag_components()

    keyword_scores = []
    response_times = []
    faithfulness_scores = []

    detailed_results = []

    for item in questions:

        question = item["question"]
        expected_keywords = item["expected_keywords"]

        # --------------------------------
        # 1. Generate RAG Answer
        # --------------------------------

        start_time = time.perf_counter()

        answer, retrieved_chunks = ask_question(
            question,
            model,
            index,
            metadata
        )

        response_time = time.perf_counter() - start_time

        # --------------------------------
        # 2. Keyword Coverage Evaluation
        # --------------------------------

        answer_lower = answer.lower()

        matched_keywords = [
            keyword
            for keyword in expected_keywords
            if keyword.lower() in answer_lower
        ]

        if expected_keywords:

            keyword_score = (
                len(matched_keywords) / len(expected_keywords)
            )

        else:
            keyword_score = 0.0

        keyword_scores.append(keyword_score)

        response_times.append(response_time)

        # --------------------------------
        # 3. Prepare Retrieved Context
        # --------------------------------

        context = build_context(retrieved_chunks)

        # --------------------------------
        # 4. Faithfulness Evaluation
        # --------------------------------

        faithfulness_result = evaluate_faithfulness(
            context=context,
            question=question,
            answer=answer
        )

        faithfulness_score = float(
            faithfulness_result["faithfulness_score"]
        )

        faithfulness_scores.append(faithfulness_score)

        # --------------------------------
        # 5. Retrieved Sources
        # --------------------------------

        retrieved_sources = [
            chunk["document_name"]
            for chunk in retrieved_chunks
        ]

        # --------------------------------
        # 6. Save Individual Results
        # --------------------------------

        result = {

            "question": question,

            "answer": answer,

            "expected_keywords": expected_keywords,

            "matched_keywords": matched_keywords,

            "keyword_coverage": keyword_score,

            "response_time_seconds": response_time,

            "retrieved_sources": retrieved_sources,

            "faithfulness": faithfulness_result

        }

        detailed_results.append(result)

        # --------------------------------
        # 7. Print Individual Results
        # --------------------------------

        print("\n" + "=" * 60)

        print("QUESTION:", question)

        print("\nGENERATED ANSWER:")
        print(answer)

        print("\nEXPECTED KEYWORDS:", expected_keywords)

        print("MATCHED KEYWORDS:", matched_keywords)

        print(
            "KEYWORD COVERAGE:",
            round(keyword_score, 4)
        )

        print(
            "RESPONSE TIME:",
            round(response_time, 2),
            "seconds"
        )

        print("\nFAITHFULNESS SCORE:")
        print(round(faithfulness_score, 4))

        print(
            "SUPPORTED CLAIMS:",
            faithfulness_result["supported_claims"]
        )

        print(
            "UNSUPPORTED CLAIMS:",
            faithfulness_result["unsupported_claims"]
        )

        print(
            "EVALUATOR REASONING:",
            faithfulness_result["reasoning"]
        )

        print("\nRETRIEVED SOURCES:")

        for chunk in retrieved_chunks:

            print(
                chunk["document_name"],
                "| Page:",
                chunk["page_number"]
            )

    # --------------------------------
    # 8. Calculate Average Metrics
    # --------------------------------

    total_questions = len(questions)

    if total_questions == 0:
        raise ValueError("No evaluation questions available.")

    average_keyword_score = (
        sum(keyword_scores) / total_questions
    )

    average_response_time = (
        sum(response_times) / total_questions
    )

    average_faithfulness = (
        sum(faithfulness_scores) / total_questions
    )

    # --------------------------------
    # 9. Save Detailed Evaluation
    # --------------------------------

    os.makedirs(
        os.path.dirname(RESULTS_PATH),
        exist_ok=True
    )

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            detailed_results,
            file,
            indent=4,
            ensure_ascii=False
        )

    # --------------------------------
    # 10. Final Evaluation Report
    # --------------------------------

    print("\n" + "=" * 60)

    print("========== ANSWER EVALUATION ==========")

    print("Total questions:", total_questions)

    print(
        "Average keyword coverage:",
        round(average_keyword_score, 4)
    )

    print(
        "Average faithfulness:",
        round(average_faithfulness, 4)
    )

    print(
        "Average response time:",
        round(average_response_time, 2),
        "seconds"
    )

    print("\nResults saved:", RESULTS_PATH)


if __name__ == "__main__":
    evaluate_answers()
