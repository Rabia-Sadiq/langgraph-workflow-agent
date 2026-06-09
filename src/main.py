import json
from src.graph import workflow
from src.state import AgentState
from src.utils.logger import logger, print_trace


def run_query(user_query: str) -> dict:
    """
    Runs a single user query through the full LangGraph workflow.
    Returns the final structured output.
    """
    logger.info(f"Running query: {user_query!r}")

    initial_state: AgentState = {
        "user_query":      user_query,
        "intent":          None,
        "route":           None,
        "raw_answer":      None,
        "review_status":   None,
        "review_feedback": None,
        "final_answer":    None,
        "retry_count":     0,
        "trace":           [],
    }

    result = workflow.invoke(initial_state)

    # Print trace to console
    print_trace(result.get("trace", []))

    # Parse and return final answer
    final_raw = result.get("final_answer", "{}")
    try:
        return json.loads(final_raw)
    except json.JSONDecodeError:
        return {"final_answer": final_raw}


if __name__ == "__main__":
    print("\n LangGraph Workflow Agent")
    print("Type your query below. Press Ctrl+C to exit.\n")

    while True:
        try:
            query = input("You: ").strip()
            if not query:
                continue

            output = run_query(query)

            print("\n── RESULT ──────────────────────────────────────────")
            print(json.dumps(output, indent=2, ensure_ascii=False))
            print("────────────────────────────────────────────────────\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break