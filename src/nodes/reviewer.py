import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import GEMINI_API_KEY, MODEL_NAME, TEMPERATURE
from src.state import AgentState

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=0.1,
    google_api_key=GEMINI_API_KEY,
)

REVIEWER_SYSTEM_PROMPT = """
You are a strict answer quality reviewer.
Given a user query and an AI-generated answer, evaluate the answer quality.

Respond with ONLY a valid JSON object. No explanation. No markdown. Example:
{"status": "passed", "feedback": "Answer is clear and complete."}

Rules:
- status must be exactly "passed" or "needs_revision"
- Mark "needs_revision" if the answer is vague, too short, off-topic, or factually suspicious
- Mark "passed" if the answer is relevant, clear, and reasonably complete
- feedback must be one sentence explaining your decision
"""


def reviewer_node(state: AgentState) -> AgentState:
    """
    Node 3: Reviews the raw answer. Sets review_status to 'passed' or 'needs_revision'.
    """
    trace = state.get("trace", [])
    trace.append(">> [reviewer_node] Reviewing answer quality...")

    messages = [
        SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"User query: {state['user_query']}\n\n"
                f"AI answer: {state['raw_answer']}"
            )
        ),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    try:
        parsed = json.loads(raw)
        status = parsed.get("status", "passed")
        feedback = parsed.get("feedback", "")
    except json.JSONDecodeError:
        status = "passed"
        feedback = "Could not parse reviewer response, defaulting to passed."
        trace.append("   [reviewer_node] WARNING: JSON parse failed, defaulting to passed")

    trace.append(f"   [reviewer_node] Status: {status} | Feedback: {feedback}")

    return {
        **state,
        "review_status": status,
        "review_feedback": feedback,
        "trace": trace,
    }