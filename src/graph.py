from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.config import MAX_RETRIES
from src.nodes import (
    classifier_node,
    summary_node,
    qa_node,
    creative_node,
    general_node,
    reviewer_node,
    final_response_node,
)


# ── Routing functions ──────────────────────────────────────────────────────────

def route_by_intent(state: AgentState) -> str:
    """
    After classifier: decide which handler node to go to.
    """
    route = state.get("route", "general_node")
    valid_routes = {"summary_node", "qa_node", "creative_node", "general_node"}
    if route not in valid_routes:
        return "general_node"
    return route


def route_after_review(state: AgentState) -> str:
    """
    After reviewer: either retry the handler or move to final response.
    """
    status = state.get("review_status", "passed")
    retry_count = state.get("retry_count", 0)

    if status == "needs_revision" and retry_count < MAX_RETRIES:
        # Increment retry count and go back to the same handler
        state["retry_count"] = retry_count + 1
        return state.get("route", "general_node")

    return "final_response_node"


# ── Build the graph ────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Register all nodes
    graph.add_node("classifier_node",    classifier_node)
    graph.add_node("summary_node",       summary_node)
    graph.add_node("qa_node",            qa_node)
    graph.add_node("creative_node",      creative_node)
    graph.add_node("general_node",       general_node)
    graph.add_node("reviewer_node",      reviewer_node)
    graph.add_node("final_response_node", final_response_node)

    # Entry point
    graph.set_entry_point("classifier_node")

    # Classifier → conditional route to handler
    graph.add_conditional_edges(
        "classifier_node",
        route_by_intent,
        {
            "summary_node":  "summary_node",
            "qa_node":       "qa_node",
            "creative_node": "creative_node",
            "general_node":  "general_node",
        },
    )

    # Each handler → reviewer
    for handler in ["summary_node", "qa_node", "creative_node", "general_node"]:
        graph.add_edge(handler, "reviewer_node")

    # Reviewer → conditional: retry handler OR final response
    graph.add_conditional_edges(
        "reviewer_node",
        route_after_review,
        {
            "summary_node":        "summary_node",
            "qa_node":             "qa_node",
            "creative_node":       "creative_node",
            "general_node":        "general_node",
            "final_response_node": "final_response_node",
        },
    )

    # Final response → END
    graph.add_edge("final_response_node", END)

    return graph.compile()


# Compiled graph instance — imported by main.py
workflow = build_graph()