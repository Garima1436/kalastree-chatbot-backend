
# graph.py

from langgraph.graph import StateGraph, END

from nodes import (
    GraphState,
    rewrite_query,
    grade_retrieval,
    generate_answer,
    check_grounding
)

from retrieval import retrieve_context


# =====================================================
# RETRIEVE NODE
# =====================================================

def retrieve(state):

    query = state["rewritten_query"]

    context, docs = retrieve_context(
        query,
        k=10
    )

    return {
        "context": context,
        "docs": docs
    }


# =====================================================
# RETRY RETRIEVAL
# =====================================================

def retrieve_again(state):

    query = state["rewritten_query"]

    context, docs = retrieve_context(
        query,
        k=25
    )

    return {
        "context": context,
        "docs": docs
    }


# =====================================================
# DECISION
# =====================================================

def retrieval_router(state):

    grade = state["retrieval_grade"]

    if "BAD" in grade:

        return "retrieve_again"

    return "generate_answer"


# =====================================================
# GROUNDING ROUTER
# =====================================================

def grounding_router(state):

    grade = state["grounding_grade"]

    if "UNSUPPORTED" in grade:

        return END

    return END


# =====================================================
# GRAPH
# =====================================================

workflow = StateGraph(GraphState)

workflow.add_node(
    "rewrite_query",
    rewrite_query
)

workflow.add_node(
    "retrieve",
    retrieve
)

workflow.add_node(
    "grade_retrieval",
    grade_retrieval
)

workflow.add_node(
    "retrieve_again",
    retrieve_again
)

workflow.add_node(
    "generate_answer",
    generate_answer
)

workflow.add_node(
    "check_grounding",
    check_grounding
)

workflow.set_entry_point(
    "rewrite_query"
)

workflow.add_edge(
    "rewrite_query",
    "retrieve"
)

workflow.add_edge(
    "retrieve",
    "grade_retrieval"
)

workflow.add_conditional_edges(
    "grade_retrieval",
    retrieval_router
)

workflow.add_edge(
    "retrieve_again",
    "generate_answer"
)

workflow.add_edge(
    "generate_answer",
    "check_grounding"
)

workflow.add_conditional_edges(
    "check_grounding",
    grounding_router
)

graph = workflow.compile()

