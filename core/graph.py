import streamlit as st
from langgraph.graph import END, StateGraph
from agents import architect, statistician, visualizer, summary, insights
from core.state import AnalysisState


@st.cache_resource
def build_graph():
    """Architect -> Statistician -> Visualizer -> Summary -> Insights.
    Cached as a resource so Streamlit doesn't rebuild on every rerun.
    """
    graph = StateGraph(AnalysisState)

    graph.add_node("architect",   architect.run)
    graph.add_node("statistician", statistician.run)
    graph.add_node("visualizer",  visualizer.run)
    graph.add_node("summary",     summary.run)
    graph.add_node("insights",    insights.run)

    graph.set_entry_point("architect")
    graph.add_edge("architect",    "statistician")
    graph.add_edge("statistician", "visualizer")
    graph.add_edge("visualizer",   "summary")
    graph.add_edge("summary",      "insights")
    graph.add_edge("insights",     END)

    return graph.compile()