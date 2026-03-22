import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
from core.state import AnalysisState
from agents import architect, statistician, visualizer, insights


def build_graph():
    graph = StateGraph(AnalysisState)

    graph.add_node("architect", architect.run)
    graph.add_node("statistician", statistician.run)
    graph.add_node("visualizer", visualizer.run)
    graph.add_node("insights", insights.run)

    graph.set_entry_point("architect")
    graph.add_edge("architect", "statistician")
    graph.add_edge("statistician", "visualizer")
    graph.add_edge("visualizer", "insights")
    graph.add_edge("insights", END)

    return graph.compile()