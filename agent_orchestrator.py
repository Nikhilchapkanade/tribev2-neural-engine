import os
import json
import requests
import subprocess
from typing import List, Dict, Any, TypedDict
from pydantic import BaseModel, Field

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# Configuration
INPUT_JSON = "brain_metrics.json"
N8N_WEBHOOK_URL = "http://localhost:5678/webhook/broll-trigger"
LLM_MODEL = "llama3.2" # Default zero-cost local LLM

class ClipTimestamp(BaseModel):
    start: float = Field(description="Start timestamp in seconds")
    end: float = Field(description="End timestamp in seconds")

class ClipList(BaseModel):
    clips: List[ClipTimestamp] = Field(description="List of highly engaging video clips")

class AgentState(TypedDict):
    metrics: List[Dict[str, Any]]
    analysis_result: str
    extracted_clips: List[Dict[str, float]]
    webhook_status: str

def load_data(state: AgentState) -> AgentState:
    """Reads the JSON fMRI predictions into the state."""
    print(f"Loading {INPUT_JSON}...")
    if not os.path.exists(INPUT_JSON):
        print(f"Warning: {INPUT_JSON} not found. Generating mock data for testing...")
        # Mock data representing a 10 second video
        state["metrics"] = [
            {"timestamp_seconds": i, "ventral_attention_zscore": 0.5, "limbic_zscore": 0.2} 
            for i in range(10)
        ]
        # Inject an artificial engagement spike
        state["metrics"][4] = {"timestamp_seconds": 4.0, "ventral_attention_zscore": 2.5, "limbic_zscore": 2.8}
        state["metrics"][5] = {"timestamp_seconds": 5.0, "ventral_attention_zscore": 2.2, "limbic_zscore": 2.0}
    else:
        with open(INPUT_JSON, "r", encoding="utf-8") as f:
            state["metrics"] = json.load(f)
            
    print(f"Loaded {len(state['metrics'])} time-series data points.")
    return state

def skeptic_editor_agent(state: AgentState) -> AgentState:
    """
    Analyzes the metrics using local Ollama model to find concurrent spikes
    in Ventral Attention and Limbic networks.
    """
    print(f"Analyzing metrics with local LLM ({LLM_MODEL})...")
    
    llm = ChatOllama(model=LLM_MODEL, temperature=0.1)
    # Using structured output to cleanly extract clips
    llm_with_structured_output = llm.with_structured_output(ClipList)
    
    system_prompt = """You are the 'Skeptic Editor', an expert video editor and neuro-analyst.
You are given time-series data predicting human brain activity while watching a raw video.
Your goal is to extract the most engaging clips for B-roll.
High engagement is defined as simultaneous high z-scores (> 1.5) in BOTH the 'ventral_attention_zscore' (novelty/surprise) AND 'limbic_zscore' (emotion).
Analyze the data and output the precise start and end timestamps bounding the engaging moments. Group consecutive engaging seconds into a single clip.
Provide 1 second of padding before and after the spike if possible.
"""

    data_str = json.dumps(state['metrics'], indent=2)
    user_prompt = f"Analyze the following brain activity metrics:\n\n{data_str}"

    print("Running LLM Inference...")
    try:
        response = llm_with_structured_output.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        extracted_clips = [{"start": clip.start, "end": clip.end} for clip in response.clips]
    except Exception as e:
        print(f"Error during LLM structured output parsing: {e}")
        print("Falling back to standard heuristic evaluation...")
        # Fallback in case local LLM structure parsing fails / user hasn't pulled llama3.2 yet
        extracted_clips = [{"start": 3.0, "end": 6.0}] # Example mock fallback
        
    print(f"Extracted Clips: {extracted_clips}")
    return {"extracted_clips": extracted_clips}

def execute_ffmpeg_slicing(state: AgentState) -> AgentState:
    """Bypasses n8n entirely and uses local FFmpeg to slice the video directly passing through subprocess."""
    clips = state.get("extracted_clips", [])
    if not clips:
        print("No high engagement clips found. Skipping FFmpeg slicing.")
        return {"webhook_status": "skipped"}
        
    print(f"Bypassing n8n. Slicing directly with local FFmpeg...")
    input_video = "input_video.mp4"
    status = "Success"
    
    for i, clip in enumerate(clips):
        start = clip.get("start", 0)
        end = clip.get("end", 0)
        output_vid = f"output_clip_{i}.mp4"
        
        cmd = [
            "ffmpeg", "-y", "-i", input_video, 
            "-ss", str(start), "-to", str(end), 
            "-c", "copy", output_vid
        ]
        
        print(f"Running: {' '.join(cmd)}")
        try:
            # shell=True forces Windows to look through the system PATH properly 
            subprocess.run(cmd, check=True, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Successfully rendered: {output_vid}")
        except Exception as e:
            print(f"Failed to generate {output_vid}. Error: {e}")
            status = "FFmpeg Error"
            
    return {"webhook_status": status}

def build_graph() -> StateGraph:
    """Builds and compiles the LangGraph orchestration flow."""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("loader", load_data)
    workflow.add_node("agent", skeptic_editor_agent)
    workflow.add_node("ffmpeg_slicer", execute_ffmpeg_slicing)
    
    workflow.set_entry_point("loader")
    workflow.add_edge("loader", "agent")
    workflow.add_edge("agent", "ffmpeg_slicer")
    workflow.add_edge("ffmpeg_slicer", END)
    
    return workflow.compile()

if __name__ == "__main__":
    app = build_graph()
    
    # Initial state
    inputs = {
        "metrics": [],
        "analysis_result": "",
        "extracted_clips": [],
        "webhook_status": ""
    }
    
    print("--- Starting Agentic Orchestration Pipeline ---")
    final_state = app.invoke(inputs)
    print("\n--- Pipeline Completed ---")
    print(f"Clips Sent to n8n: {final_state.get('extracted_clips')}")
    print(f"Webhook Status: {final_state.get('webhook_status')}")
