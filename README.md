SafeRoad Multi-Agent Assistant (SMAA)
Problem Statement

Road accidents caused by potholes, speed breakers, stray animals, and sudden vehicle or pedestrian appearances are a major concern. Drivers often lack real-time awareness of these risks.
This project addresses the problem by building a multi-agent AI system that detects road hazards in real time using webcam, uploaded images, or video streams, and provides instant voice and text alerts.

Why AI Multi-Agent System.

A single model cannot handle all responsibilities such as detection, reasoning, risk scoring, and communication.

Multi-agent collaboration allows specialized agents (Vision, Risk, LLM, TTS) to work independently but coordinate through an orchestrator, making the system robust and scalable.

Project Description

We developed SafeRoad Multi-Agent Assistant (SMAA), a road safety application powered by Vision, Risk, LLM, and TTS agents.

Workflow:

Vision Agent

Detects potholes, speed breakers, vehicles, humans, and animals.

Uses a custom YOLOv8 model (best.pt) for road damages and yolov8s.pt for vehicles and humans.

Risk Agent

Evaluates detections and classifies risk as Low, Medium, or High based on proximity and type.

LLM Agent

Generates natural language alerts (example: “High risk: Human crossing detected ahead”).

Supports multiple modes: Offline, OpenAI, and Gemini.

TTS Agent

Converts alerts into real-time voice warnings for drivers.

Orchestrator

Manages collaboration between agents, ensuring smooth real-time operation.


Tools, Libraries, and Frameworks Used

Programming Language: Python

Core Frameworks: Streamlit, Streamlit-WebRTC, OpenCV, NumPy

Models:

best.pt → Custom-trained YOLOv8 model (potholes, speed breakers)

yolov8s.pt → Pretrained YOLOv8 (vehicles, humans, animals)

AI Agents: Custom-built Vision, Risk, LLM, TTS agents

Optional Agent Orchestration: LangChain, CrewAI, AutoGen

Speech: pyttsx3 or gTTS for Text-to-Speech

LLM Selection

Best Choice (Premium): GPT-4, for high accuracy and reliable reasoning.

Other Strong Options: Claude 3, Gemini 1.5 Pro.

Free-tier Options Used:

OpenAI GPT-3.5 (via platform.openai.com)

Google Gemini (via Google Cloud)

Open-source models from Hugging Face or Mistral
For real-time road safety alerts, latency and cost are critical. GPT-4 and Gemini Pro provide strong reasoning capabilities, but for free-tier access GPT-3.5 and Gemini were chosen. Offline fallback ensures the app works even without internet connectivity.
