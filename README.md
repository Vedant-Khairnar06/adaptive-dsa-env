title: Adaptive DSA RL Environment (Docker)
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
Adaptive DSA Reinforcement Learning Environment
Overview
This project implements a reinforcement learning (RL) environment for evaluating an agent’s understanding of Data Structures and Algorithms (DSA).
It simulates real-world evaluation scenarios such as coding interviews and automated grading systems.

The environment presents questions, evaluates responses, and assigns rewards based on correctness and partial progress.

Objective
The agent interacts with the environment to:

Answer DSA-related questions
Receive reward-based feedback
Improve performance across multiple steps
Tasks
The environment includes three task categories:

basic (easy) — fundamental concepts
debug (medium) — conceptual reasoning
optimize (hard) — advanced problem-solving
Reinforcement Learning Components
Observation

question: str
difficulty: str
score: float
task: str
Action

answer: str
Reward

1.0 → correct
0.5 → partially correct
0.0 → incorrect
Reward Design
The reward function provides:

Incremental feedback based on keyword overlap
Partial scoring for incomplete answers
Penalty for empty responses
Environment API (OpenEnv)
The environment follows OpenEnv specification:

reset() → returns initial observation
step(action) → returns (observation, reward, done, info)
state() → returns current state
Inference
The inference script:

Uses OpenAI client
Reads environment variables (HF_TOKEN, MODEL_NAME, API_BASE_URL)
Executes interaction loop
Produces standardized output format
Output Format
[START] task=<task> env=<env> model=<model>
[STEP] step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> rewards=<r1,r2,...>
Baseline Performance
A baseline run using the default model produces:

Steps: 1–5
Rewards: typically between 0.0 and 1.0
Success: depends on model response quality
Due to API quota limits, sample runs may return partial results, but the environment executes correctly.

Baseline Results
Typical run produces rewards between 1.5–3.0 over 5 steps depending on model performance.

Setup & Usage
Run locally
pip install -r requirements.txt
python inference.py
Docker Execution
Build the container:

docker build -t dsa-env .
Run the container:

docker run -p 7860:7860 dsa-env
Use Cases
Evaluation of AI models on DSA reasoning
Interview preparation systems
Reinforcement learning experimentation
Automated grading systems
Conclusion
This project demonstrates a structured RL-based evaluation environment with adaptive difficulty, meaningful reward signals, and compatibility with automated evaluation pipelines.