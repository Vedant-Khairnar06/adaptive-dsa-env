import subprocess
import gradio as gr
from fastapi import FastAPI
from env import DSAEnv
import uvicorn

env = DSAEnv()

# =========================
# FASTAPI APP
# =========================
api = FastAPI()

@api.post("/reset")
def reset():
    obs = env.reset()
    return {
        "question": obs.question,
        "difficulty": obs.difficulty,
        "score": obs.score,
        "task": obs.task
    }

@api.post("/step")
def step(action: dict):
    answer = action.get("answer", "")
    obs, reward, done, info = env.step(answer)
    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done,
        "info": info
    }

# =========================
# GRADIO UI
# =========================
def run_inference():
    result = subprocess.run(
        ["python", "inference.py"],
        capture_output=True,
        text=True
    )
    return result.stdout if result.stdout else result.stderr

with gr.Blocks() as demo:
    gr.Markdown("# DSA RL Agent (Hackathon Submission)")
    btn = gr.Button("Run Inference")
    output = gr.Textbox(lines=20)

    btn.click(run_inference, outputs=output)

# 🔥 MOUNT GRADIO AT ROOT
app = gr.mount_gradio_app(api, demo, path="/")

# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)