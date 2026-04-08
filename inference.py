import os
from openai import OpenAI
from env import DSAEnv

# =========================
# ENV VARIABLES (REQUIRED)
# =========================
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# =========================
# OPENAI CLIENT
# =========================
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

def get_action(question):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are a DSA solver. Provide only the final answer/code in one line. No explanations, no comments, no formatting. Just the answer."
            },
            {
                "role": "user",
                "content": f"Solve: {question}"
            }
        ]
    )
    return response.choices[0].message.content.strip().replace("\n", " ")

# =========================
# MAIN LOOP
# =========================
def main():
    env = DSAEnv()
    task = "dsa"
    env_name = "dsa_env"

    print(f"[START] task={task} env={env_name} model={MODEL_NAME}")

    rewards = []
    step = 0
    success = False

    try:
        q = env.reset()
        done = False

        while not done and step < 10:
            step += 1
            error = "null"

            try:
                action = get_action(q.question)
                obs, reward, done, info = env.step(action)

                if info and info.get("error"):
                    error = str(info.get("error"))

                q = obs

            except Exception as e:
                action = "error_occured"
                reward = 0.0
                done = True
                error = str(e).replace(" ", "_")

            rewards.append(reward)

            print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error}")

        success = done

    except Exception:
        pass

    finally:
        rewards_str = ",".join([f"{r:.2f}" for r in rewards]) if rewards else "0.00"

        # normalize score to [0,1]
        score = min(max(sum(rewards) / len(rewards) if rewards else 0.0, 0.0), 1.0)

        print(f"[END] success={str(success).lower()} steps={step} score={score:.2f} rewards={rewards_str}")

if __name__ == "__main__":
    main()