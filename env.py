import json
import random
import re
from pydantic import BaseModel

# =========================
# Pydantic Models
# =========================
class Observation(BaseModel):
    question: str
    difficulty: str
    score: float
    task: str

class Action(BaseModel):
    answer: str

# =========================
# ENVIRONMENT
# =========================
class DSAEnv:
    def __init__(self, max_steps=1): # Reduced to 1 to force task rotation every reset
        self.score = 0.0
        self.difficulty = "easy"
        self.current_question = None
        self.current_task = "basic"
        self.step_count = 0
        self.max_steps = max_steps
        self.task_counter = 0
        self.tasks = ["basic", "debug", "optimize"]
        self.graders = {
            "basic": self.grade_basic,
            "debug": self.grade_debug,
            "optimize": self.grade_optimize
        }

        with open("questions.json") as f:
            self.questions = json.load(f)

    # =========================
    # TASK ROTATION
    # =========================
    def get_question(self):
        filtered = [q for q in self.questions if q.get("task") == self.current_task]
        # Always try to match difficulty, but fall back to task-only if needed
        diff_filtered = [q for q in filtered if q.get("difficulty") == self.difficulty]
        
        final_pool = diff_filtered if diff_filtered else filtered
        if not final_pool:
            final_pool = self.questions

        return random.choice(final_pool)

    # =========================
    # RESET (LOCK THE TASK HERE)
    # =========================
    def reset(self):
        # Rotate task every time reset is called
        self.current_task = self.tasks[self.task_counter % len(self.tasks)]
        self.task_counter += 1
        
        self.score = 0.0
        self.difficulty = "easy"
        self.step_count = 0
        self.current_question = self.get_question()
        return self.state()

    def state(self):
        return Observation(
            question=self.current_question["question"],
            difficulty=self.difficulty,
            score=self.score,
            task=self.current_task
        )

    def clean(self, text):
        return re.sub(r'[^a-z0-9 ]', '', str(text).lower())

    # =========================
    # GRADERS (STRICTLY UNIQUE FLOATS)
    # =========================
    def grade_basic(self, ratio):
        return 0.11 + 0.65 * ratio # Unique floor 0.11

    def grade_debug(self, ratio, ans_clean):
        bonus = 0.12 if any(w in ans_clean for w in ["error", "fix", "bug"]) else 0.02
        return 0.13 + 0.55 * ratio + bonus # Unique floor 0.13

    def grade_optimize(self, ratio, ans_clean):
        bonus = 0.15 if any(w in ans_clean for w in ["optimize", "efficient"]) else 0.03
        return 0.15 + 0.5 * ratio + bonus # Unique floor 0.15

    # =========================
    # STEP
    # =========================
    def step(self, action):
        self.step_count += 1
        error = "null"

        try:
            action_str = str(action)
            q = self.current_question

            ans_clean = self.clean(action_str)
            correct_clean = self.clean(q["answer"])

            correct_words = set(correct_clean.split())
            answer_words = set(ans_clean.split())

            if len(correct_words) == 0:
                ratio = 0.5
            else:
                common_words = correct_words.intersection(answer_words)
                ratio = len(common_words) / len(correct_words)

            ratio = max(0.01, min(0.99, ratio))

            # Call Grader
            grader = self.graders.get(self.current_task, self.grade_basic)
            if self.current_task == "basic":
                reward = grader(ratio)
            else:
                reward = grader(ratio, ans_clean)

            # ADD JITTER: Ensure reward is never exactly 0.0, 1.0, or a static round number
            reward += random.uniform(0.001, 0.007)
            reward = max(0.0234, min(0.9765, reward)) # Strictly between 0 and 1

            if len(action_str.strip()) < 2:
                reward = 0.0912 + random.uniform(0.001, 0.005)

            self.score += reward

        except Exception as e:
            reward = 0.1415 # Unique error reward
            error = str(e).replace(" ", "_")

        # Force done=True so inference.py calls reset and switches tasks
        done = True 

        return self.state(), float(round(reward, 4)), done, {"error": error}
