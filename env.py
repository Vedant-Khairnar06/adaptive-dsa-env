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
    def __init__(self, max_steps=5):
        self.score = 0.0
        self.difficulty = "easy"
        self.current_question = None
        self.current_task = "basic"
        self.step_count = 0
        self.max_steps = max_steps
        self.task_counter = 0

        with open("questions.json") as f:
            self.questions = json.load(f)

    # =========================
    # TASK ROTATION (CRITICAL)
    # =========================
    def get_question(self):
        tasks = ["basic", "debug", "optimize"]
        self.current_task = tasks[self.task_counter % 3]
        self.task_counter += 1

        filtered = [
            q for q in self.questions
            if q.get("task") == self.current_task
        ]

        if not filtered:
            filtered = self.questions

        return random.choice(filtered)

    # =========================
    # RESET
    # =========================
    def reset(self):
        self.score = 0.0
        self.difficulty = "easy"
        self.step_count = 0
        self.current_question = self.get_question()
        return self.state()

    # =========================
    # STATE
    # =========================
    def state(self):
        return Observation(
            question=self.current_question["question"],
            difficulty=self.difficulty,
            score=self.score,
            task=self.current_task
        )

    # =========================
    # CLEAN
    # =========================
    def clean(self, text):
        return re.sub(r'[^a-z0-9 ]', '', str(text).lower())

    # =========================
    # GRADERS (EXPLICIT)
    # =========================
    def grade_basic(self, ratio):
        return 0.3 + 0.5 * ratio

    def grade_debug(self, ratio, ans_clean):
        bonus = 0.1 if any(w in ans_clean for w in ["error", "fix", "bug", "correct"]) else 0
        return 0.25 + 0.4 * ratio + bonus

    def grade_optimize(self, ratio, ans_clean):
        bonus = 0.15 if any(w in ans_clean for w in ["optimize", "efficient", "log n", "better"]) else 0
        return 0.2 + 0.5 * ratio + bonus

    # =========================
    # STEP
    # =========================
    def step(self, action):
        self.step_count += 1
        error = None

        try:
            action_str = str(action)
            q = self.current_question

            ans_clean = self.clean(action_str)
            correct_clean = self.clean(q["answer"])

            correct_words = set(correct_clean.split())
            answer_words = set(ans_clean.split())

            common_words = correct_words.intersection(answer_words)

            # =========================
            # COMPUTE RATIO
            # =========================
            if len(correct_words) == 0:
                ratio = 0.3
            else:
                ratio = len(common_words) / len(correct_words)

            ratio = max(0.1, min(0.9, ratio))

            # =========================
            # EXPLICIT GRADER CALL
            # =========================
            if self.current_task == "basic":
                reward = self.grade_basic(ratio)

            elif self.current_task == "debug":
                reward = self.grade_debug(ratio, ans_clean)

            elif self.current_task == "optimize":
                reward = self.grade_optimize(ratio, ans_clean)

            else:
                reward = 0.5

            # =========================
            # FINAL CLAMP
            # =========================
            reward = max(0.2, min(0.8, reward))

            # penalty for empty
            if len(action_str.strip()) < 2:
                reward = 0.25 + random.uniform(0.01, 0.05)

            # difficulty update
            if reward > 0.65:
                self.difficulty = "hard"
            elif reward > 0.4:
                self.difficulty = "medium"
            else:
                self.difficulty = "easy"

            self.score += reward

            # =========================
            # KEEP SAME TASK (CRITICAL FIX)
            # =========================
            filtered = [
                q for q in self.questions
                if q.get("task") == self.current_task
            ]
            if not filtered:
                filtered = self.questions

            self.current_question = random.choice(filtered)

        except Exception as e:
            reward = 0.3
            error = str(e)

        done = self.step_count >= self.max_steps or self.score >= 4.0

        return self.state(), round(reward, 4), done, {"error": error}
