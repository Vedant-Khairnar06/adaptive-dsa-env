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


class Reward(BaseModel):
    value: float


# =========================
# ENVIRONMENT
# =========================
class DSAEnv:

    def __init__(self, max_steps=5):
        self.score = 0
        self.difficulty = "easy"
        self.current_question = None
        self.current_task = "basic"
        self.step_count = 0
        self.max_steps = max_steps

        with open("questions.json") as f:
            self.questions = json.load(f)

    # =========================
    # QUESTION SAMPLING
    # =========================
    def get_question(self):
        self.current_task = random.choice(["basic", "debug", "optimize"])

        filtered = [
            q for q in self.questions
            if q["difficulty"] == self.difficulty and q["task"] == self.current_task
        ]

        if not filtered:
            filtered = [
                q for q in self.questions
                if q["difficulty"] == self.difficulty
            ]

        return random.choice(filtered)

    # =========================
    # RESET
    # =========================
    def reset(self):
        self.score = 0
        self.difficulty = "easy"
        self.step_count = 0
        self.current_question = self.get_question()

        return Observation(
            question=self.current_question["question"],
            difficulty=self.difficulty,
            score=self.score,
            task=self.current_task
        )

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
    # CLEAN TEXT
    # =========================
    def clean(self, text):
        return re.sub(r'[^a-z0-9 ]', '', text.lower())

    # =========================
    # STEP FUNCTION (FIXED)
    # =========================
    def step(self, action):

        self.step_count += 1
        error = None

        try:
            action_obj = Action(answer=action)
            q = self.current_question

            ans_clean = self.clean(action_obj.answer)
            correct_clean = self.clean(q["answer"])

            correct_words = set(correct_clean.split())
            answer_words = set(ans_clean.split())

            common_words = correct_words.intersection(answer_words)

            # =========================
            # BASE CONTINUOUS REWARD
            # =========================
            if len(correct_words) == 0:
                ratio = 0.0
            else:
                ratio = len(common_words) / len(correct_words)

            # smooth reward strictly between (0,1)
            base_reward = max(0.1, min(0.9, ratio))

            # =========================
            # TASK-SPECIFIC GRADERS ✅
            # =========================

            # BASIC → direct QA
            if self.current_task == "basic":
                reward = base_reward

            # DEBUG → needs reasoning keywords
            elif self.current_task == "debug":
                keyword_bonus = any(word in ans_clean for word in ["error", "fix", "bug", "correct"])
                reward = base_reward * 0.7 + (0.2 if keyword_bonus else 0)

            # OPTIMIZE → needs efficiency awareness
            elif self.current_task == "optimize":
                keyword_bonus = any(word in ans_clean for word in ["optimize", "efficient", "log n", "better"])
                reward = base_reward * 0.6 + (0.25 if keyword_bonus else 0)

            # clamp again to (0,1)
            reward = max(0.1, min(0.9, reward))

            # =========================
            # PENALTY
            # =========================
            if len(action_obj.answer.strip()) == 0:
                reward = 0.1

            # =========================
            # DIFFICULTY UPDATE (FIXED)
            # =========================
            if reward > 0.7:
                self.difficulty = "hard"
            elif reward > 0.4:
                self.difficulty = "medium"
            else:
                self.difficulty = "easy"

            self.score += reward

            # next question
            self.current_question = self.get_question()

        except Exception as e:
            reward = 0.1
            error = str(e)

        # =========================
        # DONE LOGIC
        # =========================
        done = False

        if self.step_count >= self.max_steps:
            done = True

        if self.score >= 3.0:
            done = True

        # =========================
        # OBSERVATION
        # =========================
        observation = Observation(
            question=self.current_question["question"],
            difficulty=self.difficulty,
            score=self.score,
            task=self.current_task
        )

        return observation, reward, done, {"error": error}
