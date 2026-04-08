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
        self.current_task = ["basic", "debug", "optimize"][self.step_count % 3]

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
    # STEP FUNCTION (FINAL FIX)
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
            # ROBUST RATIO (NO COLLAPSE)
            # =========================
            if len(correct_words) == 0:
                ratio = 0.2
            else:
                ratio = len(common_words) / len(correct_words)

            # prevent extreme collapse
            ratio = max(0.2, min(0.8, ratio))

            # =========================
            # TASK-SPECIFIC GRADERS
            # =========================
            if self.current_task == "basic":
                reward = ratio

            elif self.current_task == "debug":
                bonus = 0.1 if any(w in ans_clean for w in ["error", "fix", "bug", "correct"]) else 0
                reward = ratio + bonus

            elif self.current_task == "optimize":
                bonus = 0.15 if any(w in ans_clean for w in ["optimize", "efficient", "log n", "better"]) else 0
                reward = ratio + bonus

            # =========================
            # FINAL CLAMP (STRICT RANGE)
            # =========================
            reward = max(0.2, min(0.8, reward))

            # =========================
            # EMPTY ANSWER PENALTY
            # =========================
            if len(action_obj.answer.strip()) == 0:
                reward = 0.2

            # =========================
            # DIFFICULTY UPDATE
            # =========================
            if reward > 0.65:
                self.difficulty = "hard"
            elif reward > 0.4:
                self.difficulty = "medium"
            else:
                self.difficulty = "easy"

            self.score += reward

            # next question
            self.current_question = self.get_question()

        except Exception as e:
            reward = 0.2
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
