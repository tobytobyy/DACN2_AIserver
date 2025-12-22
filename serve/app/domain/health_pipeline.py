from typing import Dict, List

from PIL import Image

from app.core.config import settings
from app.domain.vqa_questions import GENERIC, MEDICINE, MED_REPORT, WOUND, VQAQuestion
from app.infra.blip_vqa import BlipVQA


def select_questions(clip_best_label: str) -> List[str]:
    # 1) Generic questions always included
    pool: List[VQAQuestion] = list(GENERIC)

    # 2) add domain-specific questions
    if clip_best_label == "medicine pills or blister pack":
        pool += MEDICINE
    elif clip_best_label == "medical report document":
        pool += MED_REPORT
    else:
        pool += WOUND

    # 3) sort by priority and pick top N <= VQA_MAX_QUESTIONS
    pool_sorted = sorted(pool, key=lambda q: q.priority)
    selected = pool_sorted[: settings.VQA_MAX_QUESTIONS]

    # 4) return texts only
    return [q.text for q in selected]


def yn_normalize(s: str | None) -> str | None:
    if not s:
        return None
    t = s.strip().lower()
    if t in {"yes", "yeah", "y", "true"}:
        return "Yes"
    if t in {"no", "n", "false"}:
        return "No"
    return s.strip()


def build_structured_context(answers: dict[str, str]) -> str:
    body_part = answers.get("What body part is this?")
    bleeding = yn_normalize(answers.get("Is it bleeding?"))
    depth = answers.get("Is it a deep wound or a superficial scratch?")
    redness = answers.get("Is there redness or swelling?")

    lines = []
    if body_part:
        lines.append(f"User sent an image of a {body_part}.")
    if depth:
        lines.append(f"It looks like: {depth}.")
    if bleeding:
        lines.append(f"Bleeding: {bleeding}.")
    if redness:
        lines.append(f"Redness/swelling: {redness}.")

    if lines:
        return " ".join(lines)

    shown = answers.get("What is shown in the image?")
    if shown:
        return f"Image shows: {shown}."
    return "Image content unclear."


class HealthPipeline:
    def __init__(self):
        self.vqa = BlipVQA()

    def analyze(self, image: Image.Image, clip_best_label: str) -> Dict:
        # choosing questions based on CLIP's best guess
        questions = select_questions(clip_best_label)

        answers = self.vqa.ask_many(image, questions)
        context = build_structured_context(answers)

        return {
            "vqa_answers": answers,
            "structured_context": context,
        }
