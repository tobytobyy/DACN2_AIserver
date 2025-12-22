from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class VQAQuestion:
    text: str
    priority: int


GENERIC: List[VQAQuestion] = [
    VQAQuestion("What is shown in the image?", 1),
    VQAQuestion("Is this related to health or medicine?", 2),
]

MEDICINE: List[VQAQuestion] = [
    VQAQuestion("Are there pills or a blister pack?", 1),
    VQAQuestion("What form is it (pill, capsule, tablet, syrup, ointment)?", 2),
    VQAQuestion("Is there a medicine label or text visible?", 3),
]

MED_REPORT: List[VQAQuestion] = [
    VQAQuestion("Is this a medical report or test result document?", 1),
    VQAQuestion("What type of document is it (lab test, prescription, report)?", 2),
    VQAQuestion("Are there any numbers or test values visible?", 3),
]

WOUND: List[VQAQuestion] = [
    VQAQuestion("What body part is this?", 1),
    VQAQuestion("Is it bleeding?", 2),
    VQAQuestion("Is it a deep wound or a superficial scratch?", 3),
    VQAQuestion("Is there redness or swelling?", 4),
]
