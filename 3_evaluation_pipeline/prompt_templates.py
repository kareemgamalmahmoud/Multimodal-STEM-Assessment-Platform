"""
Prompt templates for all four experiments.

Each template is a Python function that returns a formatted string.
Keeping all prompts in one place makes them easy to compare and audit.

Naming convention:
    prompt_<experiment>_<sub_task>()

All prompts instruct the model to output structured JSON for reliable parsing.
"""


# ---------------------------------------------------------------------------
# Experiment 1: Baseline — direct scoring
# ---------------------------------------------------------------------------

def prompt_baseline_score(
    question: str,
    reference_answer: str,
    language_hint: str = "any",
) -> str:
    """
    One-shot direct scoring. No explicit reasoning requested.
    Serves as a minimal baseline.
    """
    return f"""You are a STEM teacher grading a student's handwritten answer.

Question: {question}

Reference Answer: {reference_answer}

Look at the handwritten student answer in the image and assign an overall score.

Respond ONLY with valid JSON in exactly this format:
{{
  "total_score": <float between 0.0 and 1.0>,
  "brief_justification": "<one sentence>"
}}"""


# ---------------------------------------------------------------------------
# Experiment 2: Chain-of-Thought scoring
# ---------------------------------------------------------------------------

def prompt_cot_transcribe_and_score(
    question: str,
    reference_answer: str,
) -> str:
    """
    Chain-of-thought prompt: transcribe → reason step by step → score.
    Grounded in Wei et al. 2022 'Chain-of-Thought Prompting'.
    """
    return f"""You are an expert STEM evaluator. Follow these steps carefully.

STEP 1 — TRANSCRIBE: Write out the exact content of the handwritten answer in the image.
STEP 2 — ANALYZE: Compare the student's answer to the reference answer step by step.
  - Identify correct concepts, correct computations, and any errors.
STEP 3 — SCORE: Assign an overall score from 0.0 to 1.0.

Question: {question}
Reference Answer: {reference_answer}

Respond ONLY with valid JSON in exactly this format:
{{
  "transcription": "<exact text from the image>",
  "step_by_step_analysis": "<your reasoning>",
  "errors_identified": ["<error 1>", "<error 2>"],
  "correct_elements": ["<correct 1>"],
  "total_score": <float 0.0–1.0>
}}"""


# ---------------------------------------------------------------------------
# Experiment 3: Rubric-decomposed scoring
# ---------------------------------------------------------------------------

def prompt_generate_rubric(question: str, reference_answer: str) -> str:
    """
    Ask the model to generate a marking rubric from the question + reference answer.
    Inspired by the analytic scoring approach in Yavuz et al. 2024.
    """
    return f"""You are an expert STEM curriculum designer. Create a grading rubric for the following question.

Question: {question}
Reference Answer: {reference_answer}

Generate a rubric with 3–5 criteria. Each criterion should be independently assessable.
The weights must sum to 1.0.

Respond ONLY with valid JSON:
{{
  "criteria": [
    {{
      "name": "<criterion name>",
      "description": "<what to check>",
      "weight": <float, e.g. 0.3>
    }}
  ]
}}"""


def prompt_rubric_score_criterion(
    question: str,
    reference_answer: str,
    student_transcript: str,
    criterion_name: str,
    criterion_description: str,
) -> str:
    """
    Score one rubric criterion independently.
    Each criterion gets its own model call for focused evaluation.
    """
    return f"""You are a STEM teacher evaluating one specific aspect of a student's answer.

Question: {question}
Reference Answer: {reference_answer}

Student's Answer (transcribed from handwriting):
{student_transcript}

Criterion to evaluate: {criterion_name}
What to check: {criterion_description}

Score this criterion only (0.0 = completely wrong, 1.0 = completely correct).

Respond ONLY with valid JSON:
{{
  "criterion": "{criterion_name}",
  "score": <float 0.0–1.0>,
  "justification": "<brief reason>"
}}"""


# ---------------------------------------------------------------------------
# Experiment 4 (ASTRA): Script-aware rubric scoring with self-consistency
# ---------------------------------------------------------------------------

def prompt_astra_transcribe(language_track: str) -> str:
    """
    ASTRA Step 1: Script-aware transcription prompt.
    The language_track routing ensures the model knows what to expect.
    """
    lang_instructions = {
        "arabic": (
            "The handwriting is in ARABIC script. Preserve Arabic text exactly.\n"
            "Mathematical expressions should be converted to LaTeX wrapped in $$ ... $$."
        ),
        "english": (
            "The handwriting is in ENGLISH. Transcribe all text and convert math to LaTeX wrapped in $$ ... $$."
        ),
        "mixed": (
            "The handwriting contains BOTH Arabic and English. "
            "Preserve Arabic in Arabic script. Convert math to LaTeX wrapped in $$ ... $$."
        ),
    }
    lang_note = lang_instructions.get(language_track, lang_instructions["english"])

    return f"""You are an expert OCR system specializing in handwritten STEM content.

{lang_note}

Transcribe the COMPLETE content of the handwritten image below. Output ONLY the transcribed text — no explanations."""


def prompt_astra_generate_rubric(question: str, reference_answer: str, language_track: str) -> str:
    """
    ASTRA Step 2: Generate a rubric with language-conditioned instructions.
    """
    lang_note = (
        "Note: Student answers may be written in Arabic. Evaluate meaning, not language form."
        if language_track in ("arabic", "mixed") else ""
    )

    return f"""You are an expert STEM evaluator. Create a detailed scoring rubric.

Question: {question}
Reference Answer: {reference_answer}
{lang_note}

Generate a rubric with exactly 4 criteria covering:
  1. Conceptual correctness — does the student understand the core concept?
  2. Mathematical accuracy — are computations and formulas correct?
  3. Completeness — is the answer fully developed (no missing steps)?
  4. Scientific notation / units — are symbols, units, and notation used correctly?

Weights must sum to 1.0. Use values: [0.35, 0.30, 0.20, 0.15] unless one criterion is irrelevant (then set its weight to 0 and redistribute).

Respond ONLY with valid JSON:
{{
  "criteria": [
    {{"name": "...", "description": "...", "weight": 0.35}},
    {{"name": "...", "description": "...", "weight": 0.30}},
    {{"name": "...", "description": "...", "weight": 0.20}},
    {{"name": "...", "description": "...", "weight": 0.15}}
  ]
}}"""


def prompt_astra_score_all_criteria(
    question: str,
    reference_answer: str,
    student_transcript: str,
    rubric: dict,
    language_track: str,
) -> str:
    """
    ASTRA Step 3 (inner): Score all rubric criteria in one call.
    Called N=5 times at temperature=0.7 for self-consistency voting.

    This is the core prompt used in the ASTRA voting loop.
    """
    lang_note = (
        "The student wrote in Arabic. Evaluate mathematical content and understanding; "
        "do not penalize for language/grammatical differences."
        if language_track in ("arabic", "mixed") else ""
    )

    criteria_text = "\n".join(
        f"  {i+1}. {c['name']} (weight={c['weight']}): {c['description']}"
        for i, c in enumerate(rubric.get("criteria", []))
    )

    return f"""You are an expert STEM examiner. Score the student's answer against each rubric criterion.

Question: {question}
Reference Answer: {reference_answer}

Student's Transcribed Answer:
{student_transcript}

{lang_note}

Rubric:
{criteria_text}

Score each criterion from 0.0 (completely wrong) to 1.0 (perfect).
Compute the weighted total score.

Respond ONLY with valid JSON:
{{
  "criterion_scores": [
    {{"name": "<criterion name>", "score": <float 0.0–1.0>, "justification": "<brief reason>"}},
    ...
  ],
  "total_score": <float 0.0–1.0, weighted average>
}}"""
