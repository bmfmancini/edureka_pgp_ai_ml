# Zero-shot / One-shot / Few-shot prompting for enterprise document generation

# ## Common Helper
import json
import requests
from typing import Dict, Any

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen:0.5b"


def ollama_generate(prompt: str, model: str = DEFAULT_MODEL, stream: bool = True, options: Dict[str, Any] = None) -> str:
    """
    Calls Ollama generate endpoint and returns the model's full response text.

    Parameters
    ----------
    prompt : str
        The prompt string you want to send to the LLM.
    model : str
        Ollama model name (example: "llama3.2:3b").
    stream : bool
        If True, Ollama returns newline-delimited JSON chunks (streaming).
    options : dict
        Optional Ollama generation options. Example:
        {"temperature": 0.2, "top_p": 0.9, "num_predict": 300}

    Returns
    -------
    str
        The combined response text from the model.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
    }
    if options:
        payload["options"] = options

    resp = requests.post(OLLAMA_URL, json=payload)
    resp.raise_for_status()

    # Streaming: each line is a JSON object: {"response": "...", "done": false/true, ...}
    if stream:
        full_text = []
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            full_text.append(data.get("response", ""))
            if data.get("done", False):
                break
        return "".join(full_text)

    # Non-stream: JSON response
    data = resp.json()
    return data.get("response", "")


# generating enterprise documents like executive summaries, research summaries, business proposals,
# legal briefs, market research using zero/one/few-shot prompting

# -----------------------------
# Zero-shot / One-shot / Few-shot Prompting
# (executive summary, research summary, proposals, legal briefs, market reports)
# -----------------------------

# One shared "context" we will reuse across prompts
BASE_CONTEXT = """
Context:
You are working at NexAI, an enterprise AI innovation company.
You must generate high-quality business/technical documents quickly.
Write clearly for the requested audience (technical or non-technical) and keep it structured.
"""

# We keep prompts in variables (dict), so you can switch by prompt_key.
PROMPTS_DEMO1 = {
    # --------------------------------
    # ZERO-SHOT PROMPTS (no examples given)
    # --------------------------------
    "zero_shot_exec_summary": f"""
{BASE_CONTEXT}

Task (Zero-Shot):
Generate an executive summary for a research paper on using AI algorithms to optimize recommendation systems in e-commerce.
Focus on: methodology, results, and potential applications.
Output format:
- Title
- 3-5 bullet overview
- Methodology (4-6 bullets)
- Results (3-5 bullets with plausible metrics)
- Business applications (3-5 bullets)
""",

    "zero_shot_inventory_impact": f"""
{BASE_CONTEXT}

Task (Zero-Shot):
Describe the impact of AI-driven automation on inventory management in e-commerce.
Highlight:
1) efficiency improvements
2) operational cost reductions
Output format:
- Short intro paragraph
- Numbered list of key impacts (5-7 items)
- Short conclusion paragraph
""",

    # --------------------------------
    # ONE-SHOT PROMPTS (single example provided)
    # --------------------------------
    "one_shot_legal_brief": f"""
{BASE_CONTEXT}

Task (One-Shot):
You will be given ONE example legal-case summary style.
Follow the same style to write a new short legal brief.

Example (style reference):
"In the case of XYZ v. ABC, the plaintiff alleges that the defendant used an AI algorithm that infringed on a patented recommendation engine. 
The dispute focuses on whether the implementation replicates the patented feature set and whether prior art applies."

Now write a similar brief summary for a patent dispute involving AI-based predictive analytics used to forecast user behavior.
Output format:
- Case-style opening (1-2 sentences)
- Allegation (1-2 sentences)
- What the dispute centers on (1 sentence)
""",

    "one_shot_business_proposal": f"""
{BASE_CONTEXT}

Task (One-Shot):
You will be given ONE example proposal snippet style. Mimic the same style and structure.

Example (style reference):
"Proposal: AI-Enhanced Fraud Detection Platform
We propose an AI platform that analyzes purchasing patterns, device fingerprints, and transaction velocity to flag suspicious behavior in real time.
The system will integrate with payment gateways, maintain an explainability layer for auditors, and reduce chargeback losses."

Now create a similar business proposal for an AI-powered platform that automates supply chain optimization.
Must include:
- Problem statement
- Proposed solution
- Key components (bullets)
- Expected outcomes (bullets with plausible % improvements)
""",

    # --------------------------------
    # FEW-SHOT PROMPTS (multiple examples provided)
    # --------------------------------
    "few_shot_business_proposal_reco_engine": f"""
{BASE_CONTEXT}

Task (Few-Shot):
You will be given TWO examples. Learn the style and produce the requested business proposal.

Example 1:
"The business proposal focuses on using machine learning to automate inventory management in retail,
offering real-time tracking and demand prediction."

Example 2:
"The proposal discusses the development of an AI-powered platform for financial risk assessment,
using predictive analytics to forecast market trends."

Now, write a business proposal for an AI-powered recommendation engine for an e-commerce platform,
detailing:
- Technology stack
- Market opportunity
- Expected outcomes

Output format:
1) Title
2) Problem (short)
3) Solution overview (short)
4) Technology stack (bullets: frontend, backend, ML, data, deployment)
5) Market opportunity (bullets)
6) Expected outcomes (bullets with plausible metrics)
""",

    "few_shot_research_summary_supply_chain": f"""
{BASE_CONTEXT}

Task (Few-Shot):
You will be given TWO example research-summary styles. Follow the same style and detail level.

Example 1:
"The research paper presents an algorithm that improves personalized recommendations by analyzing browsing history
and purchase patterns in e-commerce."

Example 2:
"In the research on AI in finance, an algorithm was developed to detect fraudulent transactions by analyzing patterns
in credit card activity."

Now, summarize a research paper on AI applications in supply chain management, focusing on:
- logistics optimization
- inventory forecasting

Output format:
- 1 paragraph summary (6-8 lines)
- Key contributions (3-5 bullets)
- Practical impact (2-4 bullets)
""",
}


def run_demo1(prompt_key: str, model: str = DEFAULT_MODEL) -> str:
    """
    Runs a selected Demo 1 prompt by key.
    """
    if prompt_key not in PROMPTS_DEMO1:
        raise KeyError(f"Unknown prompt_key: {prompt_key}. Available keys: {list(PROMPTS_DEMO1.keys())}")
    prompt = PROMPTS_DEMO1[prompt_key]
    return ollama_generate(prompt=prompt, model=model, stream=True, options={"temperature": 0.4})


# Prompts for Diverse NLP Tasks

# Tasks: summarization, code generation, question answering, translation, classification,
# extraction

# -----------------------------
# Creative prompts for diverse NLP tasks
# Tasks: summarization, code generation, QA, translation, classification, extraction
# -----------------------------

PROMPTS_DEMO2 = {
    # 1) Text Summarization
    "summarize_ai_across_industries": """
Task: Text Summarization

Input Text:
Artificial Intelligence (AI) has become a transformative technology in numerous industries.
Its applications range from healthcare, where AI is used to improve diagnostics and personalize patient care,
to finance, where it helps in fraud detection and automated trading.
As AI continues to evolve, it is poised to revolutionize traditional industries, creating smarter, more efficient systems.

Instruction:
Summarize the text focusing on the key points and applications of AI across industries.
Output format:
- 1 short paragraph
- 3 bullet key takeaways
""",

    # 2) Code Generation
    "code_gen_sum_even_numbers": """
Task: Code Generation

Instruction:
Write a Python function that takes a list of integers and returns the sum of all even numbers.

Constraints:
- Handle empty list safely
- Ignore non-integer values if they appear (but mention this in docstring)
- Provide a small example usage

Output only Python code.
""",

    # 3) Question Answering
    "qa_climate_change_causes": """
Task: Question Answering (based on provided context)

Context:
The global climate change crisis is primarily driven by human activities, including the burning of fossil fuels
which increases the concentration of greenhouse gases in the atmosphere.
This leads to rising temperatures, more severe weather patterns, and disruptions in ecosystems.

Question:
What are the primary causes of climate change?

Answer rules:
- Answer in 4-6 lines
- Use simple words
- Only use the given context (no external facts)
""",

    # 4) Machine Translation
    "translate_en_to_zh": """
Task: Machine Translation

Input:
Artificial intelligence is revolutionizing the healthcare industry.

Instruction:
Translate the sentence from English to Chinese.
Output only the translated sentence (no explanation).
""",

    # 5) Text Classification (classification task)
    "classify_feedback_type": """
Task: Text Classification (feedback type)

Input:
"The new software update has greatly improved the user experience. The interface is smoother, and performance is significantly faster."

Instruction:
Classify the text into ONE category:
1) Bug Report
2) Feature Request
3) Praise / Positive Feedback

Output format:
Category: <one category>
Reason: <1-2 lines>
""",

    # 6) Data Extraction
    "extract_person_entities": """
Task: Information Extraction

Input:
"John Smith, born on April 15, 1985, works as a software engineer at InnovateTech,
located at 1234 Tech Lane, San Francisco, CA."

Instruction:
Extract:
- person_name
- date_of_birth
- job_title
- company
- address

Return STRICT JSON only (no markdown).
Example:
{"person_name":"...","date_of_birth":"...","job_title":"...","company":"...","address":"..."}
""",
}


def run_demo2(prompt_key: str, model: str = DEFAULT_MODEL) -> str:
    """
    Runs a selected Demo 2 prompt by key.
    """
    if prompt_key not in PROMPTS_DEMO2:
        raise KeyError(f"Unknown prompt_key: {prompt_key}. Available keys: {list(PROMPTS_DEMO2.keys())}")
    prompt = PROMPTS_DEMO2[prompt_key]
    return ollama_generate(prompt=prompt, model=model, stream=True, options={"temperature": 0.2})


# Iterative Prompt Debugging (Incident Analysis)

# This directly follows the storyline: incident report → initial vague prompt → stepwise refinement:
#
# sentence classification into categories
#
# extraction of key fields
#
# one-sentence Slack summary
#
# translate resolution step into Spanish

# -----------------------------
# Iterative prompt debugging & refinement for incident analysis
# (TechStream CI/CD incident)
# -----------------------------

INCIDENT_REPORT = """
Incident Report:
During a routine deployment on June 25, 2025, the DevOps team encountered a critical failure in the CI/CD pipeline affecting the production environment.
A misconfigured YAML file led to skipped integration tests, which caused undetected bugs to be pushed live.
As a result, the user authentication service returned 503 errors for approximately 45 minutes, impacting around 30% of active sessions.
The issue was identified by the SRE team, rolled back, and a patch was applied.
A post-mortem report was initiated to outline the root cause and preventive measures.
""".strip()

# Categories exactly:
CATEGORIES = "deployment, error handling, impact, resolution, preventive action"

PROMPTS_DEMO3 = {
    # 0) Vague / poorly structured prompt (shows why it can be messy)
    "step0_vague": f"""
Classify, extract, summarize, translate the report.

{INCIDENT_REPORT}
""",

    # 1) Refine to classification only
    "step1_classification_only": f"""
Task: Sentence Classification

{INCIDENT_REPORT}

Instruction:
Classify EACH sentence into exactly one of these categories:
{CATEGORIES}

Output format (strict):
1) "<sentence 1>" -> <category>
2) "<sentence 2>" -> <category>
...
""",

    # 2) Add extraction
    "step2_classify_and_extract": f"""
Task: Classification + Key Extraction

{INCIDENT_REPORT}

Instructions:
A) Classify EACH sentence into one of: {CATEGORIES}
B) Extract these fields from the report:
- affected_service
- error_code
- duration
- team_responsible_for_resolution

Output format:
1) Sentence Classification:
- "<sentence>" -> <category>
2) Extracted Fields (as JSON):
{{"affected_service":"...","error_code":"...","duration":"...","team_responsible_for_resolution":"..."}}
""",

    # 3) Add Slack one-liner summary
    "step3_add_slack_summary": f"""
Task: Classification + Extraction + Slack Summary

{INCIDENT_REPORT}

Instructions:
1) Classify sentences into: {CATEGORIES}
2) Extract fields:
   - affected_service
   - error_code
   - duration
   - team_responsible_for_resolution
3) Write ONE concise sentence suitable for a Slack incident update.

Output format:
A) Sentence Classification: ...
B) Extracted Fields JSON: ...
C) Slack Summary: "<one sentence>"
""",

    # 4) Add translation of resolution step into Spanish
    "step4_add_translation": f"""
Task: Full Incident Processing

{INCIDENT_REPORT}

Instructions:
1) Classify sentences into: {CATEGORIES}
2) Extract fields (JSON):
   - affected_service
   - error_code
   - duration
   - team_responsible_for_resolution
3) Slack Summary: ONE concise sentence
4) Translation:
   Translate exactly this phrase into Spanish: "A patch was applied"

Output format:
A) Sentence Classification: ...
B) Extracted Fields JSON: ...
C) Slack Summary: "..."
D) Spanish Translation: "..."
""",
}


def run_demo3(prompt_key: str, model: str = DEFAULT_MODEL) -> str:
    """
    Runs a selected Demo 3 step prompt by key.
    """
    if prompt_key not in PROMPTS_DEMO3:
        raise KeyError(f"Unknown prompt_key: {prompt_key}. Available keys: {list(PROMPTS_DEMO3.keys())}")
    prompt = PROMPTS_DEMO3[prompt_key]
    return ollama_generate(prompt=prompt, model=model, stream=True, options={"temperature": 0.1})


if __name__ == "__main__":
    # ---- Example run (change prompt_key to test different prompt types)
    output = run_demo1("few_shot_business_proposal_reco_engine")
    print(output)

    # ---- Example run (change prompt_key to test different tasks)
    output = run_demo2("classify_feedback_type")
    print(output)

    # ---- Example run: start vague, then refined steps
    print("==== STEP 0 (VAGUE) ====")
    print(run_demo3("step0_vague"))

    print("\n==== STEP 1 (CLASSIFICATION ONLY) ====")
    print(run_demo3("step1_classification_only"))

    print("\n==== STEP 4 (FULL) ====")
    print(run_demo3("step4_add_translation"))
