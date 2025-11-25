"""
Notes-First Paper Aggregation (English prompts + progress & timing)

What it does
------------
Given a DISEASE, a PROTEIN, and ~10 papers (PMID/title/abstract/...),
1) extracts one structured "note" per paper (direction, evidence, quote, etc.),
2) aggregates polarity (positive/negative/unclear) with a confidence score,
3) compresses top supporting/contradicting notes into dense, citable conclusions (English).

Requirements
------------
- pip install openai
- export OPENAI_API_KEY=...

How to run
----------
Fill the `papers` list at the bottom and run the script.
"""

import json, math, time, datetime
from collections import defaultdict
from openai import OpenAI
import os

# ================== CONFIG ==================
OPENAI_MODEL_NOTES = "gpt-4.1-mini"      # for structured note extraction
OPENAI_MODEL_COMPRESS = "gpt-4.1-mini"   # for conclusions compression
PRINT_PREFIX = "[TEST]"

client = OpenAI(
    api_key="", # your openai key
)
def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def fmt_seconds(s):
    # human-friendly elapsed time
    m, sec = divmod(int(s), 60)
    h, m = divmod(m, 60)
    if h: return f"{h}h {m}m {sec}s"
    if m: return f"{m}m {sec}s"
    return f"{sec}s"

# ---------- A) Structured note extraction (EN prompts) ----------
NOTE_FUNC = {
    "type": "function",
    "function": {
        "name": "submit_note",
        "description": "Extract a structured evidence note about DISEASE–PROTEIN relation from a single paper.",
        "parameters": {
            "type": "object",
            "properties": {
                "pmid": {"type": "string"},
                "direction": {  # relation sign
                    "type": "string",
                    "enum": ["positive","negative","null","mixed","unclear"]
                },
                "study_type": {
                    "type": "string",
                    "description": "e.g., meta-analysis, systematic review, RCT, cohort, case-control, cross-sectional, GWAS, MR, animal, in_vitro, other"
                },
                "sample_size": {"type": "integer", "nullable": True},
                "context": {"type": "string", "description": "tissue/cell/stage/subtype/population"},
                "protein_dimension": {
                    "type": "string",
                    "description": "mutation/expression_up/expression_down/activity_up/activity_down/level_up/level_down/unspecified"
                },
                "effect_size": {"type": "string", "nullable": True},
                "p_value": {"type": "string", "nullable": True},
                "confounders_addressed": {"type": "boolean"},
                "quote_span": {"type": "string", "description": "verbatim clause supporting the direction"},
                "limitations": {"type": "string"},
                "rationale": {"type": "string", "description": "one-sentence reason for the chosen direction"}
            },
            "required": ["pmid","direction","study_type","rationale"]
        }
    }
}

SYSTEM_NOTE = (
    "You are extracting one structured evidence note about the relationship between a DISEASE and a PROTEIN "
    "from a single paper (title/abstract). Prefer human evidence over animal/in-vitro. "
    "Be conservative: if evidence is ambiguous or non-significant, return 'unclear' or 'null' and state limitations. "
    "Return ONLY via the provided function with a complete JSON object."
)

def llm_extract_note(disease, protein, paper, model=OPENAI_MODEL_NOTES):
    user_msg = (
        f"DISEASE: {disease}\nPROTEIN: {protein}\n"
        f"PMID: {paper.get('pmid')}\nTITLE: {paper.get('title')}\n"
        f"ABSTRACT: {paper.get('abstract')}\nYEAR: {paper.get('year')}, JOURNAL: {paper.get('journal')}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":SYSTEM_NOTE},
                  {"role":"user","content":user_msg}],
        tools=[NOTE_FUNC],
        tool_choice={"type":"function","function":{"name":"submit_note"}},
        temperature=0.0
    )
    call = resp.choices[0].message.tool_calls[0]
    note = json.loads(call.function.arguments)
    note.setdefault("pmid", str(paper.get("pmid","")))
    return note

# ---------- B) Evidence strength scoring & polarity aggregation ----------
STUDY_WEIGHT = {
    "meta-analysis": 1.00, "systematic_review": 0.95, "RCT": 0.90,
    "cohort": 0.80, "case-control": 0.75, "cross-sectional": 0.65,
    "GWAS": 0.85, "MR": 0.88,
    "human_other": 0.60, "animal": 0.45, "in_vitro": 0.35, "other": 0.40
}

def map_study_type(s):
    s = (s or "").lower()
    for k in ["meta"]:
        if k in s: return "meta-analysis"
    if "systematic" in s: return "systematic_review"
    for k in ["random","rct"]:
        if k in s: return "RCT"
    if "cohort" in s: return "cohort"
    if "case-control" in s or "case control" in s: return "case-control"
    if "cross-sectional" in s or "cross sectional" in s: return "cross-sectional"
    if "gwas" in s: return "GWAS"
    if "mendel" in s or " mr" in s: return "MR"
    if "animal" in s or "mouse" in s or "mice" in s: return "animal"
    if "vitro" in s or "cell" in s: return "in_vitro"
    if "human" in s: return "human_other"
    return "other"

def strength_score(note):
    # study type
    w_study = STUDY_WEIGHT.get(map_study_type(note.get("study_type")), 0.40)
    # sample size (log scaling)
    n = note.get("sample_size") or 0
    w_n = min(1.0, math.log(max(n,1)+1, 10)/1.2)  # ~0.8 around n≈5k
    # p-value buckets
    p = (note.get("p_value") or "").lower()
    if "<" in p:
        try:
            pv = float(p.split("<")[1].strip().replace("=", ""))
            w_p = 1.0 if pv < 1e-5 else 0.9 if pv < 1e-3 else 0.75 if pv < 0.01 else 0.6 if pv < 0.05 else 0.4
        except:
            w_p = 0.55
    elif "ns" in p or "not significant" in p:
        w_p = 0.2
    else:
        w_p = 0.55  # unknown
    # confounders
    w_conf = 0.85 if note.get("confounders_addressed") else 0.65
    # final score in [0,1]
    s = max(0.0, min(1.0, 0.45*w_study + 0.25*w_n + 0.20*w_p + 0.10*w_conf))
    return s

def signed_direction(direction):
    d = (direction or "").lower()
    if d == "positive": return +1
    if d == "negative": return -1
    return 0  # null/mixed/unclear

def aggregate_polarity(notes):
    weights, signed_sum, pos_w, neg_w = 0.0, 0.0, 0.0, 0.0
    for n in notes:
        s = strength_score(n)
        sd = signed_direction(n.get("direction"))
        signed_sum += sd * s
        weights += s
        if sd > 0: pos_w += s
        if sd < 0: neg_w += s
    coverage = min(1.0, weights / max(1e-9, len(notes)))
    if weights == 0:
        return {"label":"unclear","score":0.0,"confidence":0.0,"coverage":0.0,"pos_w":0.0,"neg_w":0.0}
    raw = signed_sum / weights
    label = "positive" if raw > 0.1 else "negative" if raw < -0.1 else "unclear"
    confidence = 1/(1+math.exp(-6*abs(raw))) * (0.5 + 0.5*coverage)
    return {"label":label, "score":raw, "confidence":confidence, "coverage":coverage,
            "pos_w":pos_w, "neg_w":neg_w}

# ---------- C) Compress multiple notes into dense, citable conclusions (EN prompts) ----------
SYSTEM_COMPRESS = (
    "You are a rigorous biomedical reviewer. Given structured evidence notes about DISEASE–PROTEIN, "
    "write dense, value-added conclusions in ENGLISH with inline citations. "
    "STRICTLY ground every claim in the provided notes only; if evidence conflicts or is weak, say so explicitly."
)

def llm_compress_notes(disease, protein, notes, polarity, model=OPENAI_MODEL_COMPRESS):
    # Select Top-K support/contradict/neutral
    def key_s(n): return strength_score(n)
    supp = sorted([n for n in notes if signed_direction(n["direction"])>0], key=key_s, reverse=True)[:5]
    contra = sorted([n for n in notes if signed_direction(n["direction"])<0], key=key_s, reverse=True)[:5]
    neutral = sorted([n for n in notes if signed_direction(n["direction"])==0], key=key_s, reverse=True)[:3]

    def pack(n):
        return {
            "pmid": n.get("pmid"),
            "dir": n.get("direction"),
            "study": map_study_type(n.get("study_type")),
            "n": n.get("sample_size"),
            "ctx": n.get("context"),
            "prot_dim": n.get("protein_dimension"),
            "effect": n.get("effect_size"),
            "p": n.get("p_value"),
            "quote": n.get("quote_span"),
            "strength": round(strength_score(n),3)
        }

    payload = {
        "disease": disease, "protein": protein,
        "polarity": polarity,
        "support": [pack(x) for x in supp],
        "contradict": [pack(x) for x in contra],
        "neutral": [pack(x) for x in neutral],
    }

    user = (
        "Please produce English bullet-point conclusions with citations [PMID:xxxx] using ONLY the notes below (JSON):\n"
        "Include:\n"
        "1) Overall direction (positive/negative/unclear) + confidence + short rationale;\n"
        "2) Mechanistic signals (pathways, mutation types, up/downstream effects, expression/activity direction);\n"
        "3) Context-specific findings (tissues/cell types/disease stages/subpopulations);\n"
        "4) Causal evidence (RCT/MR/GWAS/intervention) and robustness (confounders, replication);\n"
        "5) Conflicts & uncertainties (where/why they arise) + what additional evidence would de-risk;\n"
        "6) A short 'verifiability checklist' with explicit [PMID] anchors.\n"
        f"NOTES JSON:\n{json.dumps(payload, ensure_ascii=False)}"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":SYSTEM_COMPRESS},
                  {"role":"user","content":user}],
        temperature=0.2
    )
    return resp.choices[0].message.content

# ---------- D) Orchestrator with progress & timing ----------
def run_pipeline(disease, protein, papers):
    t0 = time.perf_counter()
    print(f"{PRINT_PREFIX} {now_str()} | Start pipeline for DISEASE='{disease}' PROTEIN='{protein}'")
    print(f"{PRINT_PREFIX} Papers to process: {len(papers)}")

    # Step A: extract structured notes
    notes = []
    per_item_times = []
    tA0 = time.perf_counter()
    print(f"{PRINT_PREFIX} [A] Extracting structured notes ...")

    for i, p in enumerate(papers, start=1):
        item_start = time.perf_counter()
        pmid = p.get("pmid", f"idx{i}")
        print(f"{PRINT_PREFIX}   - ({i}/{len(papers)}) Extracting note for PMID={pmid} ...", end="", flush=True)
        try:
            if not p.get("abstract"):
                print(" SKIPPED (no abstract).")
                continue
            note = llm_extract_note(disease, protein, p)
            notes.append(note)
            dt = time.perf_counter() - item_start
            per_item_times.append(dt)
            # ETA estimate
            avg = sum(per_item_times)/len(per_item_times)
            remain = (len(papers) - i) * avg
            print(f" done in {fmt_seconds(dt)} | ETA ~ {fmt_seconds(remain)}")
        except Exception as e:
            dt = time.perf_counter() - item_start
            print(f" ERROR after {fmt_seconds(dt)}: {e}")

    tA = time.perf_counter() - tA0
    print(f"{PRINT_PREFIX} [A] Completed note extraction: {len(notes)} notes | {fmt_seconds(tA)} elapsed")

    # Step B: polarity aggregation
    tB0 = time.perf_counter()
    pol = aggregate_polarity(notes)
    label = pol["label"]
    conf = round(pol["confidence"], 3)
    score = round(pol["score"], 3)
    coverage = round(pol["coverage"], 3)
    tB = time.perf_counter() - tB0
    print(f"{PRINT_PREFIX} [B] Polarity: {label} | score={score} | confidence={conf} | coverage={coverage} "
          f"| computed in {fmt_seconds(tB)}")

    # Step C: compress conclusions
    tC0 = time.perf_counter()
    print(f"{PRINT_PREFIX} [C] Compressing notes into dense conclusions ...", end="", flush=True)
    conclusions = llm_compress_notes(disease, protein, notes, pol)
    tC = time.perf_counter() - tC0
    print(f" done in {fmt_seconds(tC)}")

    total = time.perf_counter() - t0
    print(f"{PRINT_PREFIX} {now_str()} | All done in {fmt_seconds(total)}")

    # Friendly summary line for downstream logs
    print(f"{PRINT_PREFIX} SUMMARY | direction={label} | score={score} | confidence={conf} | coverage={coverage} "
          f"| N_notes={len(notes)} | total_time={fmt_seconds(total)}")

    return {
        "direction_label": label,
        "direction_score": score,
        "confidence": conf,
        "coverage": coverage,
        "notes": notes,
        "conclusions": conclusions,
        "timing": {"extract": tA, "aggregate": tB, "compress": tC, "total": total}
    }


if __name__ == "__main__":
    Protein = "APOE"
    input_file = (f"/Users/joysw/PycharmProjects/pythonProgramming/ASearcher/SearchPaperByEmbedding-main/data4AD/AD_APOE_results10.json")
    Disease = "Alzheimer's disease"

    with open(input_file, 'r', encoding='utf-8') as reader:
        inputs = json.load(reader)
        papers = [paper["paper"] for paper in inputs["results"]]
        if not papers:
            print(f"{PRINT_PREFIX} No papers provided. Fill the 'papers' list in __main__ and re-run.")
        # papers = [
        #   {"pmid":"40385398","title":"...","abstract":"...","year":2025,"journal":"..."},
        #   ... # 共10篇
        # ]
        else:
            out = run_pipeline(Disease, Protein, papers)
            print("\n=== FINAL CONCLUSIONS ===\n")
            print(out["conclusions"])
            # print(out["direction_label"], out["confidence"])
            # print(out["conclusions"])