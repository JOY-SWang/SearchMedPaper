#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
crawlOpenTarget_fast.py — Faster fetch of literature from Open Targets GraphQL v4 with:
- Concurrent Europe PMC enrichment (resultType=core) using requests.Session + ThreadPoolExecutor
- PubMed EFetch batch backfill for missing abstracts and journal titles
- Incremental checkpointing every N records to JSON/JSONL (default 500)
- Robust GraphQL paging and defensive handling

Examples
  # Disease-wide (large count)
  python crawl.py --mode disease --disease MONDO_0004975 --out AD_ALL.json --jsonl AD_ALL.jsonl --enrich --pubmed-backfill

  # Disease × Gene intersection
  python crawl.py --mode evidence --ensembl ENSG00000128564 --disease MONDO_0005335 \
    --jsonl data/crawl/AD_VGF.jsonl --out data/crawl/AD_VGF.snapshot.json --enrich --pubmed-backfill \
    --max-workers 8 --save-every 500
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from xml.etree import ElementTree as ET

OT_GQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
EUROPE_PMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
PUBMED_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

DEFAULT_SAVE_EVERY = 500
DEFAULT_MAX_WORKERS = 12
DEFAULT_TIMEOUT = 30

# ---------------- Global sessions (connection reuse) ----------------
_ot_session = None
_epmc_session = None
_pubmed_session = None

def get_ot_session() -> requests.Session:
    global _ot_session
    if _ot_session is None:
        s = requests.Session()
        s.headers.update({"User-Agent": "Joy-OT-Fetcher/1.0"})
        _ot_session = s
    return _ot_session

def get_epmc_session() -> requests.Session:
    global _epmc_session
    if _epmc_session is None:
        s = requests.Session()
        s.headers.update({"User-Agent": "Joy-OT-Fetcher/1.0"})
        _epmc_session = s
    return _epmc_session

def get_pubmed_session() -> requests.Session:
    global _pubmed_session
    if _pubmed_session is None:
        s = requests.Session()
        s.headers.update({"User-Agent": "Joy-OT-Fetcher/1.0"})
        _pubmed_session = s
    return _pubmed_session

# ---------------- GraphQL Queries ----------------

GQL_EVIDENCE = """
query DiseaseGeneEvidence($efo: String!, $ensg: String!, $size: Int!, $cursor: String, $enableIndirect: Boolean!) {
  disease(efoId: $efo) {
    evidences(ensemblIds: [$ensg], enableIndirect: $enableIndirect, size: $size, cursor: $cursor) {
      count
      cursor
      rows {
        datasourceId
        publicationYear
        literature   # [String!] list of PMIDs
      }
    }
  }
}
"""

# Note: historical spelling is "literatureOcurrences" (one 'r'). If OT changes it, update here.
GQL_DISEASE_LIT = """
query DiseaseLiterature($efo: String!, $size: Int!, $cursor: String) {
  disease(efoId: $efo) {
    literatureOcurrences(size: $size, cursor: $cursor) {
      count
      cursor
      rows {
        pmid
        pmcid
        publicationDate
      }
    }
  }
}
"""

# ---------------- HTTP Helpers ----------------

def _post_graphql(query: str, variables: Dict, retries: int = 3, timeout: int = DEFAULT_TIMEOUT) -> Dict:
    last_err = None
    s = get_ot_session()
    for attempt in range(1, retries + 1):
        try:
            r = s.post(OT_GQL_URL, json={"query": query, "variables": variables}, timeout=timeout)
            r.raise_for_status()
            payload = r.json()
            if "errors" in payload:
                raise RuntimeError(f"GraphQL errors: {payload['errors']}")
            return payload.get("data", {})
        except Exception as e:
            last_err = e
            if attempt == retries:
                break
            time.sleep(1.2 * attempt)
    raise RuntimeError(f"GraphQL request failed after {retries} attempts: {last_err}")

# Europe PMC (core) — richer than lite; better chance to get abstract/keywords/journal
def enrich_from_europe_pmc_single(pmid: str, timeout: int = DEFAULT_TIMEOUT) -> Optional[Dict]:
    params = {
        "query": f"EXT_ID:{pmid}",
        "format": "json",
        "pageSize": 1,
        "resultType": "core",  # key change
    }
    s = get_epmc_session()
    try:
        r = s.get(EUROPE_PMC_SEARCH, params=params, timeout=timeout)
        r.raise_for_status()
        j = r.json()
        hits = j.get("resultList", {}).get("result", [])
        if not hits:
            return None
        hit = hits[0]
        title = hit.get("title")
        abstract = hit.get("abstractText")  # may still be None
        year = hit.get("pubYear")
        journal = hit.get("journalInfo").get("journal").get("title")
        if journal is None:
            journal = hit.get("reference").split(".")[0]
        author_str = hit.get("authorString")

        hit.get("authorString") or ""
        authors = [a.strip() for a in author_str.split(",")] if author_str else []
        kws = hit.get("keywordList", {}).get("keyword") if isinstance(hit.get("keywordList"), dict) else None
        keywords = kws if isinstance(kws, list) else []
        return {
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "year": str(year) if year else None,
            "journal": journal,
            "keywords": keywords
        }
    except Exception:
        return None

def enrich_many_epmc(pmids: List[str], max_workers: int) -> Dict[str, Optional[Dict]]:
    out: Dict[str, Optional[Dict]] = {}
    if not pmids:
        return out
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut2pmid = {ex.submit(enrich_from_europe_pmc_single, pmid): pmid for pmid in pmids}
        for fut in as_completed(fut2pmid):
            pmid = fut2pmid[fut]
            try:
                out[pmid] = fut.result()
            except Exception:
                out[pmid] = None
    return out

# ---------------- PubMed EFetch backfill (batch abstracts + journal) ----------------

def _parse_pubmed_biblio_xml(xml_text: str) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Parse PubMed EFetch XML; return {pmid: {"abstract": str|None, "journal": str|None}}
    """
    res: Dict[str, Dict[str, Optional[str]]] = {}
    root = ET.fromstring(xml_text)
    for art in root.findall(".//PubmedArticle"):
        pmid_el = art.find(".//MedlineCitation/PMID")
        if pmid_el is None or not pmid_el.text:
            continue
        pmid = pmid_el.text.strip()

        # Abstract: concatenate all AbstractText nodes (keep section labels if any)
        abs_nodes = art.findall(".//MedlineCitation/Article/Abstract/AbstractText")
        abstract_val: Optional[str]
        if not abs_nodes:
            abstract_val = None
        else:
            parts = []
            for node in abs_nodes:
                label = node.attrib.get("Label")
                text = (node.text or "").strip()
                if label and text:
                    parts.append(f"{label}: {text}")
                elif text:
                    parts.append(text)
            abstract_val = "\n".join(parts) if parts else None

        # Journal title: prefer Article/Journal/Title; fallback MedlineJournalInfo/MedlineTA
        journal_title_node = art.find(".//MedlineCitation/Article/Journal/Title")
        journal_val = journal_title_node.text.strip() if (journal_title_node is not None and journal_title_node.text) else None
        if not journal_val:
            ta_node = art.find(".//MedlineCitation/MedlineJournalInfo/MedlineTA")
            if ta_node is not None and ta_node.text:
                journal_val = ta_node.text.strip()

        res[pmid] = {"abstract": abstract_val, "journal": journal_val}
    return res

def fetch_pubmed_biblio_batch(pmids: List[str], api_key: Optional[str] = None,
                              chunk_size: int = 100, retries: int = 3,
                              timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Batch fetch abstracts & journal from PubMed EFetch XML.
    Returns {pmid: {"abstract": str|None, "journal": str|None}}
    """
    out: Dict[str, Dict[str, Optional[str]]] = {p: {"abstract": None, "journal": None} for p in pmids}
    if not pmids:
        return out
    s = get_pubmed_session()

    for i in range(0, len(pmids), chunk_size):
        chunk = pmids[i:i+chunk_size]
        params = {"db": "pubmed", "pmid": ",".join(chunk), "retmode": "xml"}
        if api_key:
            params["api_key"] = api_key

        last_err = None
        for attempt in range(1, retries + 1):
            try:
                r = s.get(PUBMED_EFETCH, params=params, timeout=timeout)
                r.raise_for_status()
                parsed = _parse_pubmed_biblio_xml(r.text)
                out.update(parsed)
                break
            except Exception as e:
                last_err = e
                if attempt == retries:
                    sys.stderr.write(f"[warn] PubMed EFetch failed for chunk {i//chunk_size+1}: {last_err}\n")
                time.sleep(1.2 * attempt)
    return out

def backfill_abs_journal_with_pubmed(results: List[Dict], max_backfill: Optional[int] = None) -> Tuple[int, int]:
    """
    Backfill missing abstract and journal in-place via PubMed EFetch.
    Returns (filled_abstract_count, filled_journal_count).
    """
    targets: List[str] = []
    for r in results:
        need_abs = not r.get("abstract")
        need_jnl = not r.get("journal")
        if need_abs or need_jnl:
            pid = r.get("pmid") or ""
            if pid.startswith("PMID:"):
                targets.append(pid.split("PMID:")[1])

    if max_backfill is not None:
        targets = targets[:max_backfill]
    if not targets:
        return (0, 0)

    api_key = os.environ.get("NCBI_API_KEY")
    biblio = fetch_pubmed_biblio_batch(targets, api_key=api_key, chunk_size=100)

    filled_abs = 0
    filled_jnl = 0
    for r in results:
        pmid = (r.get("pmid") or "").replace("PMID:", "")
        fill = biblio.get(pmid)
        if not fill:
            continue
        # abstract
        if not r.get("abstract") and fill.get("abstract"):
            r["abstract"] = fill["abstract"]
            filled_abs += 1
        # journal
        if not r.get("journal") and fill.get("journal"):
            r["journal"] = fill["journal"]
            filled_jnl += 1
        # annotate source
        if (fill.get("abstract") or fill.get("journal")):
            src = r.get("source") or ""
            if "PubMed" not in src:
                r["source"] = f"{src} + PubMed".strip(" +")
    return (filled_abs, filled_jnl)

# ---------------- Iterators ----------------

def iter_evidence_rows(efo: str, ensg: str, size: int = 500, enable_indirect: bool = True) -> Iterable[Dict]:
    cursor = None
    while True:
        variables = {"efo": efo, "ensg": ensg, "size": size, "cursor": cursor, "enableIndirect": enable_indirect}
        data = _post_graphql(GQL_EVIDENCE, variables)
        disease = data.get("disease")
        if not disease:
            break
        ev = disease.get("evidences") or {}
        rows = ev.get("rows") or []
        for row in rows:
            yield row
        cursor = ev.get("cursor")
        if not cursor:
            break

def iter_disease_lit_rows(efo: str, size: int = 500) -> Iterable[Dict]:
    cursor = None
    while True:
        variables = {"efo": efo, "size": size, "cursor": cursor}
        data = _post_graphql(GQL_DISEASE_LIT, variables)
        disease = data.get("disease")
        if not disease:
            break
        lit = disease.get("literatureOcurrences") or {}
        rows = lit.get("rows") or []
        for row in rows:
            yield row
        cursor = lit.get("cursor")
        if not cursor:
            break

# ---------------- Utils & Mappers ----------------

def _pmid_to_number(pmid: Optional[str]) -> Optional[int]:
    if not pmid:
        return None
    s = str(pmid)
    return int(s) if s.isdigit() else None

def map_evidence_batch(row: Dict, enrich: bool, datasource_filter: Optional[Set[str]],
                       epmc_cache: Dict[str, Optional[Dict]]) -> List[Dict]:
    ds = (row.get("datasourceId") or "").strip()
    if datasource_filter and ds not in datasource_filter:
        return []
    pmids = [str(x) for x in (row.get("literature") or []) if str(x).strip()]
    year_row = row.get("publicationYear")
    out: List[Dict] = []

    for pmid in pmids:
        title = abstract = journal = None
        authors: List[str] = []
        keywords: List[str] = []
        year_final: Optional[str] = str(year_row) if year_row is not None else None

        if enrich:
            meta = epmc_cache.get(pmid)
            if meta:
                title = meta.get("title")
                abstract = meta.get("abstract")
                authors = meta.get("authors") or []
                keywords = meta.get("keywords") or []
                journal = meta.get("journal")
                if meta.get("year"):
                    year_final = meta["year"]

        record = {
            "pmid": f"PMID:{pmid}",
            "number": _pmid_to_number(pmid),
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "keywords": keywords,
            "journal": journal,
            "source": ("Open Targets" + (f" | {ds}" if ds else "")) if not enrich else ("Open Targets + Europe PMC" + (f" | {ds}" if ds else "")),
            "forum_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "year": year_final
        }
        out.append(record)
    return out

def map_disease_row(row: Dict, enrich: bool, epmc_cache: Dict[str, Optional[Dict]]) -> Optional[Dict]:
    pmid = str(row.get("pmid") or "").strip()
    if not pmid:
        return None
    title = abstract = journal = None
    authors: List[str] = []
    keywords: List[str] = []
    year_final: Optional[str] = None
    pubdate = row.get("publicationDate")
    if isinstance(pubdate, str) and len(pubdate) >= 4 and pubdate[:4].isdigit():
        year_final = pubdate[:4]

    if enrich:
        meta = epmc_cache.get(pmid)
        if meta:
            title = meta.get("title")
            abstract = meta.get("abstract")
            authors = meta.get("authors") or []
            keywords = meta.get("keywords") or []
            journal = meta.get("journal")
            if meta.get("year"):
                year_final = meta["year"]

    return {
        "pmid": f"PMID:{pmid}",
        "number": _pmid_to_number(pmid),
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "keywords": keywords,
        "journal": journal,
        "source": "Open Targets" if not enrich else "Open Targets + Europe PMC",
        "forum_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        "year": year_final
    }

# ---------------- Incremental writers ----------------

def atomic_write_json(path: str, obj) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

class IncrementalSaver:
    def __init__(self, out_json: Optional[str], out_jsonl: Optional[str], save_every: int):
        self.out_json = out_json
        self.out_jsonl = out_jsonl
        self.save_every = save_every
        self.last_saved_idx = 0
        if self.out_jsonl:
            open(self.out_jsonl, "a", encoding="utf-8").close()

    def maybe_checkpoint(self, results: List[Dict], force: bool = False):
        if not results:
            return
        if (len(results) - self.last_saved_idx) >= self.save_every or force:
            if self.out_jsonl:
                with open(self.out_jsonl, "a", encoding="utf-8") as f:
                    for obj in results[self.last_saved_idx:]:
                        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            if self.out_json:
                atomic_write_json(self.out_json, results)
            self.last_saved_idx = len(results)
            print(f"[checkpoint] saved {self.last_saved_idx} records.", file=sys.stderr)

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description="Fetch literature from Open Targets GraphQL v4 (fast + backfill).")
    ap.add_argument("--mode", choices=["evidence", "disease"], default="evidence",
                    help="Fetch mode: 'evidence' = disease×gene intersection (default), 'disease' = disease-wide literature.")
    ap.add_argument("--ensembl", default="ENSG00000080815", help="Ensembl gene ID, e.g., ENSG00000080815 (PSEN1). Required in evidence mode.")
    ap.add_argument("--disease", default="MONDO_0004975", required=True, help="Disease MONDO/EFO ID, e.g., MONDO_0004975 (Alzheimer's).")
    ap.add_argument("--out", help="Write a JSON array snapshot to this file (checkpointed).")
    ap.add_argument("--jsonl", default="tmp.jsonl", help="Write JSONL (streaming) to this file (appended at checkpoints).")
    ap.add_argument("--size", type=int, default=500, help="GraphQL page size (default: 500).")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on number of papers.")
    ap.add_argument("--enrich", action="store_true", help="Enrich with Europe PMC (title/abstract/authors/keywords/year/journal).")
    ap.add_argument("--enable-indirect", action="store_true", default=True,
                    help="[evidence mode only] Enable indirect evidences (default True). Use --no-enable-indirect to disable.")
    ap.add_argument("--no-enable-indirect", dest="enable_indirect", action="store_false",
                    help="[evidence mode only] Disable indirect evidences.")
    ap.add_argument("--datasource-filter", type=str, default="",
                    help="[evidence mode only] Keep only these datasourceIds, e.g., 'europepmc,uniprot_variants'.")

    # Performance/robustness knobs
    ap.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY, help="Checkpoint every N records (default 500).")
    ap.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Concurrent workers for enrichment (default 12).")
    ap.add_argument("--pubmed-backfill", action="store_true",
                    help="After enrichment, backfill missing abstracts & journal via PubMed EFetch (batch).")

    args = ap.parse_args()

    if args.mode == "evidence" and not args.ensembl:
        ap.error("--ensembl is required in evidence mode.")

    ds_filter: Optional[Set[str]] = None
    if args.datasource_filter.strip():
        ds_filter = set(s.strip() for s in args.datasource_filter.split(",") if s.strip())

    results: List[Dict] = []
    seen_ids: Set[str] = set()
    saver = IncrementalSaver(args.out, args.jsonl, save_every=args.save_every)

    def add_records(new_records: List[Dict]) -> None:
        nonlocal results, seen_ids
        for r in new_records:
            key = r["pmid"]
            if key in seen_ids:
                continue
            seen_ids.add(key)
            results.append(r)

    if args.mode == "evidence":
        scanned_rows = 0
        for row in iter_evidence_rows(args.disease, args.ensembl, size=args.size, enable_indirect=args.enable_indirect):
            scanned_rows += 1
            pmids = [str(x) for x in (row.get("literature") or []) if str(x).strip()]

            # Only enrich unseen PMIDs
            to_enrich = []
            for pmid in pmids:
                key = f"PMID:{pmid}"
                if key not in seen_ids:
                    to_enrich.append(pmid)

            epmc_cache: Dict[str, Optional[Dict]] = {}
            if args.enrich and to_enrich:
                epmc_cache = enrich_many_epmc(to_enrich, max_workers=args.max_workers)

            recs = map_evidence_batch(row, enrich=args.enrich, datasource_filter=ds_filter, epmc_cache=epmc_cache)
            before = len(results)
            add_records(recs)

            if args.limit and len(results) >= args.limit:
                print(f"[info] reached limit {args.limit}", file=sys.stderr)
                break
            if scanned_rows % 200 == 0:
                print(f"[info] scanned {scanned_rows} evidence rows; unique papers: {len(results)}", file=sys.stderr)

            # checkpoint boundary crossed?
            if len(results) // args.save_every > before // args.save_every:
                saver.maybe_checkpoint(results)

    else:
        scanned_rows = 0
        batch_pmids: List[str] = []
        batch_rows: List[Dict] = []

        def flush_batch():
            nonlocal batch_pmids, batch_rows
            if not batch_rows:
                return
            unseen_pmids = [p for p in batch_pmids if f"PMID:{p}" not in seen_ids]
            epmc_cache: Dict[str, Optional[Dict]] = {}
            if args.enrich and unseen_pmids:
                epmc_cache = enrich_many_epmc(unseen_pmids, max_workers=args.max_workers)

            new_records = []
            for row in batch_rows:
                r = map_disease_row(row, enrich=args.enrich, epmc_cache=epmc_cache)
                if r:
                    new_records.append(r)
            before = len(results)
            add_records(new_records)
            if len(results) // args.save_every > before // args.save_every:
                saver.maybe_checkpoint(results)
            batch_pmids = []
            batch_rows = []

        for row in iter_disease_lit_rows(args.disease, size=args.size):
            scanned_rows += 1
            pmid = str(row.get("pmid") or "").strip()
            if pmid:
                batch_pmids.append(pmid)
            batch_rows.append(row)

            if scanned_rows % args.size == 0:
                flush_batch()

            if args.limit and len(results) >= args.limit:
                print(f"[info] reached limit {args.limit}", file=sys.stderr)
                break

            if scanned_rows % 1000 == 0:
                print(f"[info] scanned {scanned_rows} disease-lit rows; unique papers: {len(results)}", file=sys.stderr)

        flush_batch()

    # Final checkpoint of what we have so far
    saver.maybe_checkpoint(results, force=True)

    # Optional PubMed backfill for missing abstracts & journal (after EPMC enrich)
    if args.enrich and args.pubmed_backfill:
        filled_abs, filled_jnl = backfill_abs_journal_with_pubmed(results, max_backfill=None)
        if filled_abs or filled_jnl:
            print(f"[backfill] PubMed filled abstracts: {filled_abs}, journals: {filled_jnl}.", file=sys.stderr)
            saver.maybe_checkpoint(results, force=True)

    if args.jsonl:
        print(f"[done] Wrote/appended {len(results)} records to {args.jsonl}", file=sys.stderr)
    if args.out:
        print(f"[done] Snapshot ({len(results)} records) saved to {args.out}", file=sys.stderr)
    if not (args.out or args.jsonl):
        json.dump(results, sys.stdout, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()