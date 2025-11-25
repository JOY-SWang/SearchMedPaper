#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
betterPPL4crawl.py — Fetch literature from Open Targets GraphQL v4, with:
- Input as a list of (disease_name, gene_symbol) pairs
- Automatic mapping: disease name -> EFO/MONDO ID; gene symbol -> Ensembl ID
- Concurrent Europe PMC enrichment (resultType=core)
- PubMed EFetch batch backfill for missing abstracts and journal titles
- Incremental checkpointing every N records to JSON/JSONL
- Robust GraphQL paging and defensive handling
- Filtering out any non-numeric "PMID" (e.g. PPR411215) to avoid PubMed batch failures
- For each pair, output:
    "{disease}_{gene}.snapshot.json"
    "{disease}_{gene}.jsonl"
  (after slugifying disease/gene into safe file-name fragments)
"""

import json
import os
import sys
import time
import re
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

# Search endpoint for resolving names → IDs
GQL_SEARCH_DISEASE_BY_NAME = """
query SearchDisease($queryString: String!) {
  search(queryString: $queryString, entityNames: ["disease"], page: { index: 0, size: 5 }) {
    hits {
      id
      entity
      score
      object {
        ... on Disease {
          id
          name
        }
      }
    }
  }
}
"""

GQL_SEARCH_TARGET_BY_SYMBOL = """
query SearchTarget($queryString: String!) {
  search(queryString: $queryString, entityNames: ["target"], page: { index: 0, size: 5 }) {
    hits {
      id
      entity
      score
      object {
        ... on Target {
          id
          approvedSymbol
          approvedName
        }
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


# ---------------- Name → ID resolvers ----------------

def resolve_disease_id(disease_name: str) -> Optional[str]:
    """
    Resolve a human-readable disease name to an Open Targets disease ID (EFO/MONDO).

    Example:
        "Colorectal Neoplasms" -> "EFO_0004142"
    """
    variables = {"queryString": disease_name}
    data = _post_graphql(GQL_SEARCH_DISEASE_BY_NAME, variables)
    search_block = data.get("search") or {}
    hits = search_block.get("hits") or []

    best_hit = None
    for h in hits:
        if (h.get("entity") or "").lower() != "disease":
            continue
        obj = h.get("object") or {}
        name = (obj.get("name") or "").strip()
        # Prefer exact match (case-insensitive)
        if name.lower() == disease_name.lower():
            best_hit = h
            break
        if best_hit is None:
            best_hit = h

    if not best_hit:
        return None
    obj = best_hit.get("object") or {}
    return obj.get("id") or best_hit.get("id")


def resolve_target_ensembl_id(gene_symbol: str) -> Optional[str]:
    """
    Resolve HGNC gene symbol to Ensembl ID via Open Targets search.

    Example:
        "CCR5" -> "ENSG00000160791"
    """
    variables = {"queryString": gene_symbol}
    data = _post_graphql(GQL_SEARCH_TARGET_BY_SYMBOL, variables)
    search_block = data.get("search") or {}
    hits = search_block.get("hits") or []

    best_hit = None
    for h in hits:
        if (h.get("entity") or "").lower() != "target":
            continue
        obj = h.get("object") or {}
        symbol = (obj.get("approvedSymbol") or "").strip()
        # Prefer exact symbol match (case-insensitive)
        if symbol.upper() == gene_symbol.upper():
            best_hit = h
            break
        if best_hit is None:
            best_hit = h

    if not best_hit:
        return None
    obj = best_hit.get("object") or {}
    return obj.get("id") or best_hit.get("id")


# ---------------- Europe PMC enrichment ----------------

def enrich_from_europe_pmc_single(pmid: str, timeout: int = DEFAULT_TIMEOUT) -> Optional[Dict]:
    params = {
        "query": f"EXT_ID:{pmid}",
        "format": "json",
        "pageSize": 1,
        "resultType": "core",
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
        abstract = hit.get("abstractText")
        year = hit.get("pubYear")

        journal = None
        ji = hit.get("journalInfo")
        if isinstance(ji, dict):
            j_j = ji.get("journal")
            if isinstance(j_j, dict):
                journal = j_j.get("title")
        if journal is None:
            ref = hit.get("reference")
            if isinstance(ref, str) and "." in ref:
                journal = ref.split(".")[0].strip()

        author_str = hit.get("authorString") or ""
        authors = [a.strip() for a in author_str.split(",")] if author_str else []

        kws = None
        kw_list = hit.get("keywordList")
        if isinstance(kw_list, dict):
            kws = kw_list.get("keyword")
        keywords = kws if isinstance(kws, list) else []

        return {
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "year": str(year) if year else None,
            "journal": journal,
            "keywords": keywords,
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


def fetch_pubmed_biblio_batch(
    pmids: List[str],
    api_key: Optional[str] = None,
    chunk_size: int = 100,
    retries: int = 3,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Batch fetch abstracts & journal from PubMed EFetch XML.
    Returns {pmid: {"abstract": str|None, "journal": str|None}}
    """
    out: Dict[str, Dict[str, Optional[str]]] = {p: {"abstract": None, "journal": None} for p in pmids}
    if not pmids:
        return out
    s = get_pubmed_session()

    for i in range(0, len(pmids), chunk_size):
        chunk = pmids[i:i + chunk_size]
        # IMPORTANT: EFetch uses "id", not "pmid"
        params = {"db": "pubmed", "id": ",".join(chunk), "retmode": "xml"}
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
                    sys.stderr.write(f"[warn] PubMed EFetch failed for chunk {i // chunk_size + 1}: {last_err}\n")
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
                clean = pid.split("PMID:")[1]
                if clean.isdigit():
                    targets.append(clean)

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


# ---------------- Iterators (kept for completeness, not used by main API) ----------------

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

def _clean_pmid_str(x: Optional[str]) -> str:
    """
    Normalize any pmid-like string:
      "PMID:12345" -> "12345"
      "12345"      -> "12345"
      " PPR411215" -> "PPR411215" (later rejected as non-numeric)
    """
    if x is None:
        return ""
    s = str(x).strip()
    if s.upper().startswith("PMID:"):
        s = s[5:].strip()
    return s


def _pmid_to_number(pmid: Optional[str]) -> Optional[int]:
    if not pmid:
        return None
    s = _clean_pmid_str(pmid)
    return int(s) if s.isdigit() else None


def map_evidence_batch(
    row: Dict,
    enrich: bool,
    datasource_filter: Optional[Set[str]],
    epmc_cache: Dict[str, Optional[Dict]],
) -> List[Dict]:
    """
    Map one evidence row to a list of per-PMID records.

    IMPORTANT:
      - Only keep PMIDs that are pure digits.
      - Any ID containing letters (e.g. PPR411215) is discarded.
    """
    ds = (row.get("datasourceId") or "").strip()
    if datasource_filter and ds not in datasource_filter:
        return []

    # Filter to numeric PMIDs only
    raw_ids = row.get("literature") or []
    pmids: List[str] = []
    for x in raw_ids:
        cleaned = _clean_pmid_str(x)
        if cleaned and cleaned.isdigit():
            pmids.append(cleaned)

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
            "source": (
                "Open Targets" + (f" | {ds}" if ds else "")
                if not enrich
                else "Open Targets + Europe PMC" + (f" | {ds}" if ds else "")
            ),
            "forum_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "year": year_final,
        }
        out.append(record)
    return out


def map_disease_row(row: Dict, enrich: bool, epmc_cache: Dict[str, Optional[Dict]]) -> Optional[Dict]:
    """
    Map disease-wide literature row to record.

    IMPORTANT:
      - Return None for any non-numeric PMID.
    """
    pmid_raw = row.get("pmid")
    pmid = _clean_pmid_str(pmid_raw)
    if not pmid or not pmid.isdigit():
        # Drop non-numeric IDs (preprints etc.)
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
        "year": year_final,
    }


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


def _slugify_pair_name(disease_name: str, gene_symbol: str) -> str:
    """
    用 disease_name 和 gene_symbol 构造一个安全的文件名前缀。
    例如：
      ("Alzheimer's disease", "PSEN1") -> "Alzheimer_s_disease_PSEN1"
      ("Colorectal Neoplasms", "CCR5") -> "Colorectal_Neoplasms_CCR5"
    """
    base = f"{disease_name}_{gene_symbol}"
    s = re.sub(r"[^\w\-]+", "_", base)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "pair"


# ---------------- High level API for (disease_name, gene_symbol) pairs ----------------

def fetch_literature_for_disease_gene_pairs(
    disease_gene_pairs: List[Tuple[str, str]],
    # 这两个参数现在只控制“是否写文件”，具体文件名由 pair 决定
    out_json: Optional[str] = None,
    out_jsonl: Optional[str] = None,
    size: int = 500,
    limit: Optional[int] = None,
    enrich: bool = True,
    enable_indirect: bool = True,
    datasource_filter: str = "",
    save_every: int = DEFAULT_SAVE_EVERY,
    max_workers: int = DEFAULT_MAX_WORKERS,
    pubmed_backfill: bool = True,
) -> List[Dict]:
    """
    以 (disease_name, gene_symbol) list 拉取文献。

    新需求：
      - 对于 pairs 中的每个 pair，单独写：
          "{disease}_{protein}.snapshot.json"
          "{disease}_{protein}.jsonl"
        其中 disease/protein 通过 _slugify_pair_name 清洗为安全文件名。
      - 函数返回值是跨所有 pair 的去重文献列表（全局去重）。
    """

    ds_filter: Optional[Set[str]] = None
    if datasource_filter.strip():
        ds_filter = set(s.strip() for s in datasource_filter.split(",") if s.strip())

    results_all: List[Dict] = []  # 跨所有 pair 的总结果
    seen_ids: Set[str] = set()    # 全局去重（跨 pair）

    # 如果 out_json/out_jsonl 都是 None，就表示“仅返回，不写盘”
    write_files = bool(out_json or out_jsonl)

    for disease_name, gene_symbol in disease_gene_pairs:
        efo_id = resolve_disease_id(disease_name)
        ensg_id = resolve_target_ensembl_id(gene_symbol)
        if not efo_id or not ensg_id:
            sys.stderr.write(
                f"[warn] skip pair ({disease_name!r}, {gene_symbol!r}) — could not resolve IDs "
                f"(disease={efo_id}, target={ensg_id}).\n"
            )
            continue

        sys.stderr.write(
            f"[info] pair ({disease_name}, {gene_symbol}) -> disease={efo_id}, target={ensg_id}\n"
        )

        # --- 每个 pair 自己的结果和 saver ---
        results_pair: List[Dict] = []
        if write_files:
            prefix = _slugify_pair_name(disease_name, gene_symbol)
            pair_json = f"s2data500/crawl/json/{prefix}.snapshot.json"
            pair_jsonl = f"s2data500/crawl/{prefix}.jsonl"
            saver = IncrementalSaver(pair_json, pair_jsonl, save_every=save_every)
        else:
            saver = IncrementalSaver(None, None, save_every=save_every)

        def add_records_pair(new_records: List[Dict]) -> None:
            nonlocal results_pair, results_all, seen_ids
            for r in new_records:
                key = r["pmid"]
                if key in seen_ids:
                    continue
                seen_ids.add(key)
                results_pair.append(r)
                results_all.append(r)

        scanned_rows = 0
        cursor = None

        while True:
            variables = {
                "efo": efo_id,
                "ensg": ensg_id,
                "size": size,
                "cursor": cursor,
                "enableIndirect": enable_indirect,
            }
            data = _post_graphql(GQL_EVIDENCE, variables)
            disease = data.get("disease")
            if not disease:
                break
            ev = disease.get("evidences") or {}
            rows = ev.get("rows") or []
            if not rows:
                break

            for row in rows:
                scanned_rows += 1

                # 只拿合法的 numeric PMIDs 做 enrich
                raw_ids = row.get("literature") or []
                pmids: List[str] = []
                for lit_id in raw_ids:
                    cleaned = _clean_pmid_str(lit_id)
                    if cleaned and cleaned.isdigit():
                        pmids.append(cleaned)

                to_enrich: List[str] = []
                for pmid in pmids:
                    key = f"PMID:{pmid}"
                    if key not in seen_ids:
                        to_enrich.append(pmid)

                epmc_cache: Dict[str, Optional[Dict]] = {}
                if enrich and to_enrich:
                    epmc_cache = enrich_many_epmc(to_enrich, max_workers=max_workers)

                before = len(results_pair)
                recs = map_evidence_batch(
                    row,
                    enrich=enrich,
                    datasource_filter=ds_filter,
                    epmc_cache=epmc_cache,
                )
                add_records_pair(recs)

                # 全局 limit（跨 pair）
                if limit and len(results_all) >= limit:
                    print(f"[info] reached global limit {limit}", file=sys.stderr)
                    saver.maybe_checkpoint(results_pair, force=True)
                    if enrich and pubmed_backfill and results_pair:
                        filled_abs, filled_jnl = backfill_abs_journal_with_pubmed(
                            results_pair, max_backfill=None
                        )
                        if filled_abs or filled_jnl:
                            print(
                                f"[backfill] PubMed filled abstracts: {filled_abs}, journals: {filled_jnl}.",
                                file=sys.stderr,
                            )
                            saver.maybe_checkpoint(results_pair, force=True)
                    return results_all

                if scanned_rows % 200 == 0:
                    print(
                        f"[info] pair ({disease_name}, {gene_symbol}) scanned {scanned_rows} evidence rows; "
                        f"unique papers so far (global): {len(results_all)}",
                        file=sys.stderr,
                    )

                # Pair 级别 checkpoint
                if len(results_pair) // save_every > before // save_every:
                    saver.maybe_checkpoint(results_pair)

            cursor = ev.get("cursor")
            if not cursor:
                break

        # 结束当前 pair：先做 PubMed backfill，再强制写盘一次
        if enrich and pubmed_backfill and results_pair:
            filled_abs, filled_jnl = backfill_abs_journal_with_pubmed(results_pair, max_backfill=None)
            if filled_abs or filled_jnl:
                print(
                    f"[backfill] PubMed filled abstracts: {filled_abs}, journals: {filled_jnl} "
                    f"for pair ({disease_name}, {gene_symbol}).",
                    file=sys.stderr,
                )
        saver.maybe_checkpoint(results_pair, force=True)

    # 函数返回跨所有 pair 的总结果
    return results_all


# ---------------- Optional demo ----------------

if __name__ == "__main__":
    # 简单示例：直接用 disease / gene 名称列表
    demo_pairs = [
        # ("Alzheimer's disease", "PSEN1"),
        ("Lung Neoplasms", "AKT1"),
    ]
    fetch_literature_for_disease_gene_pairs(
        demo_pairs,
        # 任意给非 None 值即可触发“为每个 pair 写文件”
        out_json="test.json",
        out_jsonl="test.jsonl",
        enrich=True,
        pubmed_backfill=True,
        size=500,
        save_every=500,
        max_workers=DEFAULT_MAX_WORKERS,
    )