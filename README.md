# SearchMedPaper
Search paper by embedding, for literature on PubMed. We rank them by embedding and the source of journal, and give a conclusion of entity correlation, such as the relationship between gene and disease.

# 1. Get Literature

 We crawl data from OpenTarget, Europe PMC and PubMed. Mode "--enrich --pubmed-backfill" can fill in the missing abstract/title/authors/journal with PubMed.

- From Open Targets, fetch the id of literature, e.g. ENSG00000128564
- → enrich metadata using Europe PMC in parallel
- → Then use PubMed to batch-fill abstracts and journal titles
- → If information is still missing, try to extract it from the reference text

```python
git clone https://github.com/JOY-SWang/SearchMedPaper.git
cd SearchMedPaper
export NCBI_API_KEY="" # with NCBI_API_KEY, you can get faster crawing speed
# Disease-wide (large count)
  python crawl.py --mode disease --disease {disease_EFOID}  --out {json_file} --jsonl {jsonl_path} --enrich --pubmed-backfill

# Disease × Gene intersection
python crawl.py --mode evidence --ensembl {gene_ensemblID} --disease {disease_EFOID} \
  --jsonl {jsonl_path} --out {json_file} --enrich --pubmed-backfill \
  --max-workers 8 --save-every 500

```

Optimization:

- Use a global requests.Session to reuse connections with keep-alive, which is faster and more resource-efficient than creating a new connection for every requests.get/post.
- Define GraphQL queries and use Ensembl IDs for faster and more precise retrieval.
- Parse the XML structure in a way that supports multiple abstract sections per article.
- Use PubMed with an NCBI_API_KEY to increase rate limits.
- Append to JSONL incrementally, and rewrite the full JSON array at each checkpoint, but use atomic_write_json to ensure safe writes.



If you only have the name of genes and diseases, you can refer to the code in "betterPPL4crawl.py", with pairs of disease and protein.

```
demo_pairs = [({diseaseName}, {geneName})] # e.g. ("Lung Neoplasms", "AKT1")
fetch_literature_for_disease_gene_pairs(
      demo_pairs,
      out_json="test.json",
      out_jsonl="test.jsonl",
      enrich=True,
      pubmed_backfill=True,
      size=500,
      save_every=500,
      max_workers=DEFAULT_MAX_WORKERS,
  )
```



Finally, you can get dicts in the following format:

```
{
	"pmid": "", 
	"number": 0, 	# pmid
	"title": "", 
	"authors": ["Author #1", "Author #2", ..., "Author #N"], 
	"abstract": "", 
	"keywords": ["", "..., ""], 
	"journal": "", 
	"source": "",  # where you get this data: Open Targets/Europe PMC/...
	"forum_url": "", 
	"year": ""
}
```



# 2. Embedding Computation

We gratefully acknowledge that our Embedding Computation code is modified from the official implementation of [SearchPaperByEmbedding](https://github.com/gyj155/SearchPaperByEmbedding). Sepecially, we use a caching mechanism (_load_cache / _save_cache) that only treats the cache as valid if the number of vectors in the cache matches the number of papers, and compute an importance score based on journal weights.

![image-20251125193326583](/Users/joysw/Library/Application Support/typora-user-images/image-20251125193326583.png)

```
from search import PaperSearcher
searcher = PaperSearcher('papers.json', model_type='local') 

# export OPENAI_API_KEY='your-key'
# searcher = PaperSearcher('papers.json', model_type='openai')

searcher.compute_embeddings()
examples = ""	# long text
results = searcher.search(examples=examples, top_k=100)

# Or search with text query
results = searcher.search(query="PSEN1, Alzheimer’s disease", top_k=10)

searcher.save(results, 'output.json')
searcher.display(results, n=10)
```



# 3. Relationship Conclusion

With the most relevant literature, we can get Relation Judgement & Final Conclusion. Specially, we transfer study_type returned by the LLM for each piece of evidence is roughly matched to a standard category (meta-analysis, RCT, cohort, animal, in_vitro, etc.). For each generated piece of evidence, a “reliability/strength” score is computed, taking into account: study design (highest weight 0.45), sample size (log-scaled, weight 0.25), p-value range (0.20), and whether confounders were addressed (0.10). Each note’s direction (+1 / -1 / 0) is then weighted by its strength s and summed. Representative notes are selected, their content compressed into a summary, and English bullet-point conclusions are produced with citations like [PMID:xxxx].

```
from literatureLLM import *
Disease = ""
Protein = ""
papers = []
# papers = [
#   {"pmid":"40...98", "title":"...", "abstract":"...", "year":2025, "journal":"..."},
#   ... # 共10篇
# ]

out = run_pipeline(Disease, Protein, papers)
```



The output looks like the text in the following format:

```
[TEST] 2025-11-25 19:55:08 | Start pipeline for DISEASE='Alzheimer's disease' PROTEIN='APOE'
[TEST] Papers to process: 10
[TEST] [A] Extracting structured notes ...
[TEST]   - (1/10) Extracting note for PMID=PMID:21349439 ... done in 5s | ETA ~ 50s
[TEST]   - (2/10) Extracting note for PMID=PMID:28111074 ... done in 4s | ETA ~ 41s
[TEST]   - (3/10) Extracting note for PMID=PMID:8103819 ... done in 4s | ETA ~ 34s
[TEST]   - (4/10) Extracting note for PMID=PMID:7715296 ... done in 4s | ETA ~ 29s
[TEST]   - (5/10) Extracting note for PMID=PMID:33340485 ... done in 3s | ETA ~ 23s
[TEST]   - (6/10) Extracting note for PMID=PMID:19339974 ... done in 3s | ETA ~ 17s
[TEST]   - (7/10) Extracting note for PMID=PMID:9425904 ... done in 4s | ETA ~ 13s
[TEST]   - (8/10) Extracting note for PMID=PMID:38898183 ... done in 3s | ETA ~ 8s
[TEST]   - (9/10) Extracting note for PMID=PMID:10944562 ... done in 4s | ETA ~ 4s
[TEST]   - (10/10) Extracting note for PMID=PMID:23883936 ... done in 5s | ETA ~ 0s
[TEST] [A] Completed note extraction: 10 notes | 44s elapsed
[TEST] [B] Polarity: positive | score=1.0 | confidence=0.759 | coverage=0.523 | computed in 0s
[TEST] [C] Compressing notes into dense conclusions ... done in 10s
[TEST] 2025-11-25 19:56:03 | All done in 55s
[TEST] SUMMARY | direction=positive | score=1.0 | confidence=0.759 | coverage=0.523 | N_notes=10 | total_time=55s

=== FINAL CONCLUSIONS ===

- Overall direction: Positive association between APOE (particularly the ε4 allele) and Alzheimer's disease (AD) risk is supported with moderate-to-high confidence (confidence ~0.76; coverage ~0.52). This is based on multiple genetic and functional studies showing increased AD risk and altered brain function in APOE ε4 carriers [PMID:8103819, 10944562].

- Mechanistic signals: APOE ε4 allele carriers exhibit greater brain activation in AD-vulnerable regions (hippocampus, parietal, prefrontal cortex) during memory tasks, suggesting altered neural activity preceding clinical symptoms [PMID:10944562]. Genetic variants such as -491A polymorphism may increase APOE expression, potentially modulating AD risk independently of ε4 status [PMID:9425904]. Transcriptomic analyses reveal APOE4-associated shifts resembling late-onset AD (LOAD) profiles, implicating regulatory mediators (APBA2, FYN, RNF219, SV2A) involved in APP endocytosis and amyloid metabolism; RNF219 variants also influence amyloid deposition and age-of-onset [PMID:23883936]. APOE alleles modulate AD and cerebral amyloid angiopathy risk dose-dependently, with ε4 increasing and ε2 decreasing risk [PMID:21349439].

- Context-specific findings: Increased brain activation in APOE ε4 carriers was observed in older cognitively intact individuals, indicating early functional changes before clinical AD onset [PMID:10944562]. Transcriptomic shifts were identified in cerebral cortex tissue from unaffected APOE4 carriers and LOAD patients, suggesting molecular alterations in relevant brain regions [PMID:23883936]. Genetic associations were confirmed in sporadic AD patients versus controls [PMID:8103819].

- Causal evidence and robustness: Evidence derives primarily from observational cohort and case-control genetic association studies, supported by in vitro functional assays and transcriptomic analyses. The APOE ε4 allele frequency difference between AD and controls is statistically significant (p<0.01) and replicated across independent populations [PMID:8103819]. Functional brain imaging and molecular data provide convergent mechanistic support. However, no randomized controlled trials or Mendelian randomization studies are reported here, limiting causal inference robustness.

- Conflicts and uncertainties: No contradictory evidence was noted in the provided data. Some uncertainty remains regarding the independent effect of non-ε4 APOE variants (e.g., -491A) and the precise molecular pathways mediating APOE4’s impact on amyloid metabolism. Further longitudinal studies integrating genetic, transcriptomic, and imaging data, as well as interventional trials targeting APOE pathways, would strengthen causal inference and clarify mechanisms.

- Verifiability checklist:
  - APOE ε4 allele frequency significantly higher in sporadic AD vs controls [PMID:8103819]
  - Increased brain activation in hippocampal and related regions in cognitively intact APOE ε4 carriers [PMID:10944562]
  - -491A polymorphism associated with increased APOE expression and AD risk independent of ε4 [PMID:9425904]
  - APOE4 carrier status linked to transcriptomic shifts resembling LOAD, involving APP metabolism regulators [PMID:23883936]
  - Dose-dependent modulation of AD risk by APOE alleles ε2, ε3, ε4 [PMID:21349439]
```

# Acknowledgements
This project was completed under the supervision of Professor Bingxin Zhao at the University of Pennsylvania. I am deeply grateful to Bingxuan Li and Dr. Zichen Zhang for their generous help and guidance throughout this project.

For any questions, please contact joywang909@gmail.com.
