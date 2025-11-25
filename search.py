from __future__ import annotations

import json
import numpy as np
import os
import re
import hashlib
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

def _normalize_journal(name: str | None) -> str | None:
    if not name:
        return None
    s = name.lower().strip()
    s = re.sub(r'\(.*?\)', '', s)         # 去括号信息
    s = re.sub(r':.*$', '', s)            # 去副标题
    s = re.sub(r'[^a-z0-9\s&\-\.]', ' ', s)
    s = re.sub(r'\bthe\b', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# 你可以随时扩展/覆盖。键是“规范化后的期刊名”（见 _normalize_journal）
DEFAULT_JOURNAL_WEIGHTS = {
    # 顶级旗舰
    _normalize_journal('Nature'): 2.0,
    _normalize_journal('Science'): 2.0,
    _normalize_journal('Cell'): 2.0,
    _normalize_journal('The New England Journal of Medicine'): 2.0,
    _normalize_journal('Lancet (London, England)'): 2.0,
    _normalize_journal('JAMA'): 1.9,

    # 自然/细胞/科学子刊 & 顶会顶刊
    _normalize_journal('Nature Medicine'): 1.9,
    _normalize_journal('Nature Genetics'): 1.9,
    _normalize_journal('Nature Neuroscience'): 1.9,
    _normalize_journal('Nature Communications'): 1.6,
    _normalize_journal('Science Advances'): 1.5,
    _normalize_journal('Science Signaling'): 1.4,
    _normalize_journal('Cell Reports'): 1.5,
    _normalize_journal('Cell Death & Disease'): 1.3,
    _normalize_journal('Cell Death Discovery'): 1.2,
    _normalize_journal('PNAS'): 1.7,
    _normalize_journal('Proceedings of the National Academy of Sciences of the United States of America'): 1.7,

    # 神经/精神/阿尔兹海默相关强刊
    _normalize_journal('Neuron'): 1.8,
    _normalize_journal('Brain : a journal of neurology'): 1.6,
    _normalize_journal('Annals of Neurology'): 1.6,
    _normalize_journal('Acta Neuropathologica'): 1.7,
    _normalize_journal('Molecular Psychiatry'): 1.7,
    _normalize_journal("Alzheimer's & Dementia : the journal of the Alzheimer's Association"): 1.6,
    _normalize_journal('Human Molecular Genetics'): 1.6,
    _normalize_journal('The Journal of Neuroscience'): 1.6,
    _normalize_journal('Journal of Neuroinflammation'): 1.3,
    _normalize_journal('Neurobiology of Disease'): 1.4,
    _normalize_journal('Neurobiology of Aging'): 1.3,

    # 其它较高影响力期刊
    _normalize_journal('eLife'): 1.4,
    _normalize_journal('Genome Medicine'): 1.5,
    _normalize_journal('Theranostics'): 1.5,
    _normalize_journal('Redox Biology'): 1.4,
    _normalize_journal('EMBO Molecular Medicine'): 1.5,
    _normalize_journal('The Journal of Clinical Investigation'): 1.8,

    # 常见开放期刊/系列
    _normalize_journal('Scientific Reports'): 1.05,
    _normalize_journal('PloS one'): 0.95,
    _normalize_journal('Frontiers in Aging Neuroscience'): 1.1,
    _normalize_journal('Frontiers in Neuroscience'): 1.1,
    _normalize_journal('International Journal of Molecular Sciences'): 1.0,
    _normalize_journal('iScience'): 1.2,

    # 可能存争议或较低权重
    _normalize_journal('Oncotarget'): 0.7,
}

class PaperSearcher:
    def __init__(self, papers_file, model_type="openai", api_key=None, base_url=None,
                 journal_weights: dict | None = None,
                 journal_weight_alpha: float = 1.0,
                 journal_weight_default: float = 1.0,
                 journal_weight_minmax: tuple[float, float] = (0.5, 2.2)):
        with open(papers_file, 'r', encoding='utf-8') as f:
            self.papers = json.load(f)

        self.model_type = model_type
        self.cache_file = self._get_cache_file(papers_file, model_type)
        self.embeddings = None

        # 期刊加权相关
        self.journal_weights = (journal_weights or {})  # 用户可覆盖
        self.journal_weight_alpha = journal_weight_alpha
        self.journal_weight_default = journal_weight_default
        self.journal_weight_minmax = journal_weight_minmax

        if model_type == "openai":
            from openai import OpenAI
            self.client = OpenAI(
                api_key="", # your openai key
                base_url=base_url
            )
            self.model_name = "text-embedding-3-large"
        else:
            from sentence_transformers import SentenceTransformer
            # 建议改为模型名自动下载，避免硬编码路径
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            self.model_name = "all-MiniLM-L6-v2"

        self._load_cache()

    def _get_cache_file(self, papers_file, model_type):
        base_name = Path(papers_file).stem
        file_hash = hashlib.md5(papers_file.encode()).hexdigest()[:8]
        cache_name = f"cache_{base_name}_{file_hash}_{model_type}.npy"
        return str(Path(papers_file).parent / cache_name)

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                self.embeddings = np.load(self.cache_file)
                if len(self.embeddings) == len(self.papers):
                    print(f"Loaded cache: {self.embeddings.shape}")
                    return True
                self.embeddings = None
            except:
                self.embeddings = None
        return False

    def _save_cache(self):
        np.save(self.cache_file, self.embeddings.astype('float32'))
        print(f"Saved cache: {self.cache_file}")

    def _create_text(self, paper):
        parts = []
        if paper.get('title'):
            parts.append(f"Title: {paper['title']}")
        if paper.get('abstract'):
            parts.append(f"Abstract: {paper['abstract']}")
        if paper.get('keywords'):
            kw = ', '.join(paper['keywords']) if isinstance(paper['keywords'], list) else paper['keywords']
            parts.append(f"Keywords: {kw}")
        return ' '.join(parts)

    # ---- 新增：期刊权重器 ----
    def _journal_weight(self, jname: str | None) -> float:
        if not jname:
            return self.journal_weight_default
        key = _normalize_journal(jname)

        # 1) 明确表：用户传入 > 默认表 > 默认值
        if key in self.journal_weights:
            w = self.journal_weights[key]
        elif key in DEFAULT_JOURNAL_WEIGHTS:
            w = DEFAULT_JOURNAL_WEIGHTS[key]
        else:
            # 2) 规则推断（系列期刊）
            w = self._journal_weight_rules(key)

        # 指数缩放（alpha）+ 边界裁剪，避免过度放大/缩小
        w = float(np.clip(w, *self.journal_weight_minmax))
        if self.journal_weight_alpha != 1.0:
            w = float(np.clip(w ** self.journal_weight_alpha, *self.journal_weight_minmax))
        return w

    def _journal_weight_rules(self, key: str | None) -> float:
        if not key:
            return self.journal_weight_default
        s = key

        # 高权威系列关键词
        if s == 'nature':
            return 2.0
        if s.startswith('nature '):
            if any(x in s for x in ['medicine', 'genetics', 'neuroscience']):
                return 1.9
            if 'communications' in s:
                return 1.6
            return 1.6

        if s == 'science':
            return 2.0
        if s.startswith('science '):
            if 'advances' in s:
                return 1.5
            if 'signaling' in s:
                return 1.4
            return 1.5

        if s == 'cell':
            return 2.0
        if s.startswith('cell '):
            if 'reports' in s:
                return 1.5
            if 'death' in s:
                return 1.25
            return 1.4

        if 'new england journal of medicine' in s:
            return 2.0
        if 'lancet' in s:
            return 2.0
        if s == 'jama':
            return 1.9
        if 'jama neurology' in s:
            return 1.9

        if 'proceedings of the national academy of sciences' in s or s == 'pnas':
            return 1.7

        # 领域强刊关键词
        if 'acta neuropathologica' in s:
            return 1.7
        if 'annals of neurology' in s:
            return 1.6
        if s.startswith('brain'):
            return 1.6
        if 'molecular psychiatry' in s:
            return 1.7
        if 'human molecular genetics' in s:
            return 1.6
        if 'journal of neuroscience' in s:
            return 1.6
        if "alzheimer's & dementia" in s:
            return 1.6
        if 'neurobiology of disease' in s:
            return 1.4
        if 'neurobiology of aging' in s:
            return 1.3
        if 'journal of neuroinflammation' in s:
            return 1.3

        # 系列/开放
        if s.startswith('frontiers '):
            return 1.1
        if s == 'scientific reports':
            return 1.05
        if s in ('plos one', 'p lo s one', 'pl o s one'):
            return 0.95
        if 'international journal of molecular sciences' in s:
            return 1.0
        if 'iscience' in s:
            return 1.2

        if 'oncotarget' in s:
            return 0.7

        # 默认
        return self.journal_weight_default

    # ---- 嵌入相关 ----
    def _embed_openai(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(input=batch, model=self.model_name)
            embeddings.extend([item.embedding for item in response.data])

        return np.array(embeddings)

    def _embed_local(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, show_progress_bar=len(texts) > 100)

    def compute_embeddings(self, force=False):
        if self.embeddings is not None and not force:
            print("Using cached embeddings")
            return self.embeddings

        print(f"Computing embeddings ({self.model_name})...")
        texts = [self._create_text(p) for p in self.papers]

        if self.model_type == "openai":
            self.embeddings = self._embed_openai(texts)
        else:
            self.embeddings = self._embed_local(texts)

        print(f"Computed: {self.embeddings.shape}")
        self._save_cache()
        return self.embeddings

    def search(self, examples=None, query=None, top_k=100):
        if self.embeddings is None:
            self.compute_embeddings()

        # 生成查询向量
        if examples:
            texts = [self._create_text(ex) for ex in examples]
            embs = self._embed_openai(texts) if self.model_type == "openai" else self._embed_local(texts)
            query_emb = np.mean(embs, axis=0).reshape(1, -1)
        elif query:
            if self.model_type == "openai":
                query_emb = self._embed_openai(query).reshape(1, -1)
            else:
                query_emb = self._embed_local(query).reshape(1, -1)
        else:
            raise ValueError("Provide either examples or query")

        similarities = cosine_similarity(query_emb, self.embeddings)[0].astype('float32')

        # —— 新增：按期刊权重重排 —— #
        weights = np.array([self._journal_weight(p.get('journal')) for p in self.papers], dtype='float32')
        scores = similarities * weights  # 或者 weights ** alpha（已在 _journal_weight 中处理）

        top_indices = np.argsort(scores)[::-1][:top_k]

        return [{
            'paper': self.papers[idx],
            'score': float(scores[idx]),          # 加权后的最终得分（用于排序）
            'similarity': float(similarities[idx]),  # 原始语义相似度
            'journal_weight': float(weights[idx]),
            'journal': self.papers[idx].get('journal')
        } for idx in top_indices]

    def display(self, results, n=10):
        print(f"\n{'='*80}")
        print(f"Top {len(results)} Results (showing {min(n, len(results))})")
        print(f"{'='*80}\n")

        for i, result in enumerate(results[:n], 1):
            paper = result['paper']
            score = result.get('score', result.get('similarity', 0.0))
            sim = result.get('similarity', score)
            jw = result.get('journal_weight', 1.0)
            title = paper.get('title', 'N/A')
            print(f"{i}. [score={score:.4f} | sim={sim:.4f} | w={jw:.2f}] {title}")
            print(f"   #{paper.get('number', 'N/A')} | {paper.get('journal', 'N/A')}")
            print(f"   {paper.get('forum_url', 'N/A')}\n")

    def save(self, results, output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'model': self.model_name,
                'total': len(results),
                'results': results
            }, f, ensure_ascii=False, indent=2)
        print(f"Saved to {output_file}")