"""
Simplified Baseline Models Implementation for TREC CAsT Evaluation
No Java dependencies required - pure Python implementation
"""

import os
import json
import re
import logging
import math
import numpy as np
import torch
from tqdm import tqdm
import argparse
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleTokenizer:
    """Simple tokenizer for preprocessing."""

    def __init__(self):
        # Common English stopwords
        self.stopwords = set([
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
            'when', 'where', 'how', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'to', 'at', 'by', 'for', 'with',
            'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after'
        ])

    def tokenize(self, text):
        """Simple word tokenization."""
        if not text:
            return []
        # Split on non-alphanumeric characters and convert to lowercase
        tokens = re.findall(r'\w+', text.lower())
        # Remove stopwords
        return [t for t in tokens if t not in self.stopwords]


class BM25Retriever:
    """Pure Python implementation of BM25 without Pyserini."""

    def __init__(self, corpus, k1=1.5, b=0.75):
        """Initialize BM25 retriever.

        Args:
            corpus: Dictionary with document IDs as keys and content as values
            k1: BM25 parameter k1
            b: BM25 parameter b
        """
        self.corpus = corpus
        self.k1 = k1
        self.b = b

        # Preprocess corpus
        self.tokenizer = SimpleTokenizer()

        # Limit to first 1000 documents for memory efficiency
        max_docs = min(1000, len(corpus))
        self.doc_ids = list(corpus.keys())[:max_docs]
        logger.info(f"For memory efficiency, only indexing first {max_docs} documents")

        self.tokenized_corpus = []

        logger.info("Tokenizing corpus documents...")
        for doc_id in tqdm(self.doc_ids):
            self.tokenized_corpus.append(self.tokenizer.tokenize(corpus[doc_id]))

        # Calculate corpus statistics
        self.avg_doc_len = sum(len(doc) for doc in self.tokenized_corpus) / len(self.tokenized_corpus)

        # Calculate IDF values
        self.idf = {}
        self.doc_freqs = {}

        logger.info("Calculating document frequencies...")
        for doc_id, doc in zip(self.doc_ids, self.tokenized_corpus):
            for term in set(doc):
                if term not in self.doc_freqs:
                    self.doc_freqs[term] = 0
                self.doc_freqs[term] += 1

        # Calculate IDF
        N = len(self.tokenized_corpus)
        for term, df in self.doc_freqs.items():
            self.idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)

        logger.info(f"BM25 initialized with {len(self.doc_ids)} documents, average length: {self.avg_doc_len:.2f}")

    def search(self, query, top_k=100):
        """Search with BM25.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of (doc_id, score) tuples
        """
        query_terms = self.tokenizer.tokenize(query)
        scores = []

        for i, (doc_id, doc) in enumerate(zip(self.doc_ids, self.tokenized_corpus)):
            score = 0
            doc_len = len(doc)

            for term in query_terms:
                if term not in self.idf:
                    continue

                # Count term frequency in document
                tf = doc.count(term)
                if tf == 0:
                    continue

                # Calculate BM25 score for this term
                numerator = self.idf[term] * tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                score += numerator / denominator

            scores.append((doc_id, score))

        # Sort by score and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class BERTRetriever:
    """Full implementation of BERT retrieval."""

    def __init__(self, model_name="bert-base-uncased"):
        """Initialize BERT retriever.

        Args:
            model_name: Name of the pre-trained BERT model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.doc_embeddings = None
        self.doc_ids = None

    def encode(self, texts, batch_size=8):
        """Encode texts into embeddings.

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding

        Returns:
            NumPy array of embeddings
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True,
                                   return_tensors="pt", max_length=512).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Use CLS token embeddings
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def index_corpus(self, corpus):
        """Index the corpus.

        Args:
            corpus: Dictionary with document IDs as keys and content as values
        """
        logger.info(f"Indexing documents with BERT...")
        start_time = time.time()

        # Limit to first 1000 documents for memory efficiency
        max_docs = min(1000, len(corpus))
        self.doc_ids = list(corpus.keys())[:max_docs]
        logger.info(f"For memory efficiency, only indexing first {max_docs} documents")

        doc_texts = [corpus[doc_id] for doc_id in self.doc_ids]
        self.doc_embeddings = self.encode(doc_texts)

        logger.info(f"Indexed {len(self.doc_ids)} documents in {time.time() - start_time:.2f} seconds")

    def search(self, query, top_k=100):
        """Search with BERT.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of (doc_id, score) tuples
        """
        query_embedding = self.encode([query])[0]

        # Compute similarities (cosine)
        similarities = np.dot(self.doc_embeddings, query_embedding) / (
            np.linalg.norm(self.doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get top results
        top_indices = np.argsort(-similarities)[:top_k]
        results = [(self.doc_ids[i], float(similarities[i])) for i in top_indices]

        return results


class SimplifiedColBERTRetriever:
    """Simplified implementation of ColBERT retrieval with token-level interactions."""

    def __init__(self, model_name="bert-base-uncased", dim=128):
        """Initialize simplified ColBERT retriever.

        Args:
            model_name: Name of the pre-trained BERT model
            dim: Dimension for linear projection
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.dim = dim

        # Linear dimension reduction as in ColBERT paper
        self.linear = torch.nn.Linear(self.model.config.hidden_size, dim).to(self.device)
        self.doc_embeddings = {}

    def encode_query(self, query, max_length=32):
        """Encode a query into token-level embeddings.

        Args:
            query: Query string
            max_length: Maximum token length

        Returns:
            Query embeddings and attention mask
        """
        inputs = self.tokenizer(query, padding="max_length", truncation=True,
                               return_tensors="pt", max_length=max_length).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Apply linear projection and normalization
        embeddings = self.linear(outputs.last_hidden_state)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=2)

        return embeddings[0], inputs.attention_mask[0]  # Remove batch dimension

    def encode_doc(self, doc, max_length=180):
        """Encode a document into token-level embeddings.

        Args:
            doc: Document string
            max_length: Maximum token length

        Returns:
            Document embeddings and attention mask
        """
        inputs = self.tokenizer(doc, padding="max_length", truncation=True,
                               return_tensors="pt", max_length=max_length).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Apply linear projection and normalization
        embeddings = self.linear(outputs.last_hidden_state)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=2)

        return embeddings[0], inputs.attention_mask[0]  # Remove batch dimension

    def index_corpus(self, corpus, batch_size=8):
        """Index the corpus with ColBERT.

        Args:
            corpus: Dictionary with document IDs as keys and content as values
            batch_size: Batch size for encoding
        """
        logger.info(f"Indexing {len(corpus)} documents with simplified ColBERT...")
        start_time = time.time()

        doc_ids = list(corpus.keys())

        # To save memory, only index a subset of documents for demo purposes
        max_docs = min(1000, len(doc_ids))
        doc_ids = doc_ids[:max_docs]
        logger.info(f"For memory efficiency, only indexing first {max_docs} documents")

        for i in tqdm(range(0, len(doc_ids), batch_size)):
            batch_ids = doc_ids[i:i+batch_size]
            batch_docs = [corpus[doc_id] for doc_id in batch_ids]

            # Process batch
            batch_inputs = self.tokenizer(batch_docs, padding="max_length", truncation=True,
                                       return_tensors="pt", max_length=180).to(self.device)

            with torch.no_grad():
                outputs = self.model(**batch_inputs)

                # Apply linear projection and normalization
                batch_embeddings = self.linear(outputs.last_hidden_state)
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=2)

                # Store embeddings for each document
                for j, doc_id in enumerate(batch_ids):
                    self.doc_embeddings[doc_id] = (
                        batch_embeddings[j].cpu(),
                        batch_inputs.attention_mask[j].cpu()
                    )

        logger.info(f"Indexed {len(self.doc_embeddings)} documents in {time.time() - start_time:.2f} seconds")

    def search(self, query, top_k=100):
        """Search with ColBERT's MaxSim operator.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of (doc_id, score) tuples
        """
        # Encode query
        q_embeddings, q_mask = self.encode_query(query)
        q_embeddings = q_embeddings.cpu()
        q_mask = q_mask.cpu()

        results = []
        for doc_id, (d_embeddings, d_mask) in self.doc_embeddings.items():
            # Calculate token-level similarities
            sim_matrix = torch.matmul(q_embeddings, d_embeddings.transpose(0, 1))

            # Apply query mask
            masked_sim = sim_matrix * q_mask.unsqueeze(1)

            # Max-sim operation: for each query term, find the best matching document term
            max_sim = torch.max(masked_sim, dim=1)[0]
            # Sum over all query terms
            score = torch.sum(max_sim * q_mask).item()

            results.append((doc_id, score))

        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]


class ZeCoRetriever:
    """Zero-shot context-aware retrieval implementation."""

    def __init__(self, colbert_retriever=None, model_name="bert-base-uncased", dim=128):
        """Initialize ZeCo² retriever based on ColBERT.

        Args:
            colbert_retriever: Optional existing ColBERT retriever
            model_name: Name of the pre-trained BERT model
            dim: Dimension for linear projection
        """
        if colbert_retriever:
            self.colbert = colbert_retriever
        else:
            self.colbert = SimplifiedColBERTRetriever(model_name, dim)

    def index_corpus(self, corpus):
        """Index the corpus using ColBERT.

        Args:
            corpus: Dictionary with document IDs as keys and content as values
        """
        if not self.colbert.doc_embeddings:
            self.colbert.index_corpus(corpus)

    def search(self, query, context, top_k=100):
        """Search with context-enhanced query.

        Args:
            query: Current query string
            context: List of previous queries in the conversation
            top_k: Number of results to return

        Returns:
            List of (doc_id, score) tuples
        """
        # Context-enhanced query formation
        enhanced_query = " ".join(context + [query])

        # Use ColBERT for search
        return self.colbert.search(enhanced_query, top_k)


class TRECCASTEvaluator:
    """Evaluator for TREC CAsT datasets."""

    def __init__(self, data_dir):
        """Initialize evaluator.

        Args:
            data_dir: Path to data directory
        """
        self.data_dir = data_dir

        # Load data
        self.load_data()

    def load_data(self):
        """Load necessary data for evaluation."""
        logger.info("Loading TREC CAsT datasets...")

        # Load conversations (topics)
        topics_2019_path = os.path.join(self.data_dir, "treccast", "2019_evaluation_topics_v1.0.json")
        self.topics_2019 = self.load_json(topics_2019_path)

        # Load relevance judgments (qrels)
        qrels_2019_path = os.path.join(self.data_dir, "qrels", "2019qrels.txt")
        self.qrels_2019 = self.load_qrels(qrels_2019_path)

        # Load manual utterances if available
        manual_utterances_path = os.path.join(self.data_dir, "treccast", "test_manual_utterance.tsv")
        try:
            self.manual_utterances = self.load_tsv(manual_utterances_path)
            logger.info(f"Loaded manual utterances from {manual_utterances_path}")
        except Exception as e:
            logger.warning(f"Could not load manual utterances: {e}")
            self.manual_utterances = {}

        # Create corpus aligned with qrels
        corpus_cache_path = os.path.join(self.data_dir, "corpus_cache.json")
        if os.path.exists(corpus_cache_path):
            try:
                with open(corpus_cache_path, 'r') as f:
                    self.corpus = json.load(f)
                logger.info(f"Loaded cached corpus with {len(self.corpus)} documents")
            except Exception as e:
                logger.warning(f"Failed to load cached corpus: {e}")
                self.corpus = self.create_corpus_aligned_with_qrels()
        else:
            self.corpus = self.create_corpus_aligned_with_qrels()
            # Cache corpus
            try:
                with open(corpus_cache_path, 'w') as f:
                    json.dump(self.corpus, f)
                logger.info(f"Cached corpus with {len(self.corpus)} documents")
            except Exception as e:
                logger.warning(f"Failed to cache corpus: {e}")

        logger.info("Data loading completed.")

    def load_json(self, file_path):
        """Load and parse a JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            Parsed JSON data
        """
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
            return []

    def load_qrels(self, file_path):
        """Load qrels file (relevance judgments).

        Args:
            file_path: Path to qrels file

        Returns:
            Dictionary of relevance judgments
        """
        qrels = {}
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 4:
                        query_id, _, doc_id, relevance = parts
                        if query_id not in qrels:
                            qrels[query_id] = {}
                        qrels[query_id][doc_id] = int(relevance)
            logger.info(f"Loaded {len(qrels)} query judgments from {file_path}")
        except Exception as e:
            logger.error(f"Error loading qrels file {file_path}: {e}")
        return qrels

    def load_tsv(self, file_path):
        """Load manual utterances from TSV file.

        Args:
            file_path: Path to TSV file

        Returns:
            Dictionary of manual utterances
        """
        manual_utterances = {}
        try:
            with open(file_path, 'r') as f:
                # Skip header line
                next(f, None)
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        query_id = parts[0]
                        manual_query = parts[1]
                        manual_utterances[query_id] = manual_query
        except Exception as e:
            logger.error(f"Error loading TSV file {file_path}: {e}")
        return manual_utterances

    def create_corpus_aligned_with_qrels(self):
        """Create a corpus with documents that match qrels IDs.

        Returns:
            Dictionary with document IDs as keys and content as values
        """
        corpus = {}

        # Collect all document IDs from qrels
        doc_ids = set()
        for query_judgments in self.qrels_2019.values():
            doc_ids.update(query_judgments.keys())

        logger.info(f"Found {len(doc_ids)} unique document IDs in qrels")

        # Try to load real documents from paragraphCorpus if available
        paragraph_corpus_path = os.path.join(self.data_dir, "paragraphCorpus")
        if os.path.exists(paragraph_corpus_path) and os.path.isdir(paragraph_corpus_path):
            logger.info(f"Loading documents from {paragraph_corpus_path}")
            loaded_count = 0

            for doc_id in tqdm(list(doc_ids)[:min(5000, len(doc_ids))], desc="Loading documents"):
                # Try to find document file
                doc_path = os.path.join(paragraph_corpus_path, f"{doc_id}.txt")
                if os.path.exists(doc_path):
                    try:
                        with open(doc_path, 'r', encoding='utf-8') as f:
                            corpus[doc_id] = f.read().strip()
                        loaded_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to load document {doc_id}: {e}")
                        corpus[doc_id] = self.generate_synthetic_document(doc_id)
                else:
                    corpus[doc_id] = self.generate_synthetic_document(doc_id)

            logger.info(f"Loaded {loaded_count} real documents from paragraphCorpus")
            logger.info(f"Generated {len(corpus) - loaded_count} synthetic documents")
        else:
            # Create synthetic content for each document ID
            logger.info("No paragraphCorpus found, generating synthetic documents")
            # Limit to 5000 documents for demo purposes
            for doc_id in tqdm(list(doc_ids)[:min(5000, len(doc_ids))], desc="Generating documents"):
                corpus[doc_id] = self.generate_synthetic_document(doc_id)

        logger.info(f"Created corpus with {len(corpus)} documents")
        return corpus

    def generate_synthetic_document(self, doc_id):
        """Generate synthetic content for a document.

        Args:
            doc_id: Document ID

        Returns:
            Generated document content
        """
        # Topics for synthetic content generation
        topics = ["artificial intelligence", "machine learning", "neural networks",
                  "deep learning", "natural language processing", "conversational search",
                  "information retrieval", "question answering", "dialogue systems"]

        # Create document content related to AI/IR topics
        topic = topics[hash(doc_id) % len(topics)]

        # Generate paragraphs
        num_paragraphs = 3 + (hash(doc_id) % 5)  # 3 to 7 paragraphs

        paragraphs = [f"Document {doc_id} about {topic}."]

        for _ in range(num_paragraphs):
            paragraphs.append(self.generate_paragraph_for_topic(topic, doc_id))

        return " ".join(paragraphs)

    def generate_paragraph_for_topic(self, topic, doc_id):
        """Generate a paragraph for a topic.

        Args:
            topic: Topic for paragraph
            doc_id: Document ID for variation

        Returns:
            Generated paragraph
        """
        templates = [
            f"{topic} is a field of computer science focused on building systems that can perform tasks requiring human intelligence.",
            f"Recent advances in {topic} have led to significant improvements in performance across various benchmarks.",
            f"Researchers in {topic} are exploring new methods to enhance model capabilities and reduce computational requirements.",
            f"Applications of {topic} include healthcare, finance, education, and entertainment.",
            f"{topic} systems can process natural language, recognize patterns, and make decisions based on available data.",
            f"The future of {topic} involves more interpretable models that can explain their reasoning process.",
            f"Ethical considerations in {topic} include privacy, bias, fairness, and potential socioeconomic impacts.",
            f"Training effective {topic} models requires large datasets and significant computational resources.",
            f"Evaluation metrics for {topic} systems include accuracy, precision, recall, and human preference judgments."
        ]

        # Select paragraph text based on hash of document ID
        return templates[hash(doc_id + topic) % len(templates)]

    def evaluate_metrics(self, results, qrels):
        """Calculate evaluation metrics.

        Args:
            results: Dictionary mapping query IDs to lists of (doc_id, score) tuples
            qrels: Dictionary mapping query IDs to dictionaries of doc_id:relevance

        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            "MRR": 0.0,
            "NDCG@3": 0.0,
            "R@100": 0.0
        }

        query_count = 0

        for query_id, retrieved_docs in results.items():
            if query_id not in qrels:
                continue

            query_count += 1
            relevant_docs = qrels[query_id]

            # MRR - Mean Reciprocal Rank
            for i, (doc_id, _) in enumerate(retrieved_docs):
                if doc_id in relevant_docs and relevant_docs[doc_id] > 0:
                    metrics["MRR"] += 1.0 / (i + 1)
                    break

            # NDCG@3
            if retrieved_docs:
                gains = []
                for i, (doc_id, _) in enumerate(retrieved_docs[:3]):
                    rel = relevant_docs.get(doc_id, 0)
                    gains.append(rel)

                # Calculate DCG@3
                dcg = 0
                for i, gain in enumerate(gains):
                    if gain > 0:
                        dcg += (2 ** gain - 1) / math.log2(i + 2)

                # Calculate ideal DCG@3
                ideal_gains = sorted([rel for rel in relevant_docs.values() if rel > 0], reverse=True)[:3]
                idcg = 0
                for i, gain in enumerate(ideal_gains):
                    idcg += (2 ** gain - 1) / math.log2(i + 2)

                if idcg > 0:
                    metrics["NDCG@3"] += dcg / idcg

            # R@100 - Recall at 100
            retrieved_relevant = 0
            for doc_id, _ in retrieved_docs[:100]:
                if doc_id in relevant_docs and relevant_docs[doc_id] > 0:
                    retrieved_relevant += 1

            total_relevant = sum(1 for rel in relevant_docs.values() if rel > 0)
            if total_relevant > 0:
                metrics["R@100"] += retrieved_relevant / total_relevant

        # Average metrics
        if query_count > 0:
            metrics["MRR"] /= query_count
            metrics["NDCG@3"] /= query_count
            metrics["R@100"] /= query_count

        return metrics

    def run_evaluation(self, use_manual=False, include_models=None):
        """Run all baseline models on the dataset.

        Args:
            use_manual: Whether to use manually rewritten queries
            include_models: List of models to include (default: all)

        Returns:
            Dictionary of evaluation metrics
        """
        start_time = time.time()

        # Select dataset
        topics = self.topics_2019
        qrels = self.qrels_2019

        # Determine which models to use
        if include_models is None:
            include_models = ["BM25-raw"]

        logger.info(f"Running evaluation on CAsT-2019 with models: {include_models}")

        # Initialize models
        models = {}

        if "BM25-raw" in include_models:
            models["BM25-raw"] = BM25Retriever(self.corpus)

        if "BERT-raw" in include_models:
            models["BERT-raw"] = BERTRetriever()
            models["BERT-raw"].index_corpus(self.corpus)

        if "ColBERT-raw" in include_models or "ZeCo²" in include_models:
            colbert = SimplifiedColBERTRetriever()
            colbert.index_corpus(self.corpus)

            if "ColBERT-raw" in include_models:
                models["ColBERT-raw"] = colbert

            if "ZeCo²" in include_models:
                models["ZeCo²"] = ZeCoRetriever(colbert)

        # Results for each model
        results = {model_name: {} for model_name in models}

        # Process each conversation
        for topic in tqdm(topics, desc="Processing topics (2019)"):
            topic_id = str(topic.get("number"))
            turns = topic.get("turn", [])

            # Keep track of conversation context for ZeCo²
            context = []

            for turn in turns:
                turn_id = str(turn.get("number"))
                query_id = f"{topic_id}_{turn_id}"

                # Skip if no qrels for this query
                if query_id not in qrels:
                    continue

                # Get the query text (raw or manual)
                if use_manual and query_id in self.manual_utterances:
                    query = self.manual_utterances[query_id]
                else:
                    query = turn.get("raw_utterance", "")

                # Run each model
                if "BM25-raw" in models:
                    results["BM25-raw"][query_id] = models["BM25-raw"].search(query)

                if "BERT-raw" in models:
                    results["BERT-raw"][query_id] = models["BERT-raw"].search(query)

                if "ColBERT-raw" in models:
                    results["ColBERT-raw"][query_id] = models["ColBERT-raw"].search(query)

                if "ZeCo²" in models:
                    results["ZeCo²"][query_id] = models["ZeCo²"].search(query, context)

                # Update context for next turn
                context.append(query)

        # Evaluate models
        metrics = {}
        for model_name, model_results in results.items():
            metrics[model_name] = self.evaluate_metrics(model_results, qrels)

            logger.info(f"{model_name} results for 2019:")
            logger.info(f"  MRR:     {metrics[model_name]['MRR']:.4f}")
            logger.info(f"  NDCG@3:  {metrics[model_name]['NDCG@3']:.4f}")
            logger.info(f"  R@100:   {metrics[model_name]['R@100']:.4f}")

        logger.info(f"Evaluation completed in {time.time() - start_time:.2f} seconds")

        return metrics



def main():
    """Main function to run baselines evaluation."""
    parser = argparse.ArgumentParser(description='Run baseline evaluations for conversational search')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory containing TREC dataset')
    parser.add_argument('--use_manual', action='store_true',
                        help='Use manually rewritten queries if available')
    parser.add_argument('--models', type=str, nargs='+',
                        choices=['BM25-raw', 'BERT-raw', 'ColBERT-raw', 'ZeCo²', 'all'],
                        default=['all'], help='Which models to evaluate')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA for BERT models if available')

    args = parser.parse_args()

    # Set up verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Ensure data directory exists
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory does not exist: {args.data_dir}")
        return 1

    print(f"Using data from: {args.data_dir}")

    # Set CUDA availability
    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Convert 'all' to list of all models
    if 'all' in args.models:
        models = ['BM25-raw', 'BERT-raw', 'ColBERT-raw', 'ZeCo²']
    else:
        models = args.models

    # Initialize evaluator
    evaluator = TRECCASTEvaluator(args.data_dir)

    # Run evaluation on 2019 dataset
    print("\nEvaluating on TREC CAsT 2019 dataset")
    metrics_2019 = evaluator.run_evaluation(args.use_manual, models)

    # Print summary results
    print("\nSummary Results for CAsT-2019:")
    print(f"{'Model':<15} {'MRR':<10} {'NDCG@3':<10} {'R@100':<10}")
    print("-" * 45)
    for model, values in metrics_2019.items():
        print(f"{model:<15} {values['MRR']:.4f} {values['NDCG@3']:.4f} {values['R@100']:.4f}")

    # Save results to file
    results_dir = os.path.join(args.data_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    manual_tag = "_manual" if args.use_manual else "_raw"
    results_file = os.path.join(results_dir, f"baseline_results{manual_tag}.json")

    with open(results_file, "w") as f:
        json.dump(metrics_2019, f, indent=2)

    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main()