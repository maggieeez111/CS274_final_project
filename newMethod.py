"""
Enhanced implementation of Zero-shot Response-Aware (ZeRA) methods
for conversational search evaluation.
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
import sys
from transformers import AutoTokenizer, AutoModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import required classes from baseline_implementation4.py
# This assumes the baseline file is in the same directory
sys.path.append('.')
from oldBaseline import SimplifiedColBERTRetriever, BM25Retriever, TRECCASTEvaluator


class ZeRARetriever:
    """Improved Zero-shot Response-Aware retriever implementation."""

    def __init__(self, colbert_retriever=None, model_name="bert-base-uncased", dim=128):
        """Initialize Improved ZeRA retriever based on ColBERT."""
        if colbert_retriever:
            self.colbert = colbert_retriever
        else:
            self.colbert = SimplifiedColBERTRetriever(model_name, dim)

        # Adjusted parameters based on analysis of results
        self.alpha = 0.65  # Increased weight for term-level expansion (BM25 performed best)
        self.beta = 0.25   # Decreased weight for sentence-level
        self.theta = 0.10  # Lower threshold for passage-level
        self.tau = 0.15    # Lower threshold for term selection

        # Initialize corpus
        self.corpus = None

    def index_corpus(self, corpus):
        """Index the corpus using ColBERT."""
        if not self.colbert.doc_embeddings:
            self.colbert.index_corpus(corpus)
        self.corpus = corpus

    def search(self, query, context, top_k=100):
        """Search with improved query expansion strategy."""
        # Ensure corpus is available
        if self.corpus is None and hasattr(self.colbert, 'corpus'):
            self.corpus = self.colbert.corpus

        if self.corpus is None:
            logger.error("No corpus available. Please call index_corpus first.")
            return []

        # 1. First, get results from basic BM25 for high recall
        bm25_results = self._get_bm25_results(query, top_k*2)

        # 2. Get candidate documents
        candidate_docs = {}
        for doc_id, _ in bm25_results[:top_k*3]:  # Use larger pool
            if doc_id in self.corpus:
                candidate_docs[doc_id] = self.corpus[doc_id]

        # 3. Create expanded query using BM25-inspired term expansion
        term_expanded_query = self._get_term_expanded_query(query, candidate_docs)

        # 4. Create context-aware query based on conversation history
        contextual_query = query
        if context:
            # For first turn queries, add first query context
            first_query = context[0]
            # For multi-turn queries, add most recent context
            if len(context) > 1:
                recent_query = context[-1]
                similarity = self._calculate_similarity(query, recent_query)

                if similarity > self.theta:
                    contextual_query = recent_query + " " + query
                else:
                    contextual_query = first_query + " " + query
            else:
                contextual_query = first_query + " " + query

        # 5. Get results from all approaches
        bm25_scores = {doc_id: score for doc_id, score in bm25_results}
        term_results = self.colbert.search(term_expanded_query, top_k*2)
        context_results = self.colbert.search(contextual_query, top_k*2)

        # 6. Combine results with weighted fusion
        combined_results = self._combine_results(
            bm25_scores, term_results, context_results, top_k
        )

        return combined_results

    def _get_bm25_results(self, query, top_k=100):
        """Get BM25 results with optimized parameters."""
        corpus_subset = {}
        # Use all available documents
        for doc_id in list(self.colbert.doc_embeddings.keys()):
            if doc_id in self.corpus:
                corpus_subset[doc_id] = self.corpus[doc_id]

        if not corpus_subset:
            logger.warning("No corpus available for BM25")
            return []

        # Use improved BM25 parameters
        bm25 = BM25Retriever(corpus_subset, k1=1.8, b=0.65)  # Higher k1, lower b for better recall
        return bm25.search(query, top_k=top_k)

    def _get_term_expanded_query(self, query, candidate_docs):
        """Get improved term expansion."""
        query_terms = set(re.findall(r'\w+', query.lower()))

        # Calculate document scores
        doc_scores = {}
        for doc_id, doc_text in candidate_docs.items():
            # Simple BM25-like scoring for efficiency
            doc_tokens = set(re.findall(r'\w+', doc_text.lower()))
            overlap = len(query_terms.intersection(doc_tokens))
            doc_scores[doc_id] = overlap / max(1, len(query_terms))

        # Get top documents for expansion
        top_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:5]

        # Extract terms from top documents
        expansion_terms = set()
        for doc_id, _ in top_docs:
            doc_text = candidate_docs[doc_id]
            doc_tokens = re.findall(r'\w+', doc_text.lower())

            # Filter terms
            for term in doc_tokens:
                if term not in query_terms and len(term) > 3:
                    expansion_terms.add(term)

        # Limit expansion terms
        expansion_terms = list(expansion_terms)[:10]

        # Create expanded query
        expanded_query = query
        if expansion_terms:
            expanded_query += " " + " ".join(expansion_terms)

        return expanded_query

    def _calculate_similarity(self, text1, text2):
        """Improved similarity calculation using token overlap."""
        tokens1 = set(re.findall(r'\w+', text1.lower()))
        tokens2 = set(re.findall(r'\w+', text2.lower()))

        if not tokens1 or not tokens2:
            return 0.0

        # Jaccard similarity
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))

        return intersection / union

    def _combine_results(self, bm25_scores, term_results, context_results, top_k):
        """Combine results with emphasis on BM25."""
        scores = {}

        # Process BM25 results (highest weight)
        for doc_id, score in bm25_scores.items():
            scores[doc_id] = self.alpha * score

        # Normalize ColBERT scores for better fusion
        max_term_score = max([score for _, score in term_results]) if term_results else 1.0
        max_context_score = max([score for _, score in context_results]) if context_results else 1.0

        # Process term expansion results
        for doc_id, score in term_results:
            if doc_id not in scores:
                scores[doc_id] = 0
            # Normalize score
            scores[doc_id] += (1.0 - self.alpha - self.beta) * (score / max_term_score)

        # Process context results
        for doc_id, score in context_results:
            if doc_id not in scores:
                scores[doc_id] = 0
            # Normalize score
            scores[doc_id] += self.beta * (score / max_context_score)

        # Convert to list and sort
        results = [(doc_id, score) for doc_id, score in scores.items()]
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]


class ZeRADTRetriever:

    def __init__(self, colbert_retriever=None, model_name="bert-base-uncased", dim=128):
        """Initialize Improved ZeRA-DT retriever."""
        if colbert_retriever:
            self.colbert = colbert_retriever
        else:
            self.colbert = SimplifiedColBERTRetriever(model_name, dim)

        # Base parameters aligned with successful ZeRA
        self.base_alpha = 0.65  # Higher weight for term-level/BM25
        self.base_beta = 0.25   # Lower weight for context
        self.corpus = None

    def index_corpus(self, corpus):
        """Index the corpus using ColBERT."""
        if not self.colbert.doc_embeddings:
            self.colbert.index_corpus(corpus)
        self.corpus = corpus

    def search(self, query, context, turn_number, top_k=100):
        """Search with simplified dynamic parameter approach."""
        # Ensure corpus is available
        if self.corpus is None and hasattr(self.colbert, 'corpus'):
            self.corpus = self.colbert.corpus

        if self.corpus is None:
            logger.error("No corpus available. Please call index_corpus first.")
            return []

        # 1. Get BM25 results directly (successful in ZeRA)
        bm25_results = self._get_bm25_results(query, top_k*2)

        # 2. Adjust weights based on turn number - SIMPLIFIED
        # For early turns: rely more on current query
        # For later turns: incorporate more context
        alpha = self.base_alpha
        beta = self.base_beta

        if turn_number > 2:
            # Gradually reduce alpha, increase beta for later turns
            turn_factor = min(0.3, (turn_number - 2) * 0.1)
            alpha = max(0.4, self.base_alpha - turn_factor)
            beta = min(0.5, self.base_beta + turn_factor)

        # 3. Get candidate documents for expansion
        candidate_docs = {}
        for doc_id, _ in bm25_results[:top_k*3]:
            if doc_id in self.corpus:
                candidate_docs[doc_id] = self.corpus[doc_id]

        # 4. Create query expansions based on turn number
        # Term expansion for all turns
        term_expanded_query = self._get_term_expanded_query(query, candidate_docs)

        # Context-aware query based on turn number
        context_query = query

        if context and turn_number > 1:
            # For turn 2: only use first query for context
            if turn_number == 2:
                context_query = context[0] + " " + query
            # For turn 3+: use first query + most recent query
            elif turn_number >= 3:
                if len(context) >= 2:
                    context_query = context[0] + " " + context[-1] + " " + query
                else:
                    context_query = context[0] + " " + query

        # 5. Get results from all approaches
        bm25_scores = {doc_id: score for doc_id, score in bm25_results}
        term_results = self.colbert.search(term_expanded_query, top_k*2)
        context_results = self.colbert.search(context_query, top_k*2)

        # 6. Combine results with dynamic weights
        combined_results = self._combine_results(
            bm25_scores, term_results, context_results,
            alpha, beta, turn_number, top_k
        )
        return combined_results

    def _get_bm25_results(self, query, top_k=100):
        """Get BM25 results with optimized parameters."""
        corpus_subset = {}
        for doc_id in list(self.colbert.doc_embeddings.keys()):
            if doc_id in self.corpus:
                corpus_subset[doc_id] = self.corpus[doc_id]

        if not corpus_subset:
            return []

        # Use improved BM25 parameters
        bm25 = BM25Retriever(corpus_subset, k1=1.8, b=0.65)
        return bm25.search(query, top_k=top_k)

    def _get_term_expanded_query(self, query, candidate_docs):
        """Simplified term expansion focused on effectiveness."""
        query_terms = set(re.findall(r'\w+', query.lower()))

        # Simple frequency-based term selection
        term_counts = {}

        for doc_id, doc_text in candidate_docs.items():
            doc_tokens = re.findall(r'\w+', doc_text.lower())

            for term in doc_tokens:
                if term not in query_terms and len(term) > 3:
                    if term not in term_counts:
                        term_counts[term] = 0
                    term_counts[term] += 1

        # Select top 5-8 expansion terms
        expansion_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:8]

        # Create expanded query
        expanded_query = query
        if expansion_terms:
            expanded_query += " " + " ".join(term for term, _ in expansion_terms)

        return expanded_query

    def _combine_results(self, bm25_scores, term_results, context_results, alpha, beta, turn_number, top_k):
        """Combine results with turn-sensitive weighting."""
        scores = {}

        # Process BM25 results (highest weight)
        for doc_id, score in bm25_scores.items():
            scores[doc_id] = alpha * score

        # Normalize ColBERT scores
        max_term_score = max([score for _, score in term_results]) if term_results else 1.0
        max_context_score = max([score for _, score in context_results]) if context_results else 1.0

        # Process term expansion results
        term_weight = 1.0 - alpha - beta
        for doc_id, score in term_results:
            if doc_id not in scores:
                scores[doc_id] = 0
            # Normalize score
            scores[doc_id] += term_weight * (score / max_term_score)

        # Process context results with turn-based scaling
        for doc_id, score in context_results:
            if doc_id not in scores:
                scores[doc_id] = 0
            # For later turns, give more weight to context matches
            turn_factor = min(1.0, turn_number / 3.0)
            context_weight = beta * turn_factor
            scores[doc_id] += context_weight * (score / max_context_score)

        # Final ranking - boost documents that appear in multiple result sets
        result_sets = [
            set(bm25_scores.keys()),
            set(doc_id for doc_id, _ in term_results),
            set(doc_id for doc_id, _ in context_results)
        ]

        for doc_id in scores:
            # Count appearances in result sets
            appearances = sum(1 for result_set in result_sets if doc_id in result_set)
            if appearances > 1:
                # Modest boost for consensus
                scores[doc_id] *= (1.0 + 0.05 * (appearances - 1))

        # Convert to list and sort by score
        results = [(doc_id, score) for doc_id, score in scores.items()]
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]


def run_evaluation(data_dir, use_manual=False, models=None, verbose=False, cuda=False):
    """Run evaluation for ZeRA and ZeRA-DT models.

    Args:
        data_dir: Path to data directory
        use_manual: Whether to use manually rewritten queries
        models: List of models to evaluate (ZeRA, ZeRA-DT, or both)
        verbose: Enable verbose logging
        cuda: Use CUDA if available

    Returns:
        Evaluation results
    """
    if models is None:
        models = ["ZeRA", "ZeRA-DT"]

    # Set up CUDA
    if cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Initialize evaluator
    evaluator = TRECCASTEvaluator(data_dir)

    # Run evaluation for 2019
    print(f"\nEvaluating on TREC CAsT 2019 dataset")

    # Select dataset
    topics = evaluator.topics_2019
    qrels = evaluator.qrels_2019

    # Initialize models
    model_instances = {}

    if "ZeRA" in models:
        colbert = SimplifiedColBERTRetriever()
        colbert.index_corpus(evaluator.corpus)
        zera = ZeRARetriever(colbert)
        zera.index_corpus(evaluator.corpus)  # Make sure to index the corpus
        model_instances["ZeRA"] = zera

    if "ZeRA-DT" in models:
        colbert = SimplifiedColBERTRetriever()
        colbert.index_corpus(evaluator.corpus)
        zera_dt = ZeRADTRetriever(colbert)
        zera_dt.index_corpus(evaluator.corpus)  # Make sure to index the corpus
        model_instances["ZeRA-DT"] = zera_dt

    # Results for each model
    results = {model_name: {} for model_name in model_instances}

    # Process each conversation
    for topic in tqdm(topics, desc="Processing topics (2019)"):
        topic_id = str(topic.get("number"))
        turns = topic.get("turn", [])

        # Keep track of conversation context
        context = []

        for turn_idx, turn in enumerate(turns):
            turn_id = str(turn.get("number"))
            query_id = f"{topic_id}_{turn_id}"
            turn_number = turn_idx + 1

            # Skip if no qrels for this query
            if query_id not in qrels:
                continue

            # Get query text
            if use_manual and query_id in evaluator.manual_utterances:
                query = evaluator.manual_utterances[query_id]
            else:
                query = turn.get("raw_utterance", "")

            # Run each model
            if "ZeRA" in model_instances:
                results["ZeRA"][query_id] = model_instances["ZeRA"].search(query, context)

            if "ZeRA-DT" in model_instances:
                results["ZeRA-DT"][query_id] = model_instances["ZeRA-DT"].search(query, context, turn_number)

            # Update context
            context.append(query)

    # Evaluate models
    metrics = {}
    for model_name, model_results in results.items():
        metrics[model_name] = evaluator.evaluate_metrics(model_results, qrels)

        print(f"{model_name} results for 2019:")
        print(f"  MRR:     {metrics[model_name]['MRR']:.4f}")
        print(f"  NDCG@3:  {metrics[model_name]['NDCG@3']:.4f}")
        print(f"  R@100:   {metrics[model_name]['R@100']:.4f}")

    # Print summary results
    print(f"\nSummary Results for CAsT-2019:")
    print(f"{'Model':<15} {'MRR':<10} {'NDCG@3':<10} {'R@100':<10}")
    print("-" * 45)
    for model, values in metrics.items():
        print(f"{model:<15} {values['MRR']:.4f} {values['NDCG@3']:.4f} {values['R@100']:.4f}")

    # Save results to file
    results_dir = os.path.join(data_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    manual_tag = "_manual" if use_manual else "_raw"
    results_file = os.path.join(results_dir, f"zera_results{manual_tag}.json")

    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {results_file}")
    return metrics


def main():
    """Main function to run ZeRA evaluations."""
    parser = argparse.ArgumentParser(description='Run ZeRA evaluations for conversational search')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory containing TREC datasets')
    parser.add_argument('--use_manual', action='store_true',
                        help='Use manually rewritten queries if available')
    parser.add_argument('--models', type=str, nargs='+',
                        choices=['ZeRA', 'ZeRA-DT', 'both'],
                        default=['both'], help='Which models to evaluate')
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

    # Convert 'both' to list of all models
    if 'both' in args.models:
        models = ['ZeRA', 'ZeRA-DT']
    else:
        models = args.models

    # Run evaluation
    run_evaluation(
        data_dir=args.data_dir,
        use_manual=args.use_manual,
        models=models,
        verbose=args.verbose,
        cuda=args.cuda
    )

    return 0


if __name__ == "__main__":
    main()