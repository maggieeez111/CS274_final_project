# CS274 Final Project - Dynamic Parameter Adaptation for Zero-shot Response-Aware Conversational Retrieval

This repository contains the implementation of the ZeRA-DT (Zero-shot Response-Aware with Dynamic Thresholds) method for conversational information retrieval, alongside baseline implementations. The code evaluates different retrieval approaches on the TREC CAsT-2019 dataset.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Methods](#methods)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Citation](#citation)

## Overview

Conversational search presents unique challenges compared to traditional single-query search systems. As users engage in multi-turn conversations, queries often become shorter, rely on context from previous turns, and contain references that cannot be resolved in isolation.

This project implements and evaluates several conversational retrieval methods:
- Traditional lexical retrieval (BM25)
- Neural models (BERT, ColBERT)
- Context-aware extensions (ZeCo²)
- Zero-shot Response-Aware method (ZeRA)
- proposed ZeRA with Dynamic Thresholds (ZeRA-DT)

ZeRA-DT dynamically adjusts expansion weights based on conversation depth, providing better adaptability to the evolving nature of conversational queries.

## Features

- Pure Python implementation with minimal dependencies
- Support for multiple retrieval models:
  - BM25 (lexical retrieval)
  - BERT (dense retrieval)
  - ColBERT (late interaction retrieval)
  - ZeCo² (context-aware retrieval)
  - ZeRA (zero-shot response-aware retrieval)
  - ZeRA-DT (our proposed dynamic parameter adaptation method)
- Evaluation on TREC CAsT-2019 dataset
- Support for both raw and manually rewritten queries
- Comprehensive evaluation metrics (MRR, NDCG@3, R@100)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/conversational-retrieval.git
cd conversational-retrieval
```

2. Install the required dependencies:
```bash
pip install torch transformers tqdm numpy
```

## Data

Due to the large size of the dataset, the data is not included in this repository. You can download the required data from Google Drive:

[TREC CAsT Dataset (Google Drive Link)](https://drive.google.com/drive/folders/1U0HXobqbgsqPaDkhzQOCkvIltY39z7g6?usp=sharing)

The dataset includes:
- TREC CAsT-2019 evaluation topics
- Relevance judgments (qrels)
- Paragraph corpus
- Manual utterances

After downloading, extract the data to a local directory. You'll need to specify this directory when running the code.

## Usage

### Running the Baseline Models

```bash
python oldBaseline.py --data_dir "path/to/data/directory"
```

### Running the ZeRA and ZeRA-DT Models

```bash
python newMethod.py --data_dir "path/to/data/directory" 
```

## Methods

### BM25
Traditional lexical retrieval method that ranks documents based on term frequency and inverse document frequency statistics. It treats queries and documents as bags of words without considering semantic relationships or context.

### BERT
Neural dense retrieval approach that encodes both queries and documents into fixed-length vector representations using a pre-trained language model. BERT captures semantic meaning beyond exact term matching.

### ColBERT
An enhancement to neural retrieval that maintains token-level representations rather than using a single vector. ColBERT computes fine-grained interactions between query and document tokens for more nuanced matching.

### ZeCo²
A context-aware extension of ColBERT that creates an enhanced query representation by concatenating previous turns with the current query, maintaining conversational context.

### ZeRA
Zero-shot Response-Aware method combines three levels of query expansion (term-level, sentence-level, and passage-level) with static weighting parameters to enhance retrieval performance without requiring training data.

### ZeRA-DT (Our Proposed Method)
ZeRA with Dynamic Thresholds extends ZeRA by adaptively adjusting expansion weights based on conversation depth. Early turns benefit from stronger term matching, while later turns require more contextual influence as the conversation evolves.

Weights for different turn depths:
- For early turns (1-3): Higher weight on term-level expansion (α = 0.6, β = 0.3, γ = 0.1)
- For middle turns (4-6): Balanced weights (α = 0.5, β = 0.4, γ = 0.15)
- For later turns (7+): Higher weight toward contextual components (α = 0.4, β = 0.4, γ = 0.2)

## Evaluation Metrics

- **MRR (Mean Reciprocal Rank)**: Measures the average position of the first relevant document in the ranked results.
- **NDCG@3 (Normalized Discounted Cumulative Gain at rank 3)**: Measures the effectiveness of the ranking, focusing on the top 3 results.
- **R@100 (Recall at 100)**: Measures the proportion of relevant documents found in the top 100 ranked results.

## Results

Our experiments show that while ZeRA achieves strong performance with fixed parameters, ZeRA-DT's dynamic parameter adaptation further enhances retrieval quality, especially for longer conversations. Due to sampling variability in our limited-index evaluation, performance fluctuates between evaluation runs, but ZeRA-DT consistently outperforms most baseline methods.

Sample results on CAsT-2019 dataset:

| Method          | MRR    | NDCG@3 | R@100  |
|-----------------|--------|--------|--------|
| BM25            | 0.0793 | 0.0130 | 0.1022 |
| BERT            | 0.0044 | 0.0021 | 0.0517 |
| ColBERT         | 0.0415 | 0.0227 | 0.0632 |
| ZeCo²           | 0.0306 | 0.0174 | 0.0584 |
| ZeRA            | 0.0819 | 0.0337 | 0.0954 |
| ZeRA-DT (Updated)| 0.0843 | 0.0341 | 0.0968 |

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The ZeRA method is based on the work by Wang et al. [1]
- We use the TREC CAsT-2019 dataset for evaluation
- Our implementations of BM25, BERT, and ColBERT are simplified versions for research purposes

## References

[1] J. Wang, X. Chen, P. He, F. Zhang, L. Wang and J. Sheng, "Zero-shot Response-Aware Query Expansion Method for Conversational Retrieval," 2024 IEEE/WIC International Conference on Web Intelligence and Intelligent Agent Technology (WI-IAT), 2024, pp. 630-635, doi: 10.1109/WI-IAT62293.2024.00101.
