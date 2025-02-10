## Welcome to TokScope!
### Overview
TokScope provides tools for token analysis, including vocabulary mapping, language classification, and dimensionality reduction. This repo helps you explore token embeddings—the foundation of large language models (LLMs). Paired with an interactive website, it offers a clear and intuitive way to understand tokenization and embeddings.

### User Guide
This interactive dashboard visualizes token embeddings (numerical representations of words/subwords) from a pre-trained model using t-SNE dimensionality reduction. Each point in the scatter plot represents a word or subword from the model's vocabulary, with spatial relationships reflecting semantic similarity:

- Explore Token Relationships: Words with similar meanings (e.g., "cat" and "dog") tend to cluster together, while unrelated words are more dispersed.
- Identify Semantic Clusters: Words of the same category (e.g., sentiment words, technical terms) often form groups, revealing how the model understands language.
- Analyze Distribution Patterns: Observe how common words, rare words, and symbols are distributed.
With TokenScope, you can intuitively understand how the model encodes language, discover potential biases, and evaluate the quality of token embeddings. Start exploring now!

### What is a Token?
A token is the fundamental unit of text processing. Depending on the model’s tokenization strategy, a token can represent a complete word, a subword, or even a single character. In modern natural language processing (NLP) models, tokens are transformed into high-dimensional embedding vectors, which encode both semantic and syntactic features. These embeddings power key NLP tasks such as machine translation and text summarization. In this dashboard, each point in the scatter plot corresponds to a token’s vector representation, allowing users to visually explore how tokens relate and cluster across different languages or contexts.

### How t-SNE Works
t-Distributed Stochastic Neighbor Embedding (t-SNE) is a dimensionality reduction algorithm designed specifically for visualizing high-dimensional data. It projects high-dimensional token embeddings into a two-dimensional space while preserving local structure, ensuring that semantically similar tokens remain close together in the plot. This makes t-SNE an ideal tool for exploring token embedding distributions—whether analyzing cross-lingual relationships or studying the internal structure of a model’s vocabulary, t-SNE helps users gain insights into how NLP models encode knowledge.
