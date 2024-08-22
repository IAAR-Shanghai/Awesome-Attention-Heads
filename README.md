﻿<h2 align='center'> Awesome-Attention-Heads </h2>
<div align='center'>

[![Awesome Attention Heads](https://img.shields.io/static/v1?label=&message=Awesome+Attention+Heads&color=black&logo=awesomelists)](https://github.com/IAAR-Shanghai/Awesome-Attention-Heads) ![](https://img.shields.io/github/last-commit/IAAR-Shanghai/Awesome-Attention-Heads?color=green)

</div>

Welcome to **Awesome-Attention-Heads**! This is a platform to get the latest research on different kinds of LLM's Attention Heads. We hope to provide complete and clear cutting-edge informations for researchers studying LLM interpretability and LLM hallucination. We will be grateful if you can :star: the project or commit a PR!

### Background
With the development of large language models, their underlying network structure, the Transformer, is being extensively studied. Researching the Transformer structure helps us enhance our understanding of this "black box" and improve model interpretability. Recently, there has been an increasing body of work suggesting that the model contains two distinct partitions: attention mechanisms used for behavior, inference, and analysis, and feed-forward networks (FFN) for knowledge storage. The former is crucial for revealing the functional capabilities of the model, leading to a series of studies exploring various functions within attention mechanisms, which we have termed **Attention Head Mining**.

### Table of Contents
- [Latest Papers](#lastest-papers)
- [Star Trends](#star-trends)

### Lastest Papers
Papers below are ordered by **publication date**:

#### Year 2024

| Date | Paper & Summary | Tags | Links |
| --- | --- | --- | --- |
| 2024-08-01 | **Enhancing Semantic Consistency of Large Language Models through Model Editing: An Interpretability-Oriented Approach**<br><sub></sub> | ![](https://img.shields.io/badge/Consistency_Head-blue) | [![Paper](https://img.shields.io/badge/ACL_Findings-Paper-%23D2691E)](https://aclanthology.org/2024.findings-acl.199/) |
| 2024-07-31 | **Correcting Negative Bias in Large Language Models through Negative Attention Score Alignment**<br><sub><ul><li>Introduced Negative Attention Score (NAS) to quantify and correct negative bias in language models.</li><li>Identified negatively biased attention heads and proposed Negative Attention Score Alignment (NASA) for fine-tuning.</li><li>NASA effectively reduced the precision-recall gap while preserving generalization in binary decision tasks.</li></ul></sub> | ![](https://img.shields.io/badge/Negative_Head-blue) | [![Paper](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/2408.00137) |
| 2024-07-29 | **Detecting and Understanding Vulnerabilities in Language Models via Mechanistic Interpretability**<br><sub>The researchers found that the GPT-2 Small model had a strong tendency to misclassify samples where the third letter of the acronym was either 'A' or 'S'. They identified that the attention head 10.10 in the model was the main contributor to this vulnerability, as it had a strong tendency to predict the letter 'Q' instead of the correct letter.</sub> | ![](https://img.shields.io/badge/Vulnerable_Head-blue) | [![Paper](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/2407.19842) |
| 2024-07-22 | **RazorAttention: Efficient KV Cache Compression Through Retrieval Heads**<br><sub>The paper introduces RazorAttention, a training-free KV cache compression algorithm. For models using ALiBi positional embedding, RazorAttention can directly determine the retrieval heads based on their attention scope. For models using RoPE positional embedding, RazorAttention identifies the retrieval heads by analyzing their attention behavior on randomly generated tokens. RazorAttention maintains a full cache for the retrieval heads and compresses the remote tokens in non-retrieval heads using a "compensation token".</sub> | ![](https://img.shields.io/badge/Retrieval_Head-blue) | [![Paper](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/2407.15891) |
| 2024-07-21 | **Answer, Assemble, Ace: Understanding How Transformers Answer Multiple Choice Questions**<br><sub>The researchers use two complementary techniques - vocabulary projection and activation patching - to analyze the internal mechanisms of transformer models (Olmo and Llama2) on MCQA tasks. They design carefully-crafted prompts and a synthetic task to disentangle different aspects of model behavior.</sub> | ![](https://img.shields.io/badge/Answer_Head-blue) | [![Paper](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/2407.15018) |
| 2024-07-09 | **Induction Heads as an Essential Mechanism for Pattern Matching in In-context Learning**<br><sub>This study explores the role of induction heads, a specific type of attention mechanism in large language models, in enabling their ability to learn from a few examples (in-context learning).</sub> | ![](https://img.shields.io/badge/Induction_Head-blue) | [![Paper](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/2407.07011) |
| 2024-07-01 | **Steering Large Language Models for Cross-lingual Information Retrieval**<br><sub></sub> | ![](https://img.shields.io/badge/Accuracy_Head-blue) ![](https://img.shields.io/badge/Coherence_Head-blue) | [![Paper](https://img.shields.io/badge/SIGIR-Paper-%23D2691E)](https://dl.acm.org/doi/10.1145/3626772.3657819) |
| 2024-06-21 | **MoA: Mixture of Sparse Attention for Automatic Large Language Model Compression**<br><sub>The paper proposes a new method called Mixture of Attention (MoA) to address the efficiency challenges of large language models (LLMs) in long context scenarios. LLMs use attention mechanisms that compute interactions between tokens, but this becomes computationally expensive as the input length grows.</sub> | ![](https://img.shields.io/badge/Local--context_Head-blue) | [![Paper](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/2406.14909) [![Code](https://img.shields.io/badge/GitHub-Code-brightgreen?logo=github)](https://github.com/thu-nics/MoA) |
| 2024-06-19 | **On the Difficulty of Faithful Chain-of-Thought Reasoning in Large Language Models**<br><sub>The paper aims to investigate whether three common techniques - activation editing, fine-tuning, and in-context learning - can be used to improve the faithfulness of CoT reasoning generated by LLMs. The researchers hypothesize that these techniques may help steer the LLMs to produce more faithful CoT explanations.</sub> | ![](https://img.shields.io/badge/Faithfulness_Head-blue) | [![Paper](https://img.shields.io/badge/ICML-Paper-%23D2691E)](https://openreview.net/forum?id=3h0kZdPhAC) |
| 2024-05-28 | **Knowledge Circuits in Pretrained Transformers**<br><sub>The authors construct "Knowledge Circuits" - a critical subgraph in the computation graph of LLMs that captures the information flow and interactions between different components. They utilize ablation-based methods to identify these circuits associated with various expressions of knowledge.</sub> | ![](https://img.shields.io/badge/Mover_Head-blue) ![](https://img.shields.io/badge/Relation_Head-blue) | [![Paper](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/2405.17969) [![Code](https://img.shields.io/badge/GitHub-Code-brightgreen?logo=github)](https://github.com/zjunlp/KnowledgeCircuits) |
| 2024-05-23 | **Linking In-context Learning in Transformers to Human Episodic Memory**<br><sub>The paper examines the relationship between attention heads in Transformer models and human episodic memory. It focuses on a specific type of attention heads called "induction heads," which are crucial for the in-context learning capabilities of large language models.</sub> | ![](https://img.shields.io/badge/Induction_Head-blue) | [![Paper](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/2405.14992) |
| 2024-05-07 | **How does GPT-2 Predict Acronyms? Extracting and Understanding a Circuit via Mechanistic Interpretability**<br><sub>The main objective of the paper is to discover the circuit responsible for the task of three-letter acronym prediction on the GPT-2 model. They find that the most important heads, called "letter mover heads", attend from the previously predicted letter to the current capital letter and copy its information to make the next prediction.</sub> | ![](https://img.shields.io/badge/Letter_Mover_Head-blue) | [![Paper](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/2405.04156) |
| 2024-05-02 | **What needs to go right for an induction head? A mechanistic study of in-context learning circuits and their formation**<br><sub></sub> | ![](https://img.shields.io/badge/Induction_Head-blue) | [![Paper](https://img.shields.io/badge/ICML-Paper-%23D2691E)](https://openreview.net/forum?id=O8rrXl71D5) [![Code](https://img.shields.io/badge/GitHub-Code-brightgreen?logo=github)](https://github.com/aadityasingh/icl-dynamics) |
| 2024-04-24 | **Retrieval Head Mechanistically Explains Long-Context Factuality**<br><sub></sub> | ![](https://img.shields.io/badge/Retrieval_Head-blue) | [![Paper](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/2404.15574) [![Code](https://img.shields.io/badge/GitHub-Code-brightgreen?logo=github)](https://github.com/nightdessert/Retrieval_Head) |
| 2024-03-27 | **Non-Linear Inference Time Intervention: Improving LLM Truthfulness**<br><sub></sub> | ![](https://img.shields.io/badge/Truthfulness_Head-blue)  | [![Paper](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/2403.18680) [![Code](https://img.shields.io/badge/GitHub-Code-brightgreen?logo=github)](https://github.com/Samsung/NL-ITI) |
| 2024-02-28 | **Cutting Off the Head Ends the Conflict: A Mechanism for Interpreting and Mitigating Knowledge Conflicts in Language Models**<br><sub></sub> | ![](https://img.shields.io/badge/Memory_Head-blue) ![](https://img.shields.io/badge/Context_Head-blue) | [![Paper](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/2402.18154) |
| 2024-02-27 | **Information Flow Routes: Automatically Interpreting Language Models at Scale**<br><sub></sub> | ![](https://img.shields.io/badge/Positional_Head-blue) ![](https://img.shields.io/badge/Subword_merging_Head-blue) | [![Paper](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/2403.00824) [![Code](https://img.shields.io/badge/GitHub-Code-brightgreen?logo=github)](https://github.com/facebookresearch/llm-transparency-tool) |
| 2024-02-20 | **Identifying Semantic Induction Heads to Understand In-Context Learning** | ![](https://img.shields.io/badge/Induction_Head-blue)  | [![Paper](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/2402.13055v1) |
| 2024-02-16 | **The Evolution of Statistical Induction Heads: In-Context Learning Markov Chains** | ![](https://img.shields.io/badge/Induction_Head-blue) | [![Paper](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/2402.11004) |
| 2024-02-11 | **Summing Up the Facts: Additive Mechanisms Behind Factual Recall in LLMs** | ![](https://img.shields.io/badge/Mover_Head-blue) ![](https://img.shields.io/badge/Relation_Head-blue) | [![Paper](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://www.arxiv.org/abs/2402.07321) |
| 2024-02-05 | **How do Large Language Models Learn In-Context? Query and Key Matrices of In-Context Heads are Two Towers for Metric Learning** | ![](https://img.shields.io/badge/In--Context_Head-blue) | [![Paper](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/2402.02872) |
| 2024-01-16 | **Circuit Component Reuse Across Tasks in Transformer Language Models** | ![](https://img.shields.io/badge/Content_Gatherer_Head-blue) | [![Paper](https://img.shields.io/badge/ICLR-Paper-%23D2691E)](https://openreview.net/forum?id=fpoAYV6Wsk)  [![Code](https://img.shields.io/badge/GitHub-Code-brightgreen?logo=github)](https://github.com/jmerullo/circuit_reuse) 
| 2024-01-16 | **Successor Heads: Recurring, Interpretable Attention Heads In The Wild** | ![](https://img.shields.io/badge/Successor_Head-blue) | [![Paper](https://img.shields.io/badge/ICLR-Poster-%23D2691E)](https://openreview.net/forum?id=kvcbV8KQsi) |
| 2024-01-16 | **Function Vectors in Large Language Models** | ![](https://img.shields.io/badge/Function_Vector_Head-blue) | [![Paper](https://img.shields.io/badge/ICLR-Paper-%23D2691E)](https://openreview.net/forum?id=AwyxtyMwaG&noteId=6Qv7kx00La) [![Project](https://img.shields.io/badge/Git-Page-black?logo=internet-explorer)](https://functions.baulab.info/) [![Code](https://img.shields.io/badge/GitHub-Code-brightgreen?logo=github)](https://github.com/ericwtodd/function_vectors) [![Data](https://img.shields.io/badge/GitHub-Data-brightgreen?logo=github)](https://github.com/ericwtodd/function_vectors/tree/main/dataset_files) |

#### Year 2023
| Date | Paper & Summary | Tags | Links |
| --- | --- | --- | --- |
| 2023-10-23 | **Linear Representations of Sentiment in Large Language Models** | ![](https://img.shields.io/badge/Direct_effect_Head-blue) | [![Paper](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/2310.15154) |
| 2023-10-06 | **Copy Suppression: Comprehensively Understanding an Attention Head**<br><sub>The paper aims to explain the main role of an attention head (L10H7) in the GPT-2 Small language model across its entire training distribution. The researchers found that this attention head primarily performs "copy suppression" - it detects when the model is predicting a token that already appears in the context, and then suppresses that prediction.</sub> | ![](https://img.shields.io/badge/Copy_Suppression_Head-blue) | [![Paper](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/2310.04625) [![Demo](https://img.shields.io/badge/Demo-View-purple?logo=internet-explorer)](https://copy-suppression.streamlit.app/) |
| 2023-09-22 | **Inference-Time Intervention: Eliciting Truthful Answers from a Language Model** | ![](https://img.shields.io/badge/Truthfulness_Head-blue) | [![Paper](https://img.shields.io/badge/NeurIPS-Paper-%23D2691E)](https://openreview.net/forum?id=aLLuYpn83y) [![Code](https://img.shields.io/badge/GitHub-Code-brightgreen?logo=github)](https://github.com/likenneth/honest_llama) |
| 2023-09-22 | **Birth of a Transformer: A Memory Viewpoint** | ![](https://img.shields.io/badge/Induction_Head-blue) | [![Paper](https://img.shields.io/badge/NeurIPS-Paper-%23D2691E)](https://openreview.net/forum?id=3X2EbBLNsk) [![Code](https://img.shields.io/badge/GitHub-Code-brightgreen?logo=github)](https://github.com/albietz/transformer-birth) |
| 2023-07-18 | **Does Circuit Analysis Interpretability Scale? Evidence from Multiple Choice Capabilities in Chinchilla**<br><sub>The researchers were able to identify a set of 45 key nodes (attention heads and MLPs) that are directly responsible for selecting the correct answer letter. They categorized these nodes into groups like "correct letter heads", "content gatherer heads", and "amplification heads", based on their attention patterns and functions. </sub> | ![](https://img.shields.io/badge/Correct_Letter_Head-blue) ![](https://img.shields.io/badge/Content_Gatherer_Head-blue) ![](https://img.shields.io/badge/Amplification_Head-blue) ![](https://img.shields.io/badge/Constant_Head-blue) ![](https://img.shields.io/badge/Single_Letter_Head-blue) | [![Paper](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/2307.09458) |
| 2023-02-02 | **Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small** | ![](https://img.shields.io/badge/Induction_Head-blue)  ![](https://img.shields.io/badge/S--Inhibition_Head-blue) ![](https://img.shields.io/badge/Name_Mover_Head-blue) ![](https://img.shields.io/badge/Previous_Token_Head-blue) ![](https://img.shields.io/badge/Duplicate_Token_Head-blue) ![](https://img.shields.io/badge/Negative_Name_Mover_Head-blue) ![](https://img.shields.io/badge/Backup_Name_Mover_Head-blue) | [![Paper](https://img.shields.io/badge/ICLR-Paper-%23D2691E)](https://openreview.net/forum?id=NpsVSN6o4ul) [![Code](https://img.shields.io/badge/GitHub-Code-brightgreen?logo=github)](https://github.com/redwoodresearch/Easy-Transformer) |

#### Before ChatGPT Announced
| Date | Paper & Summary | Tags | Links |
| --- | --- | --- | --- |
| 2022-03-08 | **In-context Learning and Induction Heads** | ![](https://img.shields.io/badge/Induction_Head-blue) | [![Paper](https://img.shields.io/badge/Anthropic-Paper-%23D2691E)](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) |
| 2021-12-22 | **A Mathematical Framework for Transformer Circuits** | ![](https://img.shields.io/badge/Induction_Head-blue) | [![Paper](https://img.shields.io/badge/Anthropic-Paper-%23D2691E)](https://transformer-circuits.pub/2021/framework/index.html) |
| 2021-05-18 | **The Heads Hypothesis: A Unifying Statistical Approach Towards Understanding Multi-Headed Attention in BERT** |  ![](https://img.shields.io/badge/Local_Head-blue)   ![](https://img.shields.io/badge/Syntactic_Head-blue)  ![](https://img.shields.io/badge/Delimiter_Head-blue)  ![](https://img.shields.io/badge/Block_Head-blue)    | [![Paper](https://img.shields.io/badge/AAAI-Paper-%23D2691E)](https://ojs.aaai.org/index.php/AAAI/article/view/17605) [![Code](https://img.shields.io/badge/GitHub-Code-brightgreen?logo=github)](https://github.com/iitmnlp/heads-hypothesis) |
| 2021-04-01 | **Have Attention Heads in BERT Learned Constituency Grammar?**<br><sub>This paper investigates whether the attention heads in BERT and RoBERTa language models have learned constituency grammar. The researchers use an unsupervised method to extract constituency parsing trees from the attention weights of these models.</sub> | ![](https://img.shields.io/badge/Constituency_grammar_inducing_Head-blue) | [![Paper](https://img.shields.io/badge/ACL-Paper-%23D2691E)](https://aclanthology.org/2021.eacl-srw.2/) |
| 2019-11-27 | **Do Attention Heads in BERT Track Syntactic Dependencies?**<br><sub>The researchers investigate if the attention heads in pre-trained transformer language models like BERT and RoBERTa can capture syntactic dependency relations between words.</sub> | ![](https://img.shields.io/badge/Syntactic_dependency_Head-blue) | [![Paper](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/1911.12246) |
| 2019-11-01 | **Adaptively Sparse Transformers** | ![](https://img.shields.io/badge/Positional_Head-blue) ![](https://img.shields.io/badge/BPE--merging_Head-blue) ![](https://img.shields.io/badge/Interrogation_Head-blue) | [![Paper](https://img.shields.io/badge/EMNLP-Paper-%23D2691E)](https://aclanthology.org/D19-1223/) [![Code](https://img.shields.io/badge/GitHub-Code-brightgreen?logo=github)](https://github.com/deep-spin/entmax) |
| 2019-08-01 | **What does BERT look at? An Analysis of BERT’s Attention** |  ![](https://img.shields.io/badge/Syntactic_Head-blue)  | [![Paper](https://img.shields.io/badge/ACL-Paper-%23D2691E)](https://aclanthology.org/W19-4828/) [![Code](https://img.shields.io/badge/GitHub-Code-brightgreen?logo=github)](https://github.com/clarkkev/attention-analysis) |
| 2019-07-01 | **Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned** | ![](https://img.shields.io/badge/Positional_Head-blue) ![](https://img.shields.io/badge/Syntactic_Head-blue) ![](https://img.shields.io/badge/Rare_words_Head-blue) | [![Paper](https://img.shields.io/badge/ACL-Paper-%23D2691E)](https://aclanthology.org/P19-1580/) [![Code](https://img.shields.io/badge/GitHub-Code-brightgreen?logo=github)](https://github.com/lena-voita/the-story-of-heads) |
| 2016-03-21 | **Incorporating Copying Mechanism in Sequence-to-Sequence Learning** | ![](https://img.shields.io/badge/Retrieval_Head-blue)  | [![Paper](https://img.shields.io/badge/ACL-Paper-%23D2691E)](https://aclanthology.org/P16-1154/) |

## Star Trends

<a href="https://star-history.com/#IAAR-Shanghai/Awesome-Attention-Heads&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=IAAR-Shanghai/Awesome-Attention-Heads&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=IAAR-Shanghai/Awesome-Attention-Heads&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=IAAR-Shanghai/Awesome-Attention-Heads&type=Date" />
  </picture>
</a>
