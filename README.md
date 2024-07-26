<h2 align='center'> Awesome-Attention-Heads </h2>
<div align='center'>

[![Awesome Attention Heads](https://img.shields.io/static/v1?label=&message=Awesome+Attention+Heads&color=black&logo=awesomelists)](https://github.com/IAAR-Shanghai/Awesome-Attention-Heads) ![](https://img.shields.io/github/last-commit/IAAR-Shanghai/Awesome-Attention-Heads?color=green)

</div>

Welcome to **Awesome-Attention-Heads**! This is the platform to get the latest research on Attention Heads. We hope to provide complete and clear cutting-edge informations for researchers studying LLM interpretability and LLM hallucination.

### Background
With the development of large language models, their underlying network structure, the Transformer, is being extensively studied. Researching the Transformer structure helps us enhance our understanding of this "black box" and improve model interpretability. Recently, there has been an increasing body of work suggesting that the model contains two distinct partitions: attention mechanisms used for behavior, inference, and analysis, and feed-forward networks (FFN) for knowledge storage. The former is crucial for revealing the functional capabilities of the model, leading to a series of studies exploring various functions within attention mechanisms, which we have termed **Attention Head Mining**.

### Table of Contents
- [Cite this repo](#cite-this-repo)
- [Latest Papers](#lastest-papers)
- [Star Trends](#star-trends)

### Cite this repo
```
@misc{AwesomeAttnHead_24_github_IAAR,
  author = {Song, Shichao and Zheng, Zifan and Wang, Yezhaohui and others},
  title = {Awesome-Attention-Heads},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/IAAR-Shanghai/Awesome-Attention-Heads}}
}
```

### Lastest Papers
Papers below are ordered by publication date:

| Date | Paper | Tags | Links & Summary |
| --- | --- | --- | --- |
| 2024-07-21 | **Answer, Assemble, Ace: Understanding How Transformers Answer Multiple Choice Questions** | ![](https://img.shields.io/badge/Answer_Head-blue) | [![Paper](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/2407.15018) |
| 2024-07-01 | **Steering Large Language Models for Cross-lingual Information Retrieval** | ![](https://img.shields.io/badge/Accuracy_Head-blue) ![](https://img.shields.io/badge/Coherence_Head-blue) | [![Paper](https://img.shields.io/badge/SIGIR-Paper-%23D2691E)](https://dl.acm.org/doi/10.1145/3626772.3657819) |
| 2024-06-21 | **MoA: Mixture of Sparse Attention for Automatic Large Language Model Compression** | ![](https://img.shields.io/badge/Local--context_Head-blue) | [![Paper](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/2406.14909) [![Code](https://img.shields.io/badge/GitHub-Code-brightgreen?logo=github)](https://github.com/thu-nics/MoA) |
| 2024-06-19 | **On the Difficulty of Faithful Chain-of-Thought Reasoning in Large Language Models** | ![](https://img.shields.io/badge/Faithfulness_Head-blue) | [![Paper](https://img.shields.io/badge/ICML-Paper-%23D2691E)](https://openreview.net/forum?id=3h0kZdPhAC) |
| 2024-06-16 | **Induction Heads as a Primary Mechanism for Pattern Matching in In-context Learning** | ![](https://img.shields.io/badge/Induction_Head-blue) | [![Paper](https://img.shields.io/badge/OpenReview-Paper-%23D2691E)](https://openreview.net/forum?id=np6hrTv7aW) |
| 2024-06-04 | **Iteration Head: A Mechanistic Study of Chain-of-Thought** | ![](https://img.shields.io/badge/Iteration_Head-blue) | [![Paper](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/2406.02128) |
| 2024-05-28 | **Knowledge Circuits in Pretrained Transformers** | ![](https://img.shields.io/badge/Mover_Head-blue) ![](https://img.shields.io/badge/Relation_Head-blue) | [![Paper](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/2405.17969) [![Code](https://img.shields.io/badge/GitHub-Code-brightgreen?logo=github)](https://github.com/zjunlp/KnowledgeCircuits) |
| 2024-05-23 | **Linking In-context Learning in Transformers to Human Episodic Memory** | ![](https://img.shields.io/badge/Induction_Head-blue) | [![Paper](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/2405.14992) |
| 2024-05-02 | **What needs to go right for an induction head? A mechanistic study of in-context learning circuits and their formation** | ![](https://img.shields.io/badge/Induction_Head-blue) | [![Paper](https://img.shields.io/badge/ICML-Paper-%23D2691E)](https://openreview.net/forum?id=O8rrXl71D5) [![Code](https://img.shields.io/badge/GitHub-Code-brightgreen?logo=github)](https://github.com/aadityasingh/icl-dynamics) |
| 2024-04-24 | **Retrieval Head Mechanistically Explains Long-Context Factuality** | ![](https://img.shields.io/badge/Retrieval_Head-blue) | [![Paper](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/2404.15574) [![Code](https://img.shields.io/badge/GitHub-Code-brightgreen?logo=github)](https://github.com/nightdessert/Retrieval_Head) |
| 2024-03-27 | **Non-Linear Inference Time Intervention: Improving LLM Truthfulness** | ![](https://img.shields.io/badge/Truthfulness_Head-blue)  | [![Paper](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/2403.18680) [![Code](https://img.shields.io/badge/GitHub-Code-brightgreen?logo=github)](https://github.com/Samsung/NL-ITI) |
| 2024-02-28 | **Cutting Off the Head Ends the Conflict: A Mechanism for Interpreting and Mitigating Knowledge Conflicts in Language Models** | ![](https://img.shields.io/badge/Memory_Head-blue) ![](https://img.shields.io/badge/Context_Head-blue) | [![Paper](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/2402.18154) |
| 2024-02-27 | **Information Flow Routes: Automatically Interpreting Language Models at Scale** | ![](https://img.shields.io/badge/Positional_Head-blue) ![](https://img.shields.io/badge/Subword_merging_Head-blue) | [![Paper](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/2403.00824) [![Code](https://img.shields.io/badge/GitHub-Code-brightgreen?logo=github)](https://github.com/facebookresearch/llm-transparency-tool) |
| 2024-02-20 | **Identifying Semantic Induction Heads to Understand In-Context Learning** | ![](https://img.shields.io/badge/Induction_Head-blue)  | [![Paper](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/2402.13055v1) |
| 2024-02-16 | **The Evolution of Statistical Induction Heads: In-Context Learning Markov Chains** | ![](https://img.shields.io/badge/Induction_Head-blue) | [![Paper](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/2402.11004) |
| 2024-02-11 | **Summing Up the Facts: Additive Mechanisms Behind Factual Recall in LLMs** | ![](https://img.shields.io/badge/Mover_Head-blue) ![](https://img.shields.io/badge/Relation_Head-blue) | [![Paper](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://www.arxiv.org/abs/2402.07321) |
| 2024-02-05 | **How do Large Language Models Learn In-Context? Query and Key Matrices of In-Context Heads are Two Towers for Metric Learning** | ![](https://img.shields.io/badge/In--Context_Head-blue) | [![Paper](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/2402.02872) |
| 2024-01-16 | **Circuit Component Reuse Across Tasks in Transformer Language Models** | ![](https://img.shields.io/badge/Content_Gatherer_Head-blue) | [![Paper](https://img.shields.io/badge/ICLR-Paper-%23D2691E)](https://openreview.net/forum?id=fpoAYV6Wsk)  [![Code](https://img.shields.io/badge/GitHub-Code-brightgreen?logo=github)](https://github.com/jmerullo/circuit_reuse) 
| 2024-01-16 | **Successor Heads: Recurring, Interpretable Attention Heads In The Wild** | ![](https://img.shields.io/badge/Successor_Head-blue) | [![Paper](https://img.shields.io/badge/ICLR-Poster-%23D2691E)](https://openreview.net/forum?id=kvcbV8KQsi) |
| 2024-01-16 | **Function Vectors in Large Language Models** | ![](https://img.shields.io/badge/Function_Vector_Head-blue) | [![Paper](https://img.shields.io/badge/ICLR-Paper-%23D2691E)](https://openreview.net/forum?id=AwyxtyMwaG&noteId=6Qv7kx00La) [![Project](https://img.shields.io/badge/Git-Page-black?logo=internet-explorer)](https://functions.baulab.info/) [![Code](https://img.shields.io/badge/GitHub-Code-brightgreen?logo=github)](https://github.com/ericwtodd/function_vectors) [![Data](https://img.shields.io/badge/GitHub-Data-brightgreen?logo=github)](https://github.com/ericwtodd/function_vectors/tree/main/dataset_files) |
| 2023-10-23 | **Linear Representations of Sentiment in Large Language Models** | ![](https://img.shields.io/badge/Direct_effect_Head-blue) | [![Paper](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/2310.15154) |
| 2023-10-06 | **Copy Suppression: Comprehensively Understanding an Attention Head** | ![](https://img.shields.io/badge/Copy_Suppression_Head-blue) | [![Paper](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/2310.04625) [![Demo](https://img.shields.io/badge/Demo-View-purple?logo=internet-explorer)](https://copy-suppression.streamlit.app/) |
| 2023-09-22 | **Inference-Time Intervention: Eliciting Truthful Answers from a Language Model** | ![](https://img.shields.io/badge/Truthfulness_Head-blue) | [![Paper](https://img.shields.io/badge/NeurIPS-Paper-%23D2691E)](https://openreview.net/forum?id=aLLuYpn83y) [![Code](https://img.shields.io/badge/GitHub-Code-brightgreen?logo=github)](https://github.com/likenneth/honest_llama) |
| 2023-09-22 | **Birth of a Transformer: A Memory Viewpoint** | ![](https://img.shields.io/badge/Induction_Head-blue) | [![Paper](https://img.shields.io/badge/NeurIPS-Paper-%23D2691E)](https://openreview.net/forum?id=3X2EbBLNsk) [![Code](https://img.shields.io/badge/GitHub-Code-brightgreen?logo=github)](https://github.com/albietz/transformer-birth) |
| 2023-07-18 | **Does Circuit Analysis Interpretability Scale? Evidence from Multiple Choice Capabilities in Chinchilla** | ![](https://img.shields.io/badge/Correct_Letter_Head-blue) | [![Paper](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/2307.09458) |
| 2023-02-02 | **Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small** | ![](https://img.shields.io/badge/Induction_Head-blue)  ![](https://img.shields.io/badge/S--Inhibition_Head-blue) ![](https://img.shields.io/badge/Name_Mover_Head-blue) ![](https://img.shields.io/badge/Previous_Token_Head-blue) ![](https://img.shields.io/badge/Duplicate_Token_Head-blue) ![](https://img.shields.io/badge/Negative_Name_Mover_Head-blue) ![](https://img.shields.io/badge/Backup_Name_Mover_Head-blue) | [![Paper](https://img.shields.io/badge/ICLR-Paper-%23D2691E)](https://openreview.net/forum?id=NpsVSN6o4ul) [![Code](https://img.shields.io/badge/GitHub-Code-brightgreen?logo=github)](https://github.com/redwoodresearch/Easy-Transformer) |
| 2022-03-08 | **In-context Learning and Induction Heads** | ![](https://img.shields.io/badge/Induction_Head-blue) | [![Paper](https://img.shields.io/badge/Anthropic-Paper-%23D2691E)](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) |
| 2021-12-22 | **A Mathematical Framework for Transformer Circuits** | ![](https://img.shields.io/badge/Induction_Head-blue) | [![Paper](https://img.shields.io/badge/Anthropic-Paper-%23D2691E)](https://transformer-circuits.pub/2021/framework/index.html) |
| 2021-05-18 | **The Heads Hypothesis: A Unifying Statistical Approach Towards Understanding Multi-Headed Attention in BERT** |  ![](https://img.shields.io/badge/Local_Head-blue)   ![](https://img.shields.io/badge/Syntactic_Head-blue)  ![](https://img.shields.io/badge/Delimiter_Head-blue)  ![](https://img.shields.io/badge/Block_Head-blue)    | [![Paper](https://img.shields.io/badge/AAAI-Paper-%23D2691E)](https://ojs.aaai.org/index.php/AAAI/article/view/17605) [![Code](https://img.shields.io/badge/GitHub-Code-brightgreen?logo=github)](https://github.com/iitmnlp/heads-hypothesis) |
| 2021-04-01 | **Have Attention Heads in BERT Learned Constituency Grammar?**<br><sub>This paper investigates whether the attention heads in BERT and RoBERTa language models have learned constituency grammar. The researchers use an unsupervised method to extract constituency parsing trees from the attention weights of these models.</sub> | ![](https://img.shields.io/badge/Constituency_grammar_inducing_Head-blue) | [![Paper](https://img.shields.io/badge/ACL-Paper-%23D2691E)](https://aclanthology.org/2021.eacl-srw.2/)|
| 2019-11-27 | **Do Attention Heads in BERT Track Syntactic Dependencies?**<br><sub>The researchers investigate if the attention heads in pre-trained transformer language models like BERT and RoBERTa can capture syntactic dependency relations between words.</sub> | ![](https://img.shields.io/badge/Syntactic_dependency_Head-blue) | [![Paper](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/1911.12246) |
| 2019-11-01 | **Adaptively Sparse Transformers** | ![](https://img.shields.io/badge/Positional_Head-blue) ![](https://img.shields.io/badge/BPE--merging_Head-blue) ![](https://img.shields.io/badge/Interrogation_Head-blue) | [![Paper](https://img.shields.io/badge/EMNLP-Paper-%23D2691E)](https://aclanthology.org/D19-1223/) [![Code](https://img.shields.io/badge/GitHub-Code-brightgreen?logo=github)](https://github.com/deep-spin/entmax) |
| 2019-08-01 | **What does BERT look at? An Analysis of BERT’s Attention** |  ![](https://img.shields.io/badge/Syntactic_Head-blue)  | [![Paper](https://img.shields.io/badge/ACL-Paper-%23D2691E)](https://aclanthology.org/W19-4828/) [![Code](https://img.shields.io/badge/GitHub-Code-brightgreen?logo=github)](https://github.com/clarkkev/attention-analysis) |
| 2019-05-22 | **Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned** | ![](https://img.shields.io/badge/Positional_Head-blue) ![](https://img.shields.io/badge/Syntactic_Head-blue) ![](https://img.shields.io/badge/Rare_words_Head-blue) | [![Paper](https://img.shields.io/badge/ACL-Paper-%23D2691E)](https://aclanthology.org/P19-1580/) [![Code](https://img.shields.io/badge/GitHub-Code-brightgreen?logo=github)](https://github.com/lena-voita/the-story-of-heads) |
| 2016-03-21 | **Incorporating Copying Mechanism in Sequence-to-Sequence Learning** | ![](https://img.shields.io/badge/Retrieval_Head-blue)  | [![Paper](https://img.shields.io/badge/ACL-Paper-%23D2691E)](https://aclanthology.org/P16-1154/) |

## Star Trends

<a href="https://star-history.com/#IAAR-Shanghai/Awesome-Attention-Heads&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=IAAR-Shanghai/Awesome-Attention-Heads&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=IAAR-Shanghai/Awesome-Attention-Heads&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=IAAR-Shanghai/Awesome-Attention-Heads&type=Date" />
  </picture>
</a>