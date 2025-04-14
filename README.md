# Read me first.

The current preliminary results are based on:

**- single-task-learning.ipynb**

AND

**- multi-task-learning.ipynb**


<img width="606" alt="Screenshot 2025-04-14 at 2 10 03 PM" src="https://github.com/user-attachments/assets/3744ead2-698a-4ec0-9b47-8cffb66ede95" />




Other notebooks:

**- STL-zeroshot-experiments.ipynb** contains experiments on zero-shot learning using label embeddings. This is work in progress and is not included in the preliminary results.

**-translation-GPT-4o.ipynb** contains the machine translation pipeline.

**- domain-encoding.ipynb** contains some functions to infer the domain of each article.







## Abstract:

The limitations of Transformer-based NLP systems, coupled with the fragmented nature of computational propaganda research, continue to hinder system generalisability. The diverse contexts and domains that define the field pose challenges for NLP methodologies, which remain highly susceptible to distribution shifts. While these limitations are widely acknowledged, they are rarely studied systematically. This thesis explores whether multi-task learning can improve model robustness through shared representations across persuasion detection tasks. Through a comparative evaluation of single-task and multi-task architectures, the study examines generalisation under domain and label shifts. Post-hoc interpretability methods assess whether MTL models capture transferable linguistic features rather than domain-specific artefacts. This work contributes to ongoing research on robust NLP methods in the digital humanities and other related fields.

## Research Questions:

The central research question guiding this thesis is: 

- To what extent can shared linguistic representations of persuasive strategies mitigate performance degradation caused by distribution shifts in persuasion detection NLP tasks?

The sub-questions are as follows:

RQ1: To what extent does MTL improve generalisation compared to STL in unseen domains, as measured by classification performance?
- RQ1.1: How does MTL perform compared to STL in entity framing classification under domain shifts?
- RQ1.2: How does MTL perform compared to STL in narrative classification under domain shifts?
  
RQ2: To what extent can MTL adapt to label distribution shifts across domains more effectively than STL?
- RQ2.1: How does MTL react to label distribution shifts in entity framing, compared to STL?
- RQ2.2: How does MTL react to label distribution shifts in narrative classification, compared to STL?

RQ3: To what extent does MTL learn domain-invariant representations of persuasive language, and how does this compare to STL?
- RQ3.1: Do MTL models capture transferable linguistic features across entity framing and narrative classification, as measured through feature attribution analysis, and how does this differ from STL?


Together, these questions establish an empirical framework for theorising whether multi-task learning architectures can capture transferable linguistic representations of persuasion techniques, thereby improving out-of-domain generalisation.



