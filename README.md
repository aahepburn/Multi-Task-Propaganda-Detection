# Read me first.

I run all my experiments from **master_notebook.ipynb**. The notebook makes calls to relevant APIs, which are stored in respective .py files. The most interesting files to look at would be **single_task.py** and **multi_task.py**, which contain all of the model classes I used in my experiments.

![Pipeline Concept](https://github.com/user-attachments/assets/80152276-9639-4dae-9f67-8f8ad462face)

**Figure 1:** Visual Summary of the Data Pipeline




## Abstract:

The limitations of Transformer-based natural language processing (NLP) systems, coupled with the fragmented nature of computational propaganda research, continue to hinder system generalisability. The diverse contexts and domains that define the field pose challenges for NLP methodologies, which remain highly susceptible to distribution shifts. While these limitations are widely acknowledged, they are rarely studied systematically. This thesis explores whether multi-task learning (MTL) can improve model robustness through shared representations across persuasion detection tasks, which are central to propaganda research. Through a comparative evaluation of single-task and multi-task architectures, we examine generalisation under domain and label shifts. We find that multi-task models can recover performance degradation under data scarcity through indirect data augmentation, and can improve performance under extreme class imbalance. While we cannot definitively conclude that MTL results in domain-agnostic models, we validate it as a viable and more robust approach for low-resource propaganda detection at scale.

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

## Model Architecture

![Model-Setup-Horizontal](https://github.com/user-attachments/assets/266dd68f-6e8c-47ee-ab29-3bd7f631463e)

**Figure 2:** Multi-task learning setup and multi-task learning with task-specific adapters (blue). Both setups fine-tune the parameters of shared encoder by optimising their summed loss. The single-task learning setup consists of only one of the tasks during training and does not include an adapter.

