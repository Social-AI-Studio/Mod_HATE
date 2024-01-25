# Mod_HATE
Official implementation for: Modularized Networks for Few-shot Hateful Meme Detection

This includes an original implementation of "[Modularized Networks for Few-shot Hateful Meme Detection][paper]" by Rui Cao, Roy Ka-Wei Lee, Jing Jiang.

<p align="center">
  <img src="mod-hate-arch.PNG" width="80%" height="80%">
</p>

This code provides:
- Codes for generating relevant knowledge for knowlege-intensive VQA questions with GPT-3.
- Generated knowledge from GPT-3 for each K-VQA dataset
- Codes to incorporate the generated knowledge into K-VQA models based on 1) UnifiedQA, 2) OPT and 3) GPT-3

Please leave issues for any questions about the paper or the code.

If you find our code or paper useful, please cite the paper:
```
@inproceedings{ caojiang2024kgenvqa,
    title={Knowledge Generation for Zero-shot Knowledge-based VQA},
    author={ Rui Cao, Jing Jiang},
    journal={EACL},
    year={ 2024 }
}
```

### Announcements
01/18/2023: Our paper is accepted by EACL, 2024, as Findings. 

## Content
1. [Installation](#installation)
2. [Prepare Datasets](#prepare-datasets)
    * [Step 1: Downloading Datasets](#step-1-downloading-datasets)
    * [Step 2: Caption Generation](#step-2-caption-generation) 
4. [Knowledge Generation](#knowledge-generation) (Section 3.1 of the paper)
    * [Step 1: Knowledge Initialization](#step-1-knowledge-initialization) 
    * [Step 2: Knowledge Diversification](#step-2-knowledge-diversification) 
5. [Incorporating Generated Knowledge for K-VQA](#incorporating-generated-knowledge-for-k-vqa) (Section 3.2 and 4.3 of the paper)
    * [K-VQA based on UnifiedQA](#k-vqa-based-on-unifiedqa)
    * [K-VQA based on OPT](#k-vqa-based-on-opt)
    * [K-VQA based on GPT](#k-vqa-based-on-gpt)   

## Installation
The code is tested with python 3.8. To run the code, you should install the package of transformers provided by Huggingface (version 4.29.2), PyTorch library (1.13.1 version), LAVIS package from Salesforce (version 1.0.2). The code is implemented with the CUDA of 11.2 (you can also implement with other compatible versions) on Tesla V 100 GPU card (each with 32G dedicated memory). Besides running the OPT model, all other models take one GPU each.

###
## Prepare Datasets

### Step 1: Downloading Datasets
We have tested on three benchmarks for knowledge-intensive VQA (K-VQA) datasets: *OK-VQA* and *A-OKVQA*. Datasets are available online. You can download datasets via links in the original dataset papers and put them into the desired file paths according to the code: OK_PATH, A_OK_PATH, PATH.

### Step 2: Caption Generation
As mentioned in Section 3.2, we used text-based QA model for the final K-VQA. The images should be converted into texts (i.e., image captions) so that text-based models can comprehend. We adopt a similar approach to [PNP-VQA][pnp-vqa] to generate question-aware captions. When utilizing OPT, we follow the code for [Img2LLM][imgllm] to generate synthetic question-answer pairs as demonstrating examples. The generated captions for OK-VQA can be found in *OK_VQA/large_captions* and the captions for A-OKVQA can be found in *A_OKVQA/aokvqa_val_captions_100.pkl*. The synthetic question answer pairs for OK-VQA can be found in *OK_VQA/ok_vqa_qa_img2llm.pkl* and *A_OKVQA/a-ok_vqa_qa_img2llm.pkl* for A-OKVQA.  

## Knowledge Generation

Here we describe how we generate relevant knowledge for questions by prompting GPT-3 (details in Section 3.1). Specifically, we prompt GPT-3 with demonstrations in *OK_VQA/demonstrations.txt* to initialize one piece of knowledge for each question. Then we diversify knowledge with the self-supervised knowledge diversification technique.

### Step 1: Knowledge Initialization
We leverage the in-context learning capability of GPT-3 and prompt it with a few-demonstrations. You can try to generate initialized knowledge with the help of GPT-3. The code for initialization can be found in [src/kb-gen/knowledge-initialization.ipynb](src/kb-gen/knowledge-initialization.ipynb).

### Step 2: Knowledge Diversification
We next diversify generated knowledge with self-supervised diversification technique. The code can be found in [src/kb-gen/knowledge-diversification.ipynb](src/kb-gen/knowledge-diversification.ipynb). Alternatively, you can directly leverage our generated knowledge for OK-VQA and A-OKVQA in the folder [src/OK_VQA/cluster_generated_kb](src/OK_VQA/cluster_generated_kb) and [src/A_OKVQA/cluster_generated_kb](src/A_OKVQA/cluster_generated_kb) respectively.


### Incorporating Generated Knowledge for K-VQA
We finally incoporate our generated knowledge into text-based question-answering models, to show the effectiveness of generated knowledge. We tested three text-based question answering models: UnifiedQa, OPT and GPT-3.

### K-VQA based on UnifiedQA
Code can be found in [src/unifiedQA](src/unifiedQA), which is for the 3B version of UnifiedQA. For the answers for OK-VQA, please apply code in [src/unifiedQA/Ans_Norm.ipynb](src/unifiedQA/Ans_Norm.ipynb) to conduct answer normalization. Predicted answers are also included in the [src/unifiedQA](src/unifiedQA) folder.

### K-VQA based on OPT
Code for using OPT as the text-based question answering model can be found in the [src/opt](src/opt) folder.

### K-VQA based on GPT
Zero-shot and few-shot testing with GPT-3 with the incorporation of generated knowledge can be found in [gpt/zero-shot-gpt.ipynb](gpt/zero-shot-gpt.ipynb).



[paper]: https://arxiv.org/abs/2308.08088
[pnp-vqa]: https://arxiv.org/pdf/2210.08773.pdf
[imgllm]: https://arxiv.org/abs/2212.10846
