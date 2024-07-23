# Mod_HATE
Official implementation for: Modularized Networks for Few-shot Hateful Meme Detection

This includes an original implementation of "[Modularized Networks for Few-shot Hateful Meme Detection][paper]" by Rui Cao, Roy Ka-Wei Lee, Jing Jiang.

<p align="center">
  <img src="mod-hate-arch.PNG" width="80%" height="80%">
</p>

This code provides:
- Codes for training LoRA modules for hate speech detection, meme comprehension and hateful meme interpretation.
- Composing trained LoRA modules and adapt composed modules to few-shot hateful meme detection.

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
01/25/2023: We released our implementation for our WWW submission: Modularized Networks for Few-shot Hateful Meme Detection.

## Content
1. [Installation](#installation)
2. [Prepare Datasets](#prepare-datasets)
    * [Datasets for Hateful Meme Detection](#datasets-for-hateful-meme-detection)
    * [Datasets for Module Training](#datasets-for-module-training) 
4. [Training LoRA modules](#training-lora-modules) (Section 4.2 of the paper)
5. [Modularized Networks for Hateful Meme Detection](#modularized-networks-for-hateful-meme-detection) (Section 4.3 and Section 5 of the paper)
    * [Step 1: Module Composition](#step-1-module-composition) 
    * [Step 2: Experiments](#step-2-experiments)   

## Installation
The code is tested with python 3.9. To run the code, you should install the package of transformers provided by Huggingface (version 4.33.0), PyTorch library, PEFT Library (version 0.5.0), Nevergrad. The code is implemented with the CUDA of 11.2 (you can also implement with other compatible versions) on NIVIDA A40 GPU, each with a dedicated memory of 48GB. For the implementation of LLaMA model, we leverage the HuggingFace Library with the *yahma/llama-7b-hf* checkpoint.

###
## Prepare Datasets

### Datasets for Hateful Meme Detection
We have tested on two hateful meme benchmarks: *Facebook Hateful Meme* dataset [(FHM)][fhm_dataset] and *Multimedia Automatic Misogyny Identification* dataset [(MAMI)][mami_dataset]. The datasets can be download online. If you download dataset, you need to pre-process the datasets follow the code of the [paper][pro_cap]: *Pro-Cap: Leveraging a Frozen Vision-Language Model for Hateful Meme Detection*. Alternatively, you can directly leverage the converted data shared by [Pro-Cap][pro_cap_data]. Noted, they denote MAMI dataset as mimc.

### Datasets for Module Training
We trained module capable of relevant tasks for hateful meme detection. Specifically, we focus on three relevant tasks: *hate speech detection*, *meme comprehension* and *hateful meme interpretation*. To train these modules, you need to first prepare data about these tasks. Below are datasets we considered for each task:

- Hate speech detection: we merge three hate speech detection dataset for training the module. Specifically, we leverage [DT][dt_data], [WZ][wz_data] and [Gab][gab_data] and consider hate speech detection as a generation task.
- Meme comprehension: we consider the [MemeCap][meme_cap] dataset. Given a meme, the task requires generation of its meaning. Beyond image captioning, the task also calls for recognizing and interpreting visual metaphors with respect to the text inside or around the meme.
- Hateful meme interpretation: we consider the [HatReD](hatred_data) dataset. Given a hateful meme, it requires generating the underlying hateful contextual reasons.

Alternatively, you can directly use our shared dataset in the *data* folder. *hate-speech* is the data for hate speech detection, *meme-interp* is for meme comprehension, and *hateful-meme-explain* is for explaining hateful memes.

## Training LoRA modules

After preparing all data for relevant tasks, we train individual modules for each task. We leverage the parameter-efficient technique, *low-rank adaptation* (LoRA), to tune the large language model, LLaMA and regard the LoRA module as the module capable of each task. To obtain train the modules, please run the script in [src/individual_mode.sh](src/individual_mode.sh):
```bash
bash individual_mode.sh
```

Besides using the scripts for training LoRA modules yourself, you can also directly use our provided trained LoRA modules in the *LoRA_modules* folder. **Do make sure the path of datasets is set properly (the path on your own matchine)!!!**


### Modularized Networks for Hateful Meme Detection
Based on the trained LoRA modules (*LoRA_modules/hate-speech* for hate speech detection, *LoRA_modules/meme-captions* for meme comprehension and *LoRA_modules/hate-exp* for hateful meme interpretation), we learn composed networks. **Do make sure the path of datasets is set properly (the path on your own matchine)!!!**

### Step 1: Module Composition
The learning of the composition of modules is largely based on [LoRAHub][lorahub_code]. We greatly appreciate the work of LoRAHub. By running *src/lora_learning.py*, we learn a module composer, assigning importance scores over each trained module.

### Step 2: Experiments
Based on the learned importance scores, the modularized networks is learned by weighted averaging learnt modules. Then, we can test the few-shot capabilities regarding the hateful meme detection task. You can test with the script [src/new_lora.sh](src/new_lora.sh) by:
```bash
bash new_lora.sh
```
You can also see the logging file of our reported performance in the paper in the folder *src/shot_4_LoRA* and *src/shot_8_LoRA*.

[paper]: https://arxiv.org/pdf/2402.11845
[fhm_dataset]: https://arxiv.org/abs/2005.04790
[mami_dataset]: https://aclanthology.org/2022.semeval-1.74/
[pro_cap]: https://arxiv.org/abs/2308.08088
[pro_cap_data]: https://github.com/Social-AI-Studio/Pro-Cap/tree/main/Data
[dt_data]: https://arxiv.org/abs/1703.04009
[wz_data]: https://aclanthology.org/N16-2013/
[gab_data]: https://aclanthology.org/D19-1482/
[meme_cap]: https://arxiv.org/abs/2305.13703
[hatred_data]: https://www.ijcai.org/proceedings/2023/0665.pdf
[lorahub_code]: https://github.com/sail-sg/lorahub
