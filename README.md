# Test-Time Curricula for Targeted RL

Implementation of the method and experiments for the "Learning on the Job: Test-Time Curricula for Target RL" paper.

## 🚀 Installation & Setup
This library builds on the [Test-Time Reinforcement Learning (TTRL)](https://github.com/PRIME-RL/TTRL) and the [Volcano Engine Reinforcement Learning (verl)](https://github.com/volcengine/verl) libraries. Please refer to the documentation of these libraries for basic functionality and setup.


Set the PYTHONPATH environment variable to point to the *ttc* directory. 
```
export PYTHONPATH=...
```
Install further libraries and the modified version of verl.
```
pip install word2number latex2sympy2 math-verify[antlr4_13_2]==0.8.0 
pip install -e $PYTHONPATH/TTRL/verl/. 
pip install -e $PYTHONPATH/activeft/. 
```

<!-- ## Training dataset creation

creating dataset

processing dataset -->

## 📂 Dataset Preprocessing

Use the generate_verl_data.sh script to create datasets for verl training.
```
DATA_PATH=...
bash $PYTHONPATH/generate_verl_data.sh Qwen/Qwen3-8B lasgroup/verifiable-corpus math-ai/aime25 $DATA_PATH false 500000 true false false true 
```


## 🎯 Training
To start TTC-RL training on the generated dataset:
```
bash $PYTHONPATH/training/verl_training.sh Qwen/Qwen3-8B lasgroup_verifiable-corpus_math-ai_aime25_500000 True False test grpo vtl False False 10 1000 1000
```


## 📖 Citation
Please cite our work if you use this library in your research.

```bibtex
@article{hubotter2024efficiently,
	title        = {Efficiently Learning at Test-Time: Active Fine-Tuning of LLMs},
	author       = {H{\"u}botter, Jonas and Bongni, Sascha and Hakimi, Ido and Krause, Andreas},
	year         = 2024,
	journal      = {arXiv preprint arXiv:2410.08020}
}
```