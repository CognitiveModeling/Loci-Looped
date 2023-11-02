# LOOPING LOCI: Learning Object Permanence from Videos

<b>TL;DR:</b> Introducing Loci-Looped, an extension to Loci, with a parameterized percept gate that learns to slot-wise fuse imaginations with sensations. As a result, Loci learns to track objects through occlusions and to imagine their trajectory. See our paper for more details: [arxiv](https://arxiv.org/abs/2310.10372)




https://github.com/CognitiveModeling/Loci-Looped/assets/60604502/2b6d5fcf-8eeb-47db-ab26-1a9e4a3a48f0




---
## Requirements
A suitable [conda](https://conda.io/) environment named `loci-l` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate loci-l
```


## Dataset and trained models

Preprocessed datasets together with pretrained models and results can be found [here](todo)

Download the *pretrained* folder and place it in the */out* folder. 
Download the *data* folder and place it in the */data* folder. 

## Reproducing the results from the paper
The model's performances on the testsets are stored as csv files in the respective results folders. To re-generate these results run: 

```
sh run_evaluation.py
```

To test a single Loci model on the testset run:
```
python -m scripts.exec.eval -cfg [path-to-config-file] -load [path-to-model-file] -n replica
```

To analyse the generated results, run the evaluation notebooks in the */evaluation* folder. You may need to modify the *root_path* variable in the notebooks pointing to the respective results folder.


## Training Guide
Loci-Looped can be trained using with this command: 

```
python -m scripts.exec.train -cfg [path-to-config-file] -n [run_description]
```

The original config files used for training can be found in the *cfg* folder. To continue training from an existing model use this command:

```
python -m scripts.exec.train -cfg [path-to-config-file] -load [path-to-model-file] -n [run_description]
```

## Acknowledgements 
MOT metrics are computed with py-motmetrics. The visual evaluation metrics script is borrowed from the [Slotformer](https://github.com/pairlab/SlotFormer) repository. 
