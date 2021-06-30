# voxelmorph: Learning-Based Image Registration  

**Experiments with voxelmorph on two papers of uncertainty quantification using SG-MCMC and bayesian-based learnt similarity metric**

this branch has instructions on how to run the code on the biobank dataset that gives the results for baseline method in the two papers [reference to be added later]. 

# getting the code

```
git clone --branch biobank https://github.com/mfazampour/voxelmorph/ 
```
- install the requirements listed in `setup.py`
for the learnt similiaty metric the submodule repo is also required:
```
git submodule init --remote
```

# How to use the code

## Training

Sicne the biobank dataset is pretty huge, it's better to go through the folders and create a list of traiing and validation and test images. These lists are stored so for later runs it won't need anytime for finding the paths.

The tensorboard 

```
python ./scripts/torch/train.py <DATA_SRC_FOLDER> --model-dir <FOLDER> --gpu 0 --loader-name biobank --inshape 128 128 128 --target-spacing 1.8203 --patient-list-src <FOLDER_TO_STORE_DATALIST> --display-freq 20 --steps-per-epoch 200 --epochs 200 --batch-size 1 --bidir --image-loss mse --use-probs --flow-logsigma-bias 0.0 --kl-lambda 1.0 --statistics-freq 40 --bidir --image-sigma 0.2 --lambda 1.0 --num-statistics-run 5 --num-test-imgs 10  --atlas 1007582
```
The tensorboard output is saved at the model-dir location. since the biobank dataset is pretty big, we don't get to run evaluation on all images in the eval folder. One can set the number of images to be used for evaluation using num-test-imgs flag.

most of the other flags are the same as the original voxelmorph and can be found on the original readme or explained in the code.

## Testing/Evaluation

after training the models, baseline & learnt, to compare the performance of the two models, `batch_register_biobank` script is used:

```
python ./scripts/torch/batch_register_biobank.py <DATA_SRC_FOLDER> --model-base <PATH_TO_BASELINE_MODEL> --model-test <PATH_TO_LEARNT_MODEL> --gpu 0 --loader-name biobank --inshape 128 128 128 --target-spacing 1.8203 --patient-list-src <FOLDER_TO_STORE_DATALIST>  --method-base <e.g. "voxelmorph(baseline, SSD)"> --method-test <e.g. "voxelmorph(learnt, SSD)"> --use-probs --atlas 1007582 --num-statistics-runs 3 --num-test-imgs 1000 --output-dir <OUTPUT_FOLDER>
```
the csv files would be saved in the output_dir and the names given as the method-base and method-test would be used in the csv file to differentiate between the resluts. `num-statistics-runs` determines the number of samples per image pair.



 
