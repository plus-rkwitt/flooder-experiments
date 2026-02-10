# Experiments for Flood complex

This repository contains the code used for the experiments in the NeurIPS 2025 paper: "The Flood Complex: Large-Scale Persistent Homology on Millions of Points". [📄](https://arxiv.org/abs/2509.22432)

Please cite as:
```bibtex
@inproceedings{graf2025floodcomplex,
      title={The Flood Complex: Large-Scale Persistent Homology on Millions of Points}, 
      author={Graf, Florian and Pellizzoni, Paolo and Uray, Martin and Huber, Stefan and Kwitt, Roland},
      year={2025},
      booktitle={NeurIPS},
}
```

See the [flooder](github.com/plus-rkwitt/flooder) GitHub page for the official implementation of the Flood Complex.


## Setup
In the following, we assume that the repository has been cloned into `/tmp/flooder-experiments`.

### Setup a new Anaconda environment
```bash
conda create -n "flooder-experiments" python=3.9
conda activate flooder-experiments
```

### Installing ```flooder```
First, install our implementation of the flood complex. This already installs pytorch and most dependencies required to run the experiments.
```bash
pip install flooder
```

### Installing `torchph`
```bash
git clone https://github.com/c-hofer/torchph.git 
conda develop  ../torchph/
python -c 'import torchph' # check
```

### Installing additional dependencies

```bash
pip install matplotlib
pip install flaml[automl]
```

## Scalability

Experiments on the scalability of the Flood complex can be found in the ```scalability/``` folder.

We can reproduce the results of Figure 4 by running:
```
python scalability/synthetic.py --run-alpha --outdir output/
```

Moreover, experiments on the large scale point clouds can be run using:
```
python scalability/large_scale.py --root bench_dataset/ --idx 0 --outdir output/
```

## Reproducing ML results

All code needed to reproduce the results on the application of the Flood complex to machine learning can be found in the ```learning/``` folder.

#### Obtaining PH diagrams

We can run the Flood complex on a dataset as follows:
```
python learning/ph_flood.py --dataset coral --root data/coral/
```

Similarly, we can subsample 20k points from the point clouds and run the Alpha complex:
```
python learning/ph_alpha.py --dataset coral --root data/coral/ --num-points 20000

for s in {1..5}; do python learning/ph_alpha.py --dataset coral --root data/coral/ --num-points 4000 --seed "$s"; done
```

Since point clouds in the ```rocks``` dataset vary in size, we subsample as a fraction of their size:
```
python learning/ph_alpha.py --dataset rocks --root data/rocks/ --scale-num-points 0.01
```

#### Using PH diagrams for classification and regression

We can then run the classification pipeline:
```
python learning/ph_classify.py --dataset coral --root data/coral/ --phdir data/coral/floodph/ --flaml-classifier lrl1 --stretch-quantile 0.05 --stats-file ./output/flood_corals_600s_stretch0.05.yaml --time-budget 30

python learning/ph_classify.py --dataset coral --root data/coral/ --phdir data/coral/alphaph_20000_0/ --flaml-classifier lrl1 --stretch-quantile 0.05 --stats-file ./output/flood_corals_600s_stretch0.05.yaml --time-budget 30

python learning/ph_classify_subsample.py --dataset coral --root data/coral/ --phdirs data/coral/alphaph_4000_1/ data/coral/alphaph_4000_2/ data/coral/alphaph_4000_3/ data/coral/alphaph_4000_4/ data/coral/alphaph_4000_5/ --flaml-classifier lrl1 --stretch-quantile 0.05 --stats-file ./output/avg_corals_600s_stretch0.05.yaml --time-budget 30
```

Similarly, for classification with a LGBM we run (e.g. for Flood PH):
```
python learning/ph_classify.py --dataset mcb --root data/mcb/ --phdir data/mcb/floodph/ --flaml-classifier lgbm --stretch-quantile 0.05 --stats-file ./output/flood_mcb_600s_stretch0.05.yaml --time-budget 30
```

And for regression (e.g. for Flood PH):
```
python learning/ph_regression.py --dataset rocks --root data/rocks/ --phdir data/rocks/floodph/   --flaml-classifier lgbm --stretch-quantile 0.05 --stats-file ./output/flood_mcb_600s_stretch0.05.yaml --time-budget 30
```

#### Neural network baselines

```bash
pip install learning/baselines/pointnet2_ops_lib --no-build-isolation
pip install torch-geometric
pip install torch-cluster
pip install einops
pip install ninja
pip install timm
```

We run neural network baselines to compare to PH-based methods. In particular, we use ```pointnet++```, ```pvt``` and ```pointmlp``` as baselines. 
```
python learning/nn_baselines_cls.py --model pointnet++ --dataset mcb --root data/mcb/ --data-augmentation --lr 0.001 --stats-file output_nn/mcb_pointnet.yaml
```

For the ```rocks``` dataset, we sample points from a contiguous region rather than using random sampling:
```
python learning/nn_baselines_cls.py --model pointnet++ --dataset rocks --root data/rocks/ --from-corner --lr 0.001 --stats-file output/rocks_pointnet.yaml
```


#### Scalability/accuracy trade offs on swisscheese

We can reproduce the results of Figure 5 by running:
```
python learning/cheese_sweep.py --root data/sweep/ --output output/
```
