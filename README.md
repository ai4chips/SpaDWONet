# SpaDWONet

**Spatially Resolved Decomposed Wavelet Operator Network for 3D-IC thermal simulation**

SpaDWONet predicts three-dimensional temperature fields from two-dimensional power maps. It combines a decomposed wavelet neural operator (DWNO) branch with a separable coordinate trunk. The spatially resolved branch retains local power-map features while reducing wavelet-weight storage. The framework supports data-driven, physics-driven, and residual-hybrid training.

## Benchmarks

### Two-dimensional Poisson benchmark

The two-dimensional benchmark compares DWNO and SpaDWONet with WNO, FNO, DeepOHeat+, DeepOHeat, ONO, Transolver, and LNO.

### Three-dimensional single-layer thermal benchmark

The single-layer benchmark compares SpaDWONet with DWONet, SpaCNN-DeepONet, CNN-DeepONet, Spa-DeepONet, and DeepONet.

### Multi-layer validation

The multi-layer benchmark evaluates the models on a four-high HBM stack with five heat-source planes.

## Code

The model folders use the following names:

```text
2d/          DWNO, SpaDWONet, WNO, FNO, DeepOHeat_plus,
             DeepOHeat, ONO, Transolver, LNO
3d/          SpaDWONet, DWONet, SpaCNN_DeepONet, CNN_DeepONet,
             Spa_DeepONet, DeepONet, DWNO_3D, WNO_DeepONet
multilayer/  SpaDWONet, DWONet, SpaCNN_DeepONet, CNN_DeepONet,
             Spa_DeepONet, DeepONet
```

## Running the code

Create the environment:

```bash
conda env create -f environment.yml
conda activate spadwonet
```

Download the [training and test datasets](https://pan.sjtu.edu.cn/web/share/55da1203e7969f98f52b81c58bbdfa2a) and place the `2d`, `3d`, and `multilayer` data folders under `data/` in the repository root.

Run the SpaDWONet training scripts from the repository root:

```bash
python 2d/SpaDWONet/train_spadwonet_2d.py
python 3d/SpaDWONet/data_driven/train_spadwonet_3d.py
python multilayer/SpaDWONet/train.py --z 1
```

For the multi-layer experiment, repeat the last command with `--z 4`, `8`, `12`, and `16` to train the other heat-source planes. Each benchmark directory also contains a `benchmark.py` evaluation script. The 2D and 3D scripts evaluate saved predictions; the multi-layer script evaluates trained per-plane checkpoints. Trained weights are not included.

## Citation

If you use this code, please cite the SpaDWONet paper.

## License

The project code is provided under the [Research Reproduction License](LICENSE.md).
