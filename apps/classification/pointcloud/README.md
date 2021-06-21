# ModelNet40 Classification with Adaptive Robust Pooling Nodes

Modified PyTorch PointNet code for testing adaptive robust pooling nodes.

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`

## Training

To train a model, run `main.py`:

```bash
python3 main.py --outlier_fraction [OUTLIER_FRACTION] --robust_type [ROBUST_TYPE] --scale [SCALE] --alpha [ALPHA]
```

The strings available for ROBUST_TYPE are {'Q', 'PH', 'H', 'W', 'TQ', 'AL',''} and correspond to the following penalty functions:
- Q: quadratic
- PH: pseudo-Huber
- H: Huber
- W: Welsch
- TQ: truncated quadratic
- AL: general and adaptive pooling
- None: default, max-pooling

The default number of epochs is 60 and the learning rate starts at 0.001 and decays by a factor of 2 every 20 epochs.

For example, to train PointNet from scratch on GPU 0 with 60% outliers and adaptive Huber pooling replacing max pooling; outliers are presented in training and testing. use the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --outlier_fraction 0.6 --robust_type 'H' --scale 1.0 --epoch 60  --train_scale True --train_outlier True 
```

Point clouds of ModelNet40 models in HDF5 files will be automatically downloaded (416MB) to the data folder. Each point cloud contains 2048 points uniformly sampled from a shape surface. Each cloud is zero-mean and normalized into an unit sphere. There are also text files in data/modelnet40_ply_hdf5_2048 specifying the ids of shapes in h5 files. The code works for ModelNet10 with minor modifications.

## Usage

```
PointNet [-h] [--batchsize BATCHSIZE] [--epoch EPOCH]
                [--learning_rate LEARNING_RATE] [--train_metric]
                [--optimizer OPTIMIZER] [--pretrain PRETRAIN]
                [--decay_rate DECAY_RATE] [--rotation ROTATION]
                [--model_name MODEL_NAME] [--input_transform INPUT_TRANSFORM]
                [--feature_transform FEATURE_TRANSFORM] [-e]
                [--outlier_fraction OUTLIER_FRACTION]
                [--robust_type ROBUST_TYPE] [--scale SCALE]
                [--scale_lo SCALE_LO] [--train_scale TRAIN_SCALE]
                [--alpha ALPHA] [--alpha_lo ALPHA_LO] [--alpha_hi ALPHA_HI]
                [--train_alpha TRAIN_ALPHA] [--np_seed NP_SEED]
                [--train_outlier TRAIN_OUTLIER]


optional arguments:
  -h, --help            show this help message and exit
  --batchsize BATCHSIZE
                        batch size in training
  --epoch EPOCH         number of epoch in training
  --learning_rate LEARNING_RATE
                        learning rate in training
  --train_metric        whether to evaluate on training dataset
  --optimizer OPTIMIZER
                        optimizer for training
  --pretrain PRETRAIN   whether to use pretrained model
  --decay_rate DECAY_RATE
                        decay rate of learning rate
  --rotation ROTATION   range of training rotation
  --model_name MODEL_NAME
                        model to use
  --input_transform INPUT_TRANSFORM
                        use input transform in pointnet
  --feature_transform FEATURE_TRANSFORM
                        use feature transform in pointnet
  -e, --evaluate        evaluation on test set only
  --outlier_fraction OUTLIER_FRACTION
                        fraction of data that is outliers
  --robust_type ROBUST_TYPE
                        use robust pooling {Q, PH, H, W, TQ, AL}
  --scale SCALE         outlier threshold
  --alpha ALPHA         robustness parameter (for general and adaptive pooling only)
  --train_scale         adaptive or non-adaptive pooling
  --train_alpha         adaptive or non-adaptive alpha (for general and adaptive pooling only)
  --train_outlier       outliers are present in testing only or both training and testing.
```

Further details (from the yanx27/Pointnet_Pointnet2_pytorch repository) are available at [this permalink](https://github.com/yanx27/Pointnet_Pointnet2_pytorch/tree/31deedb10b85ec30178df57a6389b2f326f7c970) for the PyTorch repository and 
[this permalink](https://github.com/charlesq34/pointnet/tree/539db60eb63335ae00fe0da0c8e38c791c764d2b) for the original TensorFlow repository.

Further details on original deep declarative networks are available at [this permalink](https://github.com/anucvml/ddn).

## Links
- [Official PointNet repository](https://github.com/charlesq34/pointnet)
- [PointNet paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf)
- [Deep Declarative Networks paper](https://arxiv.org/pdf/1909.04866.pdf)
