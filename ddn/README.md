# `ddn` Package

This document provides a brief description of the modules and utilities of adaptive robust pooling and feature projections.

This document is a subset and modification of ddn package in general. For ddn package in general see [ddn](https://github.com/anucvml/ddn).

## Basic

The `ddn.basic` package contains standard python code for experimenting with deep declarative nodes. The
implementation assumes that all inputs and outputs are vectors (or more complicated data structures
have been vectorized). In this repository, we concern about robust pooling only.

* `ddn.basic.node`: defines the interface for data processing nodes and declarative nodes.
* `ddn.basic.adaptive_robust_nodes`: implements nodes for robust pooling.


## PyTorch

The `ddn.pytorch` package includes efficient implementations of deep declarative nodes suitable for including
in an end-to-end learnable model. The code builds on the PyTorch framework and conventions. This repository concerns
about robust pooling and feature projections only.

* `ddn.pytorch.node`: defines the PyTorch interface for data processing nodes and declarative nodes.
* `ddn.pytorch.adaptive_projections`: differentiable adaptive Euclidean projection layers onto Lp balls and spheres.
* `ddn.pytorch.adaptive_robostpool`: differentiable adaptive robust pooling layers.
* `ddn.pytorch.robust_loss_pytorch`: some utility functions adapted from [robust_loss_pytorch](https://github.com/jonbarron/robust_loss_pytorch)
