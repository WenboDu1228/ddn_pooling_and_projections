# ddn_pooling_and_projections
This repository is for my honours project at the Australian National Univerisity under the supervision of [Professor Stephen Gould](https://cecs.anu.edu.au/people/stephen-gould). The project is based on [Deep Declarative Networks: A New Hope](https://arxiv.org/pdf/1909.04866.pdf) and closely related to [A General and Adaptive Robust Loss Function](https://arxiv.org/pdf/1701.03077.pdf).

The thesis will be released after the examination. You can see the [presentation slides](https://github.com/WenboDu1228/ddn_pooling_and_projections/blob/master/wenbodu_final_talk.pdf) for a overview. For all questions, please contact me at wenbodu3@gmail.com.

 
Currently, the repository only contains implementations. In general, it can be devided into adaptive robust pooling and adaptive feature projections.

## Adaptive robust pooling
We have pseudo-Huber pooling , Huber pooling , Welsch pooling, general and adaptive pooling. The former three was adaptive from [Deep Declarative Networks: A New Hope](https://arxiv.org/pdf/1909.04866.pdf), and the last is the argmin form of the robust function in [A General and Adaptive Robust Loss Function](https://arxiv.org/pdf/1701.03077.pdf).
Details of the implementations is in [adaptive_robustpool.py](https://github.com/WenboDu1228/ddn_pooling_and_projections/blob/master/ddn/pytorch/adaptive_robustpool.py). 

## Adaptive feature projections
For adaptive feature projections, we have adaptive L1, L2 and L infinity sphere and ball projections. Details of the implementations is in [adaptive_projections.py](https://github.com/WenboDu1228/ddn_pooling_and_projections/blob/master/ddn/pytorch/adaptive_projections.py).

## Others
We also have some [tutorials
](https://github.com/WenboDu1228/ddn_pooling_and_projections/tree/master/tutorials) for demonstration. The experiments on point cloud classification and image classification are under [classification](https://github.com/WenboDu1228/ddn_pooling_and_projections/tree/master/apps/classification).

This repository is still updating. In future, part of the code will merge into offical ddn repository.

For deep delarative networks in general, please visit https://github.com/anucvml/ddn.
