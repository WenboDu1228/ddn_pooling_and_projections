# ddn_pooling_and_projections
This repository is for my honours project at the Australian National Univerisity under the supervision of [Professor Stephen Gould](https://cecs.anu.edu.au/people/stephen-gould). The project is based on [Deep Declarative Networks: A New Hope](https://arxiv.org/pdf/1909.04866.pdf) and closely related to [A General and Adaptive Robust Loss Function](https://arxiv.org/pdf/1701.03077.pdf).

The thesis will be released after the examination. For all questions, please contact me at wenbodu3@gmail.com.
 
 ## Running code
Currently, the repository only contains implementations. In general, it can be devided into adaptive robust pooling and adaptive feature projections. For adaptive robust pooling, we have pseudo-Huber pooling , Huber pooling , Welsch pooling, general and adaptive pooling. The former three was adaptive from [Deep Declarative Networks: A New Hope](https://arxiv.org/pdf/1909.04866.pdf), and the last is the argmin form of the robust function in [A General and Adaptive Robust Loss Function](https://arxiv.org/pdf/1701.03077.pdf).
Details of the implementations is in [ddn/pytorch/robustpool.py](https://github.com/WenboDu1228/ddn_pooling_and_projections/blob/master/ddn/pytorch/adaptive_robustpool.py). 

For adaptive feature projections, we have adaptive L1, L2 and L infinity sphere and ball projections. Details of the implementations is in [ddn/pytorch/learnable_projections.py](https://github.com/WenboDu1228/ddn_pooling_and_projections/blob/master/ddn/pytorch/adaptive_projections.py). We also have some tutorials for demonstration under [
./tutorials/
](https://github.com/WenboDu1228/ddn_pooling_and_projections/tree/master/tutorials). The experiments on point cloud classification and image classification is under [./apps/classification](https://github.com/WenboDu1228/ddn_pooling_and_projections/tree/master/apps/classification).

This repository is still updating. For deep delarative networks in general, please visit https://github.com/anucvml/ddn.
