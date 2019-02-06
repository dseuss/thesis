# Due to, or in spite of? The effect of constraints on efficiency in quantum estimation problems

In this thesis, we study the interplay of constraints and complexity in quantum estimation.
We investigate three inference problems, where additional structure in the form of constraints is exploited to reduce the sample and/or computational complexity.
The first example is concerned with _uncertainty quantification_ in quantum state estimation, where the _positive-semidefinite constraint_ is used to construct more powerful, that is smaller, error regions.
However, as we show in this work, doing so in an optimal way constitutes a computationally hard problem, and therefore, is intractable for larger systems.
This is in stark contrast to the unconstrained version of the problem under consideration.
The second inference problem deals with _phase retrieval_ and its application to characterizing _linear optical circuits_.
The main challenge here is the fact that the measurements are insensitive to complex phases, and hence, reconstruction requires deliberate utilization of interference.
We propose a reconstruction algorithm based on ideas from _low-rank matrix recovery_.
More specifically, we map the problem of reconstruction from phase-insensitive measurements to the problem of recovering a rank-one matrix from linear measurements.
For the efficient solution of the latter it is crucial to exploit the rank-one constraint.
In this work, we adapt existing work on phase retrieval to the specific application of characterizing linear optical devices.
Furthermore, we propose a measurement ensemble tailored specifically around the limitations encountered in this application.
The main contribution of this work is the proof of efficacy and efficiency of the proposed protocol.
Finally, we investigate _low-rank tensor recovery_ -- the problem of reconstructing a low-complexity tensor embedded in an exponentially large space.
We derive a sufficient condition for reconstructing low-rank tensors from product measurements, which relates the error of the initialization and concentration properties of the measurements.
Furthermore, we provide evidence that this condition is satisfied with high probability by Gaussian product tensors with the number of measurements only depending on the target's intrinsic complexity, and hence, scaling polynomially in the order of tensor.
Therefore, the low-rank constraint can be exploited to dramatically reduce the sample complexity of the problem.
Additionally, the use of measurement tensors with an efficient representation is necessary for computational efficiency.
