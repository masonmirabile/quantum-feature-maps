# Investigating Variational Quantum Classifier Feature Maps Across Dataset Geometries

Quantum machine learning is the intersection of machine learning and quantum computing. Variational Quantum Classifiers (VQCs) are hybrid quantum-classical models. Input data is encoded into qubits using feature maps, which use various quantum circuit gates, operations, and formulas to encode data in different ways.

This project aims to investigate how, and when, a VQC can use feature mapping to boost its performance in classifying classical data.

Seven synthetic 2D binary-classification datasets were used throughout the project, allowing for a high level of control over geometry and noise levels. Each dataset contained 500 observations and 11 dataset-specific levels of Gaussian noise.

First, VQC hyperparameters were optimized using three established reference feature maps (Angle, Z, and ZZ). The main experiment compares established and original feature maps across the seven datasets, keeping the VQC architecture constant.

All experiments were implemented using Python 3.13 and PennyLane 0.45, with quantum circuits simulated using PennyLane's `lightning.qubit` simulator.
