# 1.5bit-NEAT

![Training result bipedal walker](results/best.gif)

## Authors
- Biswajit Banerjee ([sumon@gatech.edu](mailto:sumon@gatech.edu))
- Dennis Frank ([dfrank6@gatech.edu](mailto:dfrank6@gatech.edu))
- Leyao Huang ([lhuang395@gatech.edu](mailto:lhuang395@gatech.edu))

## Abstract
In this study, we present a novel approach to evolving neural network architectures for control tasks by combining neuroevolution techniques with aggressive weight quantization. Specifically, we employ the NeuroEvolution of Augmenting Topologies (NEAT) algorithm to search for effective network topologies while constraining the weights to a minimal 1.58-bit {-1, 0, 1} representation. Our key insight is that the network architecture itself can encode inductive biases well-suited for a given task, while ultra-low bit quantization allows highly compact and efficient models.

We evaluate our approach on the challenging BipedalWalker-v3 environment from OpenAI Gym, where an agent must learn to control a bipedal robot to navigate varied terrains. Through empirical analysis, we demonstrate that our 1.58-bit quantized NEAT models achieve competitive performance compared to traditional weight-agnostic neural networks, while requiring a drastically smaller memory footprint. Notably, the learned architectures exhibit emergent behaviors and strategies for robust locomotion. Our work highlights the potential of combining neuroevolution with severe quantization as a promising direction for developing resource-efficient solutions to complex control problems.

## Introduction
In recent years, the intersection of Evolutionary Strategies (ES) with Deep Learning (DL) has significantly interested and prompted exploration within the field of Artificial Intelligence (AI). This has resulted in the emergence of Evolutionary Deep Learning (EDL), a novel paradigm that diverges from traditional gradient-based approaches and draws inspiration from nature.

## Approach

### Environment
We chose the OpenAI Gym’s BipedalWalker-v3 environment to train and test our networks, providing direct comparison of our models. The environment simulates a Bipedal Walker with a state represented by a 24-element space vector encompassing various parameters. The control actions available to the Walker are encapsulated in a four-element vector that specifies the torque commands for the motors at each of the four joints.

### NEAT
We developed a framework to instantiate and evolve feedforward neural network architectures using the NEAT algorithm. We initialized our networks with a minimal topology consisting of the 24 observation nodes connected to the 4 action nodes and incrementally increased the architectural complexity of populations as governed by a configuration file with custom-specified parameters.

### Reinforcement Learning
OpenAI Gym Bipedal Walker and NEAT were combined in a reinforcement learning framework such that a network genome would predict the next action, and the Bipedal Walker environment would perform a step with it. This process was iterated until success (reaching frame 1600) or failure (the head of the Walker hitting the ground).

### Fitness
The fitness function evaluates the model's performance in the environment and its suitability for contributing to the next generation. We used NEAT sampling to create the next generation by selecting two well-performing models and crossing them.

### Quantization
Neural Network weight quantization provides similar performance to non-quantized networks while minimizing memory footprint and reducing floating point operations. We applied integer quantization and bit quantization to our models.

### Hyper-parameters
We performed a hyper-parameter search on two relevant parameters: survival threshold and beta threshold for credit assignment, to further refine model performance.

## Experimental Setup and Results

### Setup
We trained 200 networks over 300 generations with architecture evolving and performance improving with each successive generation. Hyper-parameters were tuned for the Default Genome and kept consistent, only modifying the required parameters for other models to ensure consistent and fair comparison.

### Performance
We monitored the fitness, which is the average accumulated reward for each network over 5 scenarios. The Bit Quantize genome consistently performed better in both fitness and network complexity.

### Evaluations
At the 300th generation, results showed that the Bit Quantize genome outperformed the Default Genome and Softmaxed genome in both fitness and network complexity.

## Discussion
Our study demonstrated consistent trends through multiple tests. While Softmax sampling performed poorly, it was a necessary variation to test. Models were evaluated in both normal and hardcore modes of the Bipedal Walker environment.

## Conclusion
This study presents a quantified NEAT approach for controlling a Bipedal Walker within a simulated environment. Our methodology demonstrated effective learning outcomes and has potential for scalable and memory-efficient real-world applications.

