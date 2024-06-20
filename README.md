# Autonomous Racing with Action Mapping Reinforcement Learning

This repository contains the code for our research on autonomous racing, as detailed in our paper [Learning autonomous race driving with action mapping reinforcement learning](https://doi.org/10.1016/j.isatra.2024.05.010)*.

[*] Yuanda Wang, Xin Yuan, Changyin Sun. Learning autonomous race driving with action mapping reinforcement learning. ISA Transactions 150:1-14 (2024). DOI: https://doi.org/10.1016/j.isatra.2024.05.010

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

In this repository, we provide the implementation of our autonomous racing algorithm. The code is designed to simulate and evaluate the performance of autonomous vehicles in a racing environment. 
Please refer to our paper for a detailed explanation of the methods and results.

## Installation

### Prerequisites

Ensure you have the following software and libraries installed:

- Python 3.7+
- PyTorch 1.X
- NumPy
- Matplotlib 3.8+
- ffmpeg 4.2+ (to generate the race video)

### Steps

1. Clone the repository:
    ```sh
    git clone https://github.com/YuandaWang/Auto-Race-with-AM-RL.git
    ```
2. Navigate to the project directory:
    ```sh
    cd repository
    ```

## Usage

### Training the AM-RL Driving Policy

To train the autonomous racing model, run the following script:

```sh
python Train_CarRace_TD3AM.py
```
The policy (actor network) will be saved under the `./models/` directory for every 100k training iterations 

### Evaluating the AM-RL Driving Policy 

To evaluate the trained model, use the following script:

```sh
python Run_CarRace_ExampleModel.py
```

In the `./example/model` directory, we provide two example policy models named `model_1776` and `model_1969`. By default settings, the python script will load the example model `model_1969`. 
If you want to evaluate your trained model_XXX from the `./model/` directory, change the path (line 19): `model_path = 'example_model/model_1969` to `model_path = 'models/model_XXX'`

The default evaluation script will generate four figures and one mp4 video clip in the `./results/` directory, which are:

- Trajectory for lap 1 
- Trajectory for lap 2 
- Car data log for lap 1 
- Car data log for lap 2 
- Video for lap 1 and lap 2

If your trained policy cannot finish at least one lap, the data log for lap 2 will not be given.  

### Results

The saved example model `model_1969` reaches the flying lap time of 36.94s in Track-A

### Future Work

We have several planned features and modules that we aim to integrate into the project in the future:

- Training and evaluation on Track-B (Beijing Ruisi Track). This part will be given soon. 
- Two-car head-to-head competition scenario. 
- Multi-car competition scenario.
- Integration to Xuance 

### Contributing

We welcome contributions from the community. 

### License

This project is licensed under the MIT License. See the LICENSE file for details.

### Acknowledgments

We would like to thank [Dr. Wenzhang Liu](https://github.com/wenzhangliu) for his support on open sourcing this project.

This work was supported by National Natural Science Foundation of China [grant numbers 62136008, 62103104, 62203113]; 
the Natural Science Foundation of Jiangsu Province, China [grant number BK20210215].







