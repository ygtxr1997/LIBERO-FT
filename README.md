<div align="center">
  <img src="images/LIEBRO-FT_logo2.png" width="70%">

  # LIBERO-FT: Evaluating Robotic Manipulation under Force-Related Domain Shifts
</div>


## Overview

`LIBERO-FT` is a toolkit (including benchmark) developed for evaluating robotic manipulation under **force-related** domain shifts. 
Highly based on the LIBERO benchmark, it can be considered an extension that builds upon its core functionality. 
Key features include extracting 6D wrench force-torque data from the simulator (Mujoco), replaying and saving benchmark trajectories, 
and modifying physical properties like gripper stiffness and friction to simulate domain shifts (based on [Robosuite](https://robosuite.ai/)). 
This repository aims to provide a framework for testing and evaluating the adaptability of robotic models to 
varying physical conditions, enabling thorough analysis before deploying force-based policies into real-world scenarios.


## Features

- **Read Mujoco 6D force-torque data (MujocoSensorReader)**: Reads raw 6D wrench force-torque data from the native Mujoco sensors and exposes it to the LIBERO `OffScreenRenderEnv` via the `WrenchObsWrapper`.
- **Replay original LIBERO demonstrations and save force-torque data (HDF5Replayer)**: Replays full trajectories based on LIBERO benchmark demonstrations (e.g., LIBERO-90), reading and saving force-torque data into new HDF5 files.
- **Physical domain shift (PhysicsHelper)**: Modifies physical properties like gripper stiffness, friction, and gravity in Robosuite, and evaluates model robustness against physical domain shifts using the `LiberoForceSocketEvaluator`.
- **A training&testing example**: Demonstrates training and testing of a diffusion policy with added force-torque inputs, evaluating performance under various domain shifts after in-domain training.


## Installation

`LIBERO-FT` shares most of its environment with the original [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO?tab=readme-ov-file#installtion), with only a few additional dependencies required for debugging and training purposes. To get started, follow these steps:

```shell
# Copied from original LIBERO
conda create -n libero python=3.8.13
conda activate libero
git clone https://github.com/ygtxr1997/LIBERO-FT.git  # different here
cd LIBERO-FT
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Install additional dependencies
pip install git+https://github.com/ygtxr1997/RoboKit.git@6066dbb
```


## Datasets

Our replayed trajectories with **wrench force-torque sensory data** injected: [https://huggingface.co/datasets/ygtxr1997/LIBERO-FT](https://huggingface.co/datasets/ygtxr1997/LIBERO-FT)

Original LIBERO data without force-torque data: [https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets)


## Usage

### 1. Replay original LIBERO trajectory and inject force-torque sensory data

We provide a class `libero.force.hdf5_replayer.HDF5Replayer` to replay LIBERO trajectories online on your own GPU.
`HDF5Replayer` can construct the full environment according to the meta info saved in LIBERO HDF5 files, replay actions frame-by-frame, and consequently check if the replayed trajectory succeds or not.

A usage example for `HDF5Replayer` is available at: [notebooks/force_hdf5_replayer.ipynb](./notebooks/force_hdf5_replayer.ipynb).

Below are the visualization examples of 2x speedup replayed trajectories (left to right are: original observation saved in LIBERO HDF5, online observation during replay, 6D force-torque sensory data):

<div align="center">
  <img src="./images/replay_00_rst=-1_suc=1_KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it.bddl.gif" width="100%" alt="replay_k1_open_put">
  <p>(Click to play) Based on LIBERO-90: KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it</p>
</div>

<div align="center">
  <img src="./images/replay_00_rst=-1_suc=1_KITCHEN_SCENE7_open_the_microwave.bddl.gif" width="100%" alt="replay_k1_open_put">
  <p>(Click to play) Based on LIBERO-90: KITCHEN_SCENE7_open_the_microwave</p>
</div>


### 2. 


### Data Extraction

To extract wrench mechanics data from Robosuite, use the following script:

""
python extract_wrench_data.py
"" id="do32fg"

This will extract the 6D force-torque data for evaluation. You can specify the type of tasks or robots from which the data will be extracted.

### Domain Shift Evaluation

Run the evaluation script to analyze the effect of domain shift on robotic manipulation:

""
python evaluate_domain_shift.py --task <task_name> --robot <robot_name>
"" id="0hj56d"

> **Note**: Replace `<task_name>` and `<robot_name>` with the desired task and robot configuration. The repository includes predefined tasks such as `pick_and_place`, `stacking`, etc.

### Visualization

You can visualize the wrench data and evaluation results using the following command:

""
python visualize_results.py
"" id="s43hv9"

This will plot graphs for force-torque distributions and the effects of domain shift.

## Project Structure

""
LIBERO-FT/
│
├── extract_wrench_data.py  # Script for extracting wrench data from Robosuite
├── evaluate_domain_shift.py  # Script for evaluating domain shift
├── visualize_results.py  # Visualization of evaluation results
├── requirements.txt  # List of required dependencies
├── data/  # Folder for storing extracted wrench data
└── README.md  # This file
"" id="6jy9qv"

## Contributing

We welcome contributions to `LIBERO-FT`! If you'd like to contribute, please fork the repository and submit a pull request. Before contributing, please ensure that your code follows the repository’s style and passes all tests.

### How to Contribute

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Write tests for any new functionality.
4. Make sure all tests pass.
5. Submit a pull request with a description of your changes.

## Citation

```latex
@misc{LIBEROFT2026,
  author = {{Yuan, Ge and Xu, Dong}},
  title = {{LIBERO-FT: Evaluating Robotic Manipulation under Force-Related Domain Shifts}},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/ygtxr1997/LIBERO-FT}},
}
```

## Acknowledgements

- [Robosuite](https://robosuite.ai/) for providing the robotic simulation environment.
- [LIBERO Benchmark](https://libero-benchmark.com/) for force-torque benchmarks used in this project.

---

> **Note**: If any specific part of the code, like force vectors or data handling, is unclear, you might want to include additional clarifications or placeholders where appropriate.
