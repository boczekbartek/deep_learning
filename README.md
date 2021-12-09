# Deep Learning Course @ [VU Amsterdam](https://vu.nl)

## Installation

**NOTE** works for MacOS and Linux - for Windows it might work in some setups, but was not tested

1. Clone this repo including submodules:
    ```bash
    # with https (user + password) auth
    git clone --recurse-submodules git@github.com:boczekbartek/deep_learning.git
    # with ssh (ssh key based) auth:
    git clone --recurse-submodules https://github.com/boczekbartek/deep_learning.git
    ```
2. Create conda or pip virtual env (conda is recommended)
    ```bash
    conda create --name dl python=3.9
    ```
3. Install required packages
    - If you are on machine with CUDA GPU (like Nvidia gpus) use:
    ```bash
    ./conda_install_cuda.sh
    ```

    - Or if you don't have CUDA
    ```bash
    ./conda_install.sh
    ```

## Assignment 1 - implementing gradient descent manually in numpy
Code directory: [A1](./A1)

## Assignment 2 - Automatic differentiation
Code directory: [A2](./A2)

## Assignment 3 - RNN or CNN
### RNN
Code directory: [A3-rnn](./A3-rnn)

### CNN:
Code directory: [A3-cnn](./A3-cnn)

## Assignment 4 - Generative Models / Graph CNNs / Reinforcement Learning
### VAE + GAN
Code directory: [A4-vae-gan](./A4-vae-gan)

### Graph CNNs
Code directory: [A4-graph-cnn](./A4-graph-cnn)

### Deep Reinforcement Learning
Code directory: [A3-reinforcement](./A4-reinforcement)
