# neural-networks-and-deep-learning-rs

Sample Rust code for [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/). you can find the Python version [here](https://github.com/mnielsen/neural-networks-and-deep-learning).

> **Note**
> This is a teaching project which shows how neural networks work, so it will reduce dependencies and optimizes as much as possible.
>
> **Warning**
> This project is still **work in progress**, so stability and documentation cannot be guaranteed.

## Build & Run

1. Download and extract the [MNIST database](http://yann.lecun.com/exdb/mnist/)(including the training set and test set), then put them to `/data`.
2. Run the following command:

```shell
git clone https://github.com/ho-229/neural-networks-and-deep-learning-rs
cd neural-networks-and-deep-learning-rs
cargo run -p <NAME> --release # eg. cargo run -p mnist1 --release
```
