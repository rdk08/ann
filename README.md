A simple artificial neural network implementation written in Elixir. It uses backpropagation algorithm and allows to configure basic network parameters (e.g. network structure, activation function, learning rate).

#### Building network ####

```elixir
config = %ANN.Network.Config{layers: [5, 1], activation_fn: ANN.Math.Sigmoid}
network = ANN.Network.build(config)
```

`network` is a simple struct that represents network state and can be inspected at any time.

Note: you can generate different network structure by changing `layers` list (values represent number of neurons in consecutive layers).

#### Processing input values ####

```elixir
input_values = [1.0, 0.0]
{network, output} = ANN.Network.process(network, input_values)
```

#### Training ####

```elixir
alias ANN.Training.Dataset

training_config = %ANN.Training.Config{
  method: ANN.Training.Backpropagation,
  params: %{learning_rate: 0.5, activation_fn: ANN.Math.Sigmoid},
  epochs: 10_000
}
training_dataset = [
  %Dataset{input: [1.0, 0.0], output: [1.0]},
  %Dataset{input: [0.0, 1.0], output: [1.0]},
  %Dataset{input: [0.0, 0.0], output: [0.0]},
  %Dataset{input: [1.0, 1.0], output: [0.0]}
]

# Note: to see training progress specify log options, e.g.:
# log_opts = [epoch_info: true, iteration_info: true]
# trained_network = ANN.Training.train(network, training_config, training_dataset, log_opts)

trained_network = ANN.Training.train(network, training_config, training_dataset)

# Verify output, e.g.:
output = ANN.Network.process!(trained_network, [1.0, 0.0])
```
