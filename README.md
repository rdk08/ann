A simple artificial neural network implementation written in Elixir. It uses backpropagation algorithm and allows to configure basic network parameters (e.g. network structure, activation function, learning rate).

#### Building network ####

```elixir
config = %ANN.Network.Config{layers: [5, 1], activation_fn: ANN.Math.Sigmoid}
network = ANN.Network.build(config)
```

`network` is a simple struct that represents network state and can be inspected at any time.

<<<<<<< HEAD
Note: you can generate different network structure by changing `layers` list (values represent number of neurons in consecutive layers).
=======
Note: you can generate different network structure by changing `layers` list (values represents number of neurons in consecutive layers).
>>>>>>> add information to README file

#### Processing input values ####

```elixir
input_values = [1.0, 0.0]
{network, output} = ANN.Network.process(network, input_values)
```

#### Training ####

```elixir
training_config = %ANN.Training.Config{
  method: ANN.Training.Backpropagation,
  params: %{learning_rate: 0.5, activation_fn: ANN.Math.Sigmoid},
  epochs: 10_000
}
training_dataset = [
  {[1.0, 0.0], [1.0]},
  {[0.0, 1.0], [1.0]},
  {[0.0, 0.0], [0.0]},
  {[1.0, 1.0], [0.0]}
]

trained_network = ANN.Training.train(network, training_config, training_dataset)

# Note: to see training progress specify log options, e.g.:
# log_opts = [epoch_info: true, iteration_info: true]
# trained_network = ANN.Training.train(network, training_config, training_dataset, log_opts)

# Verify output, e.g.:
output = ANN.Network.process!(trained_network, [1.0, 0.0])
```
