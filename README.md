## ANNEx ##

ANNEx is a simple artificial neural network implementation written in Elixir. It uses backpropagation algorithm and allows to configure basic network parameters (e.g. network structure, activation function, learning rate).

#### Building network ####

```elixir
config = %ANNEx.Network.Config{layers: [5, 1], activation_fn: ANNEx.Math.Sigmoid}
network = ANNEx.Network.build(config)
```

`network` is a simple struct that represents network state and can be inspected at any time.

Note: you can generate different network structure by changing `layers` list (values represent number of neurons in consecutive layers).

#### Processing input values ####

```elixir
input_values = [1.0, 0.0]
{network, output} = ANNEx.Network.process(network, input_values)
```

#### Training ####

```elixir
training_config = %ANNEx.Training.Config{
  method: ANNEx.Training.Backpropagation,
  params: %{learning_rate: 0.5, activation_fn: ANNEx.Math.Sigmoid},
  epochs: 10_000
}
training_dataset = [
  {[1.0, 0.0], [1.0]},
  {[0.0, 1.0], [1.0]},
  {[0.0, 0.0], [0.0]},
  {[1.0, 1.0], [0.0]}
]

trained_network = ANNEx.Training.train(network, training_config, training_dataset)

# Note: to see training progress specify log options, e.g.:
# log_opts = [epoch_info: true, iteration_info: true]
# trained_network = ANNEx.Training.train(network, training_config, training_dataset, log_opts)

# Verify output, e.g.:
output = ANNEx.Network.process!(trained_network, [1.0, 0.0])
```
