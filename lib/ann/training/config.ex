defmodule ANN.Training.Config do
  @enforce_keys [:method]
  defstruct method: ANN.Training.Backpropagation,
            params: %{
              learning_rate: 0.1,
              activation_fn: ANN.Math.Sigmoid
            },
            epochs: 10_000
end
