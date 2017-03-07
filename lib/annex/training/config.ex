defmodule ANNEx.Training.Config do
  @enforce_keys [:method]
  defstruct method: ANNEx.Training.Backpropagation,
            params: %{
              learning_rate: 0.1,
              activation_fn: ANNEx.Math.Sigmoid
            },
            epochs: 10_000
end
