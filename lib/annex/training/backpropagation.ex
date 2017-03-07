defmodule ANNEx.Training.Backpropagation do
  alias ANNEx.Network
  alias ANNEx.Training.Backpropagation.{PropagateErrors, UpdateWeights}

  @doc """
  Runs one iteration of backpropagation algorithm.
  """
  @spec process(%Network{}, list(float), list(float), map) :: %Network{}
  def process(network, output, expected_output, params) do
    network
    |> PropagateErrors.propagate(output, expected_output, params.activation_fn)
    |> UpdateWeights.update(params.learning_rate, params.activation_fn)
  end
end
