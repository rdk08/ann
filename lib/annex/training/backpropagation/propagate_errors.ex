defmodule ANNEx.Training.Backpropagation.PropagateErrors do
  alias ANNEx.{Layer, Network, Neuron, Signal}

  @spec propagate(%Network{}, list(float), list(float), module) :: %Network{}
  def propagate(network, output, expected_output, activation_fn) do
    layers = Enum.reverse(network.layers)
    initial_deltas = calculate_initial_deltas(output, expected_output)
    updated_layers = propagate_deltas(layers, initial_deltas, [], activation_fn)
    Network.update(network, %{layers: updated_layers})
  end

  defp calculate_initial_deltas(output, expected_output) do
    output
    |> Enum.zip(expected_output)
    |> Enum.map(fn {output, expected_output} -> expected_output - output end)
  end

  defp propagate_deltas([layer|rest], deltas, processed_layers, activation_fn) do
    neurons =
      layer.neurons
      |> Enum.zip(deltas)
      |> Enum.map(fn {neuron, delta} -> Neuron.update(neuron, %{delta: delta}) end)
    updated_layer = Layer.update(layer, %{neurons: neurons})
    previous_deltas = calculate_previous_deltas(updated_layer, deltas, activation_fn)
    propagate_deltas(rest, previous_deltas, [updated_layer|processed_layers], activation_fn)
  end
  defp propagate_deltas([], _, processed_layers, _) do
    processed_layers
  end

  defp calculate_previous_deltas(%Layer{neurons: neurons}, deltas, activation_fn) do
    weights = Enum.map(neurons, &Signal.get_weights(&1.signals))
    sums = Enum.map(neurons, &(&1.sum))
    [weights, deltas, sums]
    |> Enum.zip
    |> Enum.map(&calculate_delta_fractions(&1, activation_fn))
    |> sum_delta_fractions
  end

  defp calculate_delta_fractions({weights, delta, sum}, activation_fn) do
    Enum.map(weights, &(&1 * delta * activation_fn.call(sum, :derivative)))
  end

  defp sum_delta_fractions(delta_fractions) do
    delta_fractions
    |> Enum.zip
    |> Enum.map(&Tuple.to_list(&1))
    |> Enum.map(&Enum.reduce(&1, 0, fn (fraction, acc) -> fraction + acc end))
  end
end
