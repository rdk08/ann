defmodule ANNEx.Training.Backpropagation do
  alias ANNEx.{Layer, Network, Neuron, Signal}

  @doc """
  Runs one iteration of backpropagation algorithm (updates signal weights).
  """
  def process(network, output, expected_output, params) do
    deltas = calculate_initial_deltas(output, expected_output)
    layers = network.layers
             |> Enum.reverse
             |> propagate_deltas(deltas, [], params.activation_fn)
             |> Enum.map(fn (layer) ->
                update_weights(layer, params.learning_rate, params.activation_fn)
             end)
    Network.update(network, %{layers: layers})
  end

  defp calculate_initial_deltas(output, expected_output) do
    output
    |> Enum.zip(expected_output)
    |> Enum.map(fn {output, expected_output} -> expected_output - output end)
  end

  defp calculate_previous_deltas(%Layer{neurons: neurons}, deltas, activation_fn) do
    weights = Enum.map(neurons, &(Signal.get_weights(&1.signals)))
    sums = Enum.map(neurons, &(&1.sum))
    [weights, deltas, sums]
    |> Enum.zip
    |> Enum.map(fn {weights, delta, sum} ->
       calculate_delta_fractions(weights, delta, sum, activation_fn)
    end)
    |> sum_delta_fractions
  end

  defp calculate_delta_fractions(weights, delta, sum, activation_fn) do
    Enum.map(weights, fn (weight) ->
      weight * delta * activation_fn.call(:derivative, sum)
    end)
  end

  defp sum_delta_fractions(delta_fractions) do
    delta_fractions
    |> Enum.zip
    |> Enum.map(&(Tuple.to_list(&1)))
    |> Enum.map(&(Enum.reduce(&1, 0, fn (delta_fraction, acc) ->
      delta_fraction + acc
    end)))
  end

  defp propagate_deltas([layer|rest], deltas, processed_layers, activation_fn) do
    neurons = deltas
              |> Enum.zip(layer.neurons)
              |> Enum.map(fn {delta, neuron} ->
                 Neuron.update(neuron, %{delta: delta})
              end)
    updated_layer = Layer.update(layer, %{neurons: neurons})
    previous_deltas = calculate_previous_deltas(updated_layer, deltas, activation_fn)
    propagate_deltas(rest, previous_deltas, [updated_layer|processed_layers], activation_fn)
  end
  defp propagate_deltas([], _, processed_layers, _) do
    processed_layers
  end

  defp update_weights(%Layer{}=layer, learning_rate, activation_fn) do
    neurons = Enum.map(layer.neurons, fn (neuron) ->
                update_weights(neuron, layer.bias, learning_rate, activation_fn)
              end)
    Layer.update(layer, %{neurons: neurons})
  end
  defp update_weights(%Neuron{}=neuron, bias, learning_rate, activation_fn) do
    signals = Enum.map(neuron.signals, fn (signal) ->
                update_weights(signal, neuron.delta, neuron.sum, learning_rate, activation_fn)
              end)
    Neuron.update(neuron, %{signals: signals, sum: Signal.sum(signals, bias)})
  end
  defp update_weights(%Signal{}=signal, delta, sum, learning_rate, activation_fn) do
    weight = calculate_new_weight(
      signal.weight,
      signal.value,
      delta,
      sum,
      learning_rate,
      activation_fn
    )
    Signal.update(signal, %{weight: weight})
  end

  defp calculate_new_weight(weight, value, delta, sum, learning_rate, activation_fn) do
    weight + learning_rate*delta*value*activation_fn.call(:derivative, sum)
  end
end
