defmodule ANN.Training.Backpropagation.UpdateWeights do
  alias ANN.{Layer, Network, Neuron, Signal}

  @spec update(%Network{}, float, module) :: %Network{}
  def update(%Network{} = network, learning_rate, activation_fn) do
    layers =
      network.layers
      |> Enum.map(&Task.async(fn -> update_layer(&1, learning_rate, activation_fn) end))
      |> Enum.map(&Task.await/1)

    Network.update(network, %{layers: layers})
  end

  defp update_layer(%Layer{} = layer, learning_rate, activation_fn) do
    neurons =
      layer.neurons
      |> Enum.map(&Task.async(fn -> update_neuron(&1, layer.bias, learning_rate, activation_fn) end))
      |> Enum.map(&Task.await/1)

    Layer.update(layer, %{neurons: neurons})
  end

  defp update_neuron(%Neuron{} = neuron, bias, learning_rate, activation_fn) do
    signals =
      neuron.signals
      |> Enum.map(&Task.async(fn -> update_signal(&1, neuron.delta, neuron.sum, learning_rate, activation_fn) end))
      |> Enum.map(&Task.await/1)

    Neuron.update(neuron, %{signals: signals, sum: Signal.sum(signals, bias)})
  end

  defp update_signal(%Signal{} = signal, delta, sum, learning_rate, activation_fn) do
    weight =
      calculate_new_weight(
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
    weight + learning_rate * delta * value * activation_fn.call(sum, :derivative)
  end
end
