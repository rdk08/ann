defmodule ANNEx.Layer do
  alias __MODULE__, as: Layer
  alias ANNEx.{Neuron, Random}

  defstruct bias: nil,
            neurons: []

  @io %{random: Random}

  @doc """
  Builds layer.
  """
  def build(num_neurons, io \\ @io) when num_neurons > 0 do
    neurons = Enum.map(1..num_neurons, fn (_) -> Neuron.build end)
    %Layer{neurons: neurons, bias: io.random.bias}
  end

  @doc """
  Updates layer state.
  """
  def update(%Layer{}=layer, %{}=changes), do: Map.merge(layer, changes)

  @doc """
  Processes layer and returns new layer state.
  """
  def process({layer, values, activation_fn}) do
    neurons =
      layer.neurons
      |> Enum.map(&Task.async(fn ->
         Neuron.process(&1, values, layer.bias, activation_fn)
      end))
      |> Enum.map(&Task.await/1)
    outputs = Enum.map(neurons, &(&1.output))
    {Layer.update(layer, %{neurons: neurons}), outputs, activation_fn}
  end
end
