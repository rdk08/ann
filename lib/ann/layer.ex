defmodule ANN.Layer do
  alias __MODULE__, as: Layer
  alias ANN.{Neuron, Random}

  defstruct neurons: [],
            bias: nil

  @io %{random: Random}

  @type t :: %Layer{neurons: list(%Neuron{}), bias: float | nil}

  @spec build(integer, map) :: t
  def build(num_neurons, io \\ @io) when num_neurons > 0 do
    neurons = for _ <- 1..num_neurons, do: Neuron.build
    %Layer{neurons: neurons, bias: io.random.bias}
  end

  @spec update(t, map) :: t
  def update(%Layer{}=layer, %{}=changes), do: Map.merge(layer, changes)

  @spec process({t, list(float), module}) :: {t, list(float), module}
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
