defmodule ANN.Network do
  alias __MODULE__, as: Network
  alias ANN.{Layer, Random}

  defstruct layers: [],
            activation_fn: nil

  @io %{random: Random}

  @type t :: %Network{layers: list(%Layer{}), activation_fn: module}

  @spec build(struct, map) :: t
  def build(%Network.Config{} = config, io \\ @io) do
    layers = Enum.map(config.layers, &Layer.build(&1, io))
    %Network{layers: layers, activation_fn: config.activation_fn}
  end

  @spec update(t, map) :: t
  def update(%Network{} = network, %{} = changes) do
    Map.merge(network, changes)
  end

  @doc """
  Processes input values through network, returns tuple containing new network
  state and network output.
  """
  @spec process(t, list(float)) :: {t, list(float)}
  def process(%Network{layers: layers, activation_fn: activation_fn} = network, input) do
    {processed_layers, output} = process_layers(layers, input, activation_fn, [])
    {Network.update(network, %{layers: processed_layers}), output}
  end

  @doc """
  Same as process/2 but returns only network output.
  """
  @spec process!(t, list(float)) :: list(float)
  def process!(%Network{layers: layers, activation_fn: activation_fn}, input) do
    {_, output} = process_layers(layers, input, activation_fn, [])
    output
  end

  defp process_layers([layer | remaining], input, activation_fn, processed_layers) do
    {layer, output, activation_fn} = Layer.process({layer, input, activation_fn})
    process_layers(remaining, output, activation_fn, [layer | processed_layers])
  end

  defp process_layers([], output, _, processed_layers) do
    {Enum.reverse(processed_layers), output}
  end
end
