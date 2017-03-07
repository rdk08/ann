defmodule ANNEx.Network do
  alias __MODULE__, as: Network
  alias ANNEx.{Layer, Random}

  defstruct layers: [],
            activation_fn: nil

  @io %{random: Random}

  @doc """
  Builds initial network.
  """
  def build(%Network.Config{}=config, io \\ @io) do
    layers = Enum.map(config.layers, &Layer.build(&1, io))
    %Network{layers: layers, activation_fn: config.activation_fn}
  end

  @doc """
  Updates network state.
  """
  def update(%Network{}=network, %{}=changes) do
    Map.merge(network, changes)
  end

  @doc """
  Processes input values through network, returns new network state.
  """
  def process(%Network{layers: layers, activation_fn: activation_fn}=network, input) do
    {processed_layers, output} = _process(layers, input, activation_fn, [])
    {Network.update(network, %{layers: processed_layers}), output}
  end

  @doc """
  Same as process/2 but returns only network output.
  """
  def process!(%Network{layers: layers, activation_fn: activation_fn}, input) do
    {_, output} = _process(layers, input, activation_fn, [])
    output
  end

  defp _process([layer|remaining], values, activation_fn, processed_layers) do
    {layer, output, activation_fn} = Layer.process({layer, values, activation_fn})
    _process(remaining, output, activation_fn, [layer|processed_layers])
  end
  defp _process([], output, _, processed_layers) do
    {Enum.reverse(processed_layers), output}
  end
end
