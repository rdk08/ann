defmodule ANNEx.Neuron do
  alias __MODULE__, as: Neuron
  alias ANNEx.{Signal, Random}

  defstruct signals: [],
            sum: 0,
            output: 0,
            delta: nil

  @io %{random: Random}

  @doc """
  Builds initial neuron.
  """
  def build, do: %Neuron{}
  def build([_|_]=signal_values, io \\ @io) do
    %Neuron{signals: Signal.build(signal_values, io)}
  end

  @doc """
  Updates neuron.
  """
  def update(%Neuron{}=neuron, %{}=changes), do: Map.merge(neuron, changes)

  @doc """
  Process signal values and returns new neuron state.
  """
  def process(neuron, values, bias, activation_fn, io \\ @io)
  def process(%Neuron{signals: []}=neuron, values, bias, activation_fn, io) do
    signals = Signal.build(values, io)
    changes = _process(signals, bias, activation_fn)
    Neuron.update(neuron, changes)
  end
  def process(%Neuron{signals: signals}=neuron, values, bias, activation_fn, _io)
      when length(signals) == length(values) do

    signal_changes = Enum.map(values, &(%{value: &1}))
    signals = Signal.update(signals, signal_changes)
    changes = _process(signals, bias, activation_fn)
    Neuron.update(neuron, changes)
  end

  defp _process(signals, bias, activation_fn) do
    sum = Signal.sum(signals, bias)
    output = activation_fn.call(sum)
    %{signals: signals, sum: sum, output: output}
  end
end
