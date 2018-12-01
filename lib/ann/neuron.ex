defmodule ANN.Neuron do
  alias __MODULE__, as: Neuron
  alias ANN.{Signal, Random}

  defstruct signals: [],
            sum: 0.0,
            output: 0.0,
            delta: nil

  @io %{random: Random}

  @type t :: %Neuron{signals: list(%Signal{}), sum: float, output: float, delta: float | nil}

  @spec build() :: t
  def build do
    %Neuron{}
  end

  @spec build(nonempty_list(float), map) :: t
  def build([_ | _] = signal_values, io \\ @io) do
    signals = Enum.map(signal_values, &Signal.build(&1, io))
    %Neuron{signals: signals}
  end

  @spec update(t, map) :: t
  def update(%Neuron{} = neuron, %{} = changes), do: Map.merge(neuron, changes)

  @spec process(t, nonempty_list(float), float, module, map) :: t
  def process(neuron, values, bias, activation_fn, io \\ @io) do
    signals = build_or_update_signals(neuron.signals, values, io)
    changes = calculate_neuron_changes(signals, bias, activation_fn)
    Neuron.update(neuron, changes)
  end

  defp build_or_update_signals([], values, io) do
    Enum.map(values, &Signal.build(&1, io))
  end

  defp build_or_update_signals([_ | _] = signals, values, _io) do
    signal_changes = Enum.map(values, &%{value: &1})
    Signal.update(signals, signal_changes)
  end

  defp calculate_neuron_changes(signals, bias, activation_fn) do
    sum = Signal.sum(signals, bias)
    output = activation_fn.call(sum)
    %{signals: signals, sum: sum, output: output}
  end
end
