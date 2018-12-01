defmodule ANN.Signal do
  alias __MODULE__, as: Signal
  alias ANN.Random

  defstruct value: nil,
            weight: nil

  @io %{random: Random}

  @type t :: %Signal{value: float, weight: float}

  @spec build(float, map) :: t
  def build(value, io \\ @io) do
    %Signal{value: value, weight: io.random.weight}
  end

  @spec update(nonempty_list(float), nonempty_list(map)) :: nonempty_list(%Signal{})
  def update([_|_]=signals, [_|_]=changes) do
    signals
    |> Enum.zip(changes)
    |> Enum.map(fn {signal, change} -> Signal.update(signal, change) end)
  end
  @spec update(t, map) :: t
  def update(%Signal{}=signal, %{}=changes) do
    Map.merge(signal, changes)
  end

  @spec sum(nonempty_list(t), float) :: float
  def sum(signals, bias) do
    sum = Enum.reduce(signals, 0, &(&2 + (&1.value * &1.weight)))
    sum + bias
  end

  @spec get_weights(nonempty_list(t)) :: nonempty_list(float)
  def get_weights(signals), do: Enum.map(signals, &(&1.weight))
end
