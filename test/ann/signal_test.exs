defmodule ANN.SignalTest do
  use ExUnit.Case, async: true

  alias ANN.Signal
  alias ANN.Test.Fake

  test "build/2 - build initial signal" do
    input = %{
      value: 0.55,
      io: %{random: Fake.Random}
    }
    output = Signal.build(input.value, input.io)
    expected_output = %Signal{value: 0.55, weight: 0.35}
    assert output == expected_output
  end

  test "update/2 - updates signals with new values" do
    input = %{
      signals: [
        %Signal{value: 1, weight: 0.5},
        %Signal{value: 0, weight: 0.5}
      ],
      changes: [
        %{value: 2},
        %{value: 1},
      ]
    }
    output = Signal.update(input.signals, input.changes)
    expected_output = [
      %Signal{value: 2, weight: 0.5},
      %Signal{value: 1, weight: 0.5}
    ]
    assert output == expected_output
  end

  test "sum/2 - sums all signal values and bias" do
    input = %{
      signals: [
        %Signal{value: 1, weight: 0.75},
        %Signal{value: 2, weight: 0.5},
        %Signal{value: 3, weight: 0.2}
      ],
      bias: 0.5
    }
    output = Signal.sum(input.signals, input.bias)
    expected_output = 2.85
    assert output == expected_output
  end
end
