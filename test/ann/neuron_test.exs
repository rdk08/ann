defmodule ANN.NeuronTest do
  use ExUnit.Case, async: true

  alias ANN.{Math, Neuron, Signal}
  alias ANN.Test.Fake

  test "build/0 - builds initial neuron" do
    output = Neuron.build()
    expected_output = %Neuron{signals: [], sum: 0}
    assert output == expected_output
  end

  test "build/2 - builds neuron with signals" do
    input = %{
      signal_values: [0.5, 0.8, 0.25],
      io: %{random: Fake.Random}
    }

    output = Neuron.build(input.signal_values, input.io)

    expected_output = %Neuron{
      delta: nil,
      output: 0,
      signals: [
        %Signal{value: 0.5, weight: 0.35},
        %Signal{value: 0.8, weight: 0.35},
        %Signal{value: 0.25, weight: 0.35}
      ],
      sum: 0
    }

    assert output == expected_output
  end

  test "process/4 - processes input values (returns new neuron state)" do
    input = %{
      neuron: %Neuron{
        signals: [
          %Signal{value: 1, weight: 1},
          %Signal{value: 1, weight: 1}
        ],
        sum: 0.8807970779778823
      },
      values: [2, 2],
      bias: 0.5,
      activation_fn: Math.Sigmoid
    }

    output = Neuron.process(input.neuron, input.values, input.bias, input.activation_fn)

    expected_output = %Neuron{
      signals: [
        %Signal{value: 2, weight: 1},
        %Signal{value: 2, weight: 1}
      ],
      sum: 4.5,
      output: 0.9890130573694068
    }

    assert output == expected_output
  end
end
