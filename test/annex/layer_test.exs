defmodule ANNEx.LayerTest do
  use ExUnit.Case
  alias ANNEx.{Layer, Math, Neuron, Signal, Test}

  test "build/1" do
    input = %{
      io: %{random: Test.Fake.Random}
    }
    output = Layer.build(2, input.io)
    expected_output = Test.Values.Layer.initial
    assert output == expected_output
  end

  test "process/1 - processes layer with initial neurons" do
    input = %{
      tuple: {
        Test.Values.Layer.initial,
        [0.5, 0.2],
        Math.Sigmoid
      }
    }
    output = Layer.process(input.tuple)
    assert {
      %Layer{
        bias: 0.15,
        neurons: [
          %Neuron{
            delta: _,
            output: _,
            signals: [%Signal{value: 0.5, weight: _}, %Signal{value: 0.2, weight: _}],
            sum: _
          },
          %Neuron{
            delta: _,
            output: _,
            signals: [%Signal{value: 0.5, weight: _}, %Signal{value: 0.2, weight: _}],
            sum: _
          }
        ]
      },
      _,
      Math.Sigmoid
    } = output
  end

  test "process/1" do
    input = %{
      tuple: {
        Test.Values.Layer.initial_with_predefined_weights,
        [0.5, 0.2],
        Math.Sigmoid
      }
    }
    output = Layer.process(input.tuple)
    expected_output = {
      Test.Values.Layer.after_processing,
      [0.623633628298226, 0.6364525402815664],
      Math.Sigmoid
    }
    assert output == expected_output
  end
end
