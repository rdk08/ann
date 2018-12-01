defmodule ANN.LayerTest do
  use ExUnit.Case, async: true

  alias ANN.{Layer, Math, Neuron, Signal, Test}

  test "build/2 - builds layer with specified number of neurons" do
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
    assert {%Layer{bias: 0.15, neurons: [
      %Neuron{signals: [%Signal{value: 0.5}, %Signal{value: 0.2}]},
      %Neuron{signals: [%Signal{value: 0.5}, %Signal{value: 0.2}]}
    ]}, _, Math.Sigmoid} = output
  end

  test "process/1 - returns new layer struct and output values" do
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
