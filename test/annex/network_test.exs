defmodule ANNEx.NetworkTest do
  use ExUnit.Case
  alias ANNEx.{Math, Network}
  alias ANNEx.Test.{Fake, Values}

  test "build/1 - builds network structure based on provided config struct" do
    input = %{
      config: %Network.Config{
        layers: [2, 3, 2],
        activation_fn: Math.Sigmoid
      },
      io: %{random: Fake.Random}
    }
    output = Network.build(input.config, input.io)
    expected_output = Values.Network.initial
    assert output == expected_output
  end

  test "process/2 - returns new network state with output values" do
    input = %{
      network: Values.Network.initial_with_predefined_weights,
      values: [0.2, 0.5]
    }
    output = Network.process(input.network, input.values)
    expected_output = {
      Values.Network.processed,
      [0.6473249418260394, 0.6972554089636754]
    }
    assert output == expected_output
  end

  test "process!/2 - returns network output values for given input" do
    input = %{
      network: Values.Network.initial_with_predefined_weights,
      values: [0.2, 0.5]
    }
    output = Network.process!(input.network, input.values)
    expected_output = [0.6473249418260394, 0.6972554089636754]

    assert output == expected_output
  end
end