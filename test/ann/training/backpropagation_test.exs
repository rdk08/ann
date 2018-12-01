defmodule ANN.Training.BackpropagationTest do
  use ExUnit.Case, async: true

  alias ANN.{Math, Network}
  alias ANN.Test.Values
  alias ANN.Training.Backpropagation

  setup do
    input = [0.05, 0.10]
    {network, output} = Network.process(Values.Network.before_backpropagation, input)
    {:ok, network: network, output: output}
  end

  test "process/4 - updates network with backpropagation algorithm",
       %{network: network, output: output} do

    input = %{
      network: network,
      output: output,
      exp_output: [0.01, 0.99],
      params: %{
        learning_rate: 0.5,
        activation_fn: Math.Sigmoid,
      }
    }
    output = Backpropagation.process(
      input.network,
      input.output,
      input.exp_output,
      input.params
    )
    expected_output = Values.Network.after_backpropagation
    assert output == expected_output
  end
end
