defmodule ANNEx.Math.SigmoidTest do
  use ExUnit.Case
  alias ANNEx.Math.Sigmoid

  test "sigmoid function" do
    input = %{
      values: [-6, -1, 0, 1 , 6]
    }
    output = Enum.map(input.values, &(Sigmoid.call(&1)))
    expected_output = [
      0.0024726231566347743,
      0.2689414213699951,
      0.5,
      0.7310585786300049,
      0.9975273768433653
    ]
    assert output == expected_output
  end

  test "derivative of sigmoid function" do
    input = %{
      values: [-6, -1, 0, 1 , 6]
    }
    output = Enum.map(input.values, &(Sigmoid.call(:derivative, &1)))
    expected_output = [
      0.002466509291360048,
      0.19661193324148185,
      0.25,
      0.19661193324148185,
      0.002466509291359931,
    ]
    assert output == expected_output
  end
end
