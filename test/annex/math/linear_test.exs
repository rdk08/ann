defmodule ANNEx.Math.LinearTest do
  use ExUnit.Case
  alias ANNEx.Math.Linear

  test "linear function" do
    input = %{
      values: [-6, -1, 0, 1 , 6]
    }
    output = Enum.map(input.values, &(Linear.call(&1)))
    expected_output = [
      -6,
      -1,
      0,
      1,
      6
    ]
    assert output == expected_output
  end

  test "derivative of linear function" do
    input = %{
      values: [-6, -1, 0, 1 , 6]
    }
    output = Enum.map(input.values, &(Linear.call(:derivative, &1)))
    expected_output = [
      1,
      1,
      1,
      1,
      1
    ]
    assert output == expected_output
  end
end