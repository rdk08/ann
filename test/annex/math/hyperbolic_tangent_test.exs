defmodule ANNEx.Math.HyperbolicTangentTest do
  use ExUnit.Case, async: true

  alias ANNEx.Math.HyperbolicTangent

  test "hyperbolic tangent function" do
    input = %{
      values: [-6, -1, 0, 1 , 6]
    }
    output = Enum.map(input.values, &HyperbolicTangent.call/1)
    expected_output = [
      -0.9999877116507956,
      -0.7615941559557649,
      0.0,
      0.7615941559557649,
      0.9999877116507956
    ]
    assert output == expected_output
  end

  test "derivative of hyperbolic tangent function" do
    input = %{
      values: [-6, -1, 0, 1 , 6]
    }
    output = Enum.map(input.values, &HyperbolicTangent.call(&1, :derivative))
    expected_output = [
      2.45765474053327e-5,
      0.4199743416140261,
      1.0,
      0.4199743416140261,
      2.45765474053327e-5
    ]
    assert output == expected_output
  end
end
