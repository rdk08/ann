defmodule ANNEx.RandomTest do
  use ExUnit.Case, async: true

  alias ANNEx.Random

  test "weight/0 - returns random weight value" do
    assert Random.weight >= -0.5
    assert Random.weight <= 0.5
  end

  test "bias/0 - returns random bias value" do
    assert Random.bias >= 0.0
    assert Random.bias <= 0.25
  end
end
