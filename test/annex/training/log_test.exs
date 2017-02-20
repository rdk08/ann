defmodule ANNEx.Training.LogTest do
  use ExUnit.Case, async: true
  alias ANNEx.Training.Log

  test "_epoch/2" do
    input = %{
      epoch_num: 150,
      errors: [0.229, 0.119]
    }
    output = Log._epoch(input.epoch_num, input.errors)
    expected_output = "epoch: 150, avg err: 0.174"
    assert output == expected_output
  end

  test "_iteration/4" do
    input = %{
      input: [0.05, 0.10],
      output: [0.77, 0.75],
      exp_output: [0.01, 0.99],
      total_error: 0.3176,
    }
    output = Log._iteration(input.input, input.output, input.exp_output, input.total_error)
    expected_output = "in: [0.05, 0.1] | out: [0.77, 0.75] | exp out: [0.01, 0.99] | err: 0.3176"
    assert output == expected_output
  end
end
