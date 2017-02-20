defmodule ANNEx.Training.TrainingTest do
  use ExUnit.Case, async: true
  alias ANNEx.Training.{Backpropagation, Config}
  alias ANNEx.Training
  alias ANNEx.{Math, Test}

  test "train/3 - signle dataset (1 epoch, backpropagation)" do
    input = %{
      network: Test.Values.Network.before_backpropagation,
      training_config: %Config{
        method: Backpropagation,
        params: %{
          learning_rate: 0.5,
          activation_fn: Math.Sigmoid
        },
        epochs: 1
      },
      training_dataset: {[0.05, 0.10], [0.01, 0.99]},
      log_opts: []
    }
    output = Training.train(
      input.network,
      input.training_config,
      input.training_dataset,
      input.log_opts
    )
    expected_output = Test.Values.Network.after_backpropagation
    assert output == expected_output
  end

  test "train/3 - single dataset (100 epochs, backpropagation)" do
    input = %{
      network: Test.Values.Network.before_backpropagation,
      training_config: %Config{
        method: Backpropagation,
        params: %{
          learning_rate: 0.5,
          activation_fn: Math.Sigmoid
        },
        epochs: 100
      },
      training_dataset: {[0.05, 0.10], [0.01, 0.99]},
      log_opts: []
    }
    output = Training.train(
      input.network,
      input.training_config,
      input.training_dataset,
      input.log_opts
    )
    expected_output = Test.Values.Network.after_training
    assert output == expected_output
  end

  test "train/3 - multiple datasets (1 epoch, backpropagation)" do
    input = %{
      network: Test.Values.Network.before_backpropagation,
      training_config: %Config{
        method: Backpropagation,
        params: %{
          learning_rate: 0.5,
          activation_fn: Math.Sigmoid
        },
        epochs: 1
      },
      training_datasets: [
        {[0.01, 0.01], [0.99, 0.99]},
        {[0.20, 0.20], [0.01, 0.01]},
        {[0.99, 0.99], [0.99, 0.99]}
      ],
      log_opts: []
    }
    output = Training.train(
      input.network,
      input.training_config,
      input.training_datasets,
      input.log_opts
    )
    expected_output = Test.Values.Network.after_backpropagation_multiple_datasets
    assert output == expected_output
  end
end
