defmodule ANN.Training do
  alias ANN.Network
  alias ANN.Training.{Config, Log}

  @doc """
  Trains network based on provided configuration and datasets.
  Returns new network state.
  """
  @spec train(%Network{}, %Config{}, list(tuple) | tuple, keyword) :: %Network{}
  def train(%Network{} = network, training_config, training_datasets, log_opts \\ []) do
    %{method: method, epochs: epochs, params: params} = training_config

    {network, _} =
      Enum.reduce(1..epochs, {network, 1}, fn _, {network, epoch_num} ->
        {network, errors} = epoch(network, method, params, training_datasets, log_opts)
        Log.epoch(log_opts, epoch_num, errors)
        {network, epoch_num + 1}
      end)

    network
  end

  defp epoch(network, method, params, [_ | _] = training_datasets, log_opts) do
    Enum.reduce(training_datasets, {network, []}, fn dataset, {network, errors} ->
      {network, error} = iteration(network, method, params, dataset, log_opts)
      {network, [error | errors]}
    end)
  end

  defp epoch(network, method, params, training_dataset, log_opts) do
    {network, error} = iteration(network, method, params, training_dataset, log_opts)
    {network, [error]}
  end

  defp iteration(network, method, params, {input, exp_output}, log_opts) do
    {network, output} = Network.process(network, input)
    network = method.process(network, output, exp_output, params)
    output_after_training = Network.process!(network, input)
    total_error = calculate_total_error(output_after_training, exp_output)
    Log.iteration(log_opts, input, output_after_training, exp_output, total_error)
    {network, total_error}
  end

  defp calculate_total_error(output, exp_output) do
    output
    |> Enum.zip(exp_output)
    |> Enum.map(&calculate_squared_error/1)
    |> Enum.reduce(0, &(&2 + &1))
  end

  defp calculate_squared_error({output, exp_output}) do
    0.5 * :math.pow(exp_output - output, 2)
  end
end
