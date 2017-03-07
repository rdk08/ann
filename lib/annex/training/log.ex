defmodule ANNEx.Training.Log do
  @spec epoch(keyword, integer, list(float)) :: :ok | :noop
  def epoch(log_opts, epoch_num, errors) do
    case log_opts[:epoch_info] do
      true -> IO.puts(epoch_info(epoch_num, errors))
      _ -> :noop
    end
  end

  @spec epoch_info(integer, list(float)) :: String.t
  def epoch_info(epoch_num, errors) do
    [format_epoch_num(epoch_num), format_average_error(errors)]
    |> Enum.join(", ")
  end

  defp format_epoch_num(epoch_num), do: "epoch: #{epoch_num}"
  defp format_average_error(errors), do: "avg err: #{average_error(errors)}"

  defp average_error(errors) do
    Enum.reduce(errors, 0, &(&2 + &1)) / length(errors)
  end

  @spec iteration(keyword, list(float), list(float), list(float), float) :: :ok | :noop
  def iteration(log_opts, input, output, exp_output, total_error) do
    case log_opts[:iteration_info] do
      true -> IO.puts(iteration_info(input, output, exp_output, total_error))
      _ -> :noop
    end
  end

  @spec iteration_info(list(float), list(float), list(float), float) :: String.t
  def iteration_info(input, output, exp_output, total_error) do
    [format_input(input),
     format_output(output),
     format_exp_output(exp_output),
     format_total_error(total_error)]
    |> Enum.join(" | ")
  end

  defp format_input(input), do: "in: [#{format_list(input)}]"
  defp format_output(output), do: "out: [#{format_list(output)}]"
  defp format_exp_output(exp_output), do: "exp out: [#{format_list(exp_output)}]"
  defp format_total_error(total_error), do: "err: #{total_error}"
  defp format_list(list), do: Enum.join(list, ", ")
end
