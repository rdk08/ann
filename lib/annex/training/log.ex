defmodule ANNEx.Training.Log do
  @doc """
  Prints epoch summary.
  """
  def epoch(log_opts, epoch_num, errors) do
    case log_opts[:epoch_info] do
      true -> IO.puts(_epoch(epoch_num, errors))
      _ -> :noop
    end
  end

  def _epoch(epoch_num, errors) do
    [format_epoch_num(epoch_num), format_average_error(errors)]
    |> Enum.join(", ")
  end

  defp format_epoch_num(epoch_num), do: "epoch: #{epoch_num}"
  defp format_average_error(errors), do: "avg err: #{average_error(errors)}"

  defp average_error(errors) do
    Enum.reduce(errors, 0, &(&2 + &1)) / length(errors)
  end

  @doc """
  Prints all information about specific iteration.
  """
  def iteration(log_opts, input, output, exp_output, total_error) do
    case log_opts[:iteration_info] do
      true -> IO.puts(_iteration(input, output, exp_output, total_error))
      _ -> :noop
    end
  end

  def _iteration(input, output, exp_output, total_error) do
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
