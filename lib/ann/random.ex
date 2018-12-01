defmodule ANN.Random do
  @spec weight() :: float
  def weight do
    sign() * 0.5 * :rand.uniform
  end

  defp sign do
    if :rand.uniform > 0.5, do: 1, else: -1
  end

  @spec bias() :: float
  def bias do
    0.25 * :rand.uniform
  end
end
