defmodule ANNEx.Random do
  def weight do
    sign() * 0.5 * :rand.uniform
  end

  def bias do
    0.25 * :rand.uniform
  end

  defp sign do
    if :rand.uniform > 0.5, do: 1, else: -1
  end
end
