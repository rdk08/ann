defmodule ANNEx.Math.Linear do
  defmodule Definition do
    def function(x), do: x
    def derivative(_), do: 1
  end

  def call(x), do: Definition.function(x)
  def call(:derivative, x), do: Definition.derivative(x)
end
