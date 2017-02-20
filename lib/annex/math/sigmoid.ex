defmodule ANNEx.Math.Sigmoid do
  defmodule Definition do
    def function(x), do: (1 / (1 + :math.exp(-x)))
    def derivative(x), do: function(x) * (1 - function(x))
  end

  def call(x), do: Definition.function(x)
  def call(:derivative, x), do: Definition.derivative(x)
end
