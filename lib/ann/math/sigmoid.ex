defmodule ANN.Math.Sigmoid do
  defmodule Definition do
    def function(x), do: 1 / (1 + :math.exp(-x))
    def derivative(x), do: function(x) * (1 - function(x))
  end

  def call(x), do: Definition.function(x)
  def call(x, :derivative), do: Definition.derivative(x)
end
