defmodule ANN.Math.Linear do
  defmodule Definition do
    def function(x), do: x
    def derivative(_), do: 1
  end

  def call(x), do: Definition.function(x)
  def call(x, :derivative), do: Definition.derivative(x)
end
