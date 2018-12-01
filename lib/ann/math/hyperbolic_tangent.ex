defmodule ANN.Math.HyperbolicTangent do
  defmodule Definition do
    def function(x), do: :math.tanh(x)
    def derivative(x), do: (1 / :math.pow(:math.cosh(x), 2))
  end

  def call(x), do: Definition.function(x)
  def call(x, :derivative), do: Definition.derivative(x)
end
