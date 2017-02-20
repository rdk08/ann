defmodule ANNEx.Mixfile do
  use Mix.Project

  def project do
    [app: :annex,
     version: "0.1.0",
     elixir: "~> 1.3",
     build_embedded: Mix.env == :prod,
     start_permanent: Mix.env == :prod,
     deps: deps(),
     elixirc_paths: elixirc_paths(Mix.env)]
  end

  defp elixirc_paths(:test), do: ["lib", "test/annex"]
  defp elixirc_paths(_), do: ["lib"]

  def application do
    [applications: [:logger]]
  end

  defp deps do
    []
  end
end
