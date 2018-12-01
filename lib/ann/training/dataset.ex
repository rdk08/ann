defmodule ANN.Training.Dataset do
  defstruct input: [], output: []

  @type t :: %__MODULE__{
          input: list,
          output: list
        }
end
