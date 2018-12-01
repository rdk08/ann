defmodule ANN.Test.Values.Layer do
  alias ANN.{Neuron, Layer, Signal}

  def initial do
    %Layer{
      bias: 0.15,
      neurons: [
        %Neuron{signals: [], sum: 0},
        %Neuron{signals: [], sum: 0}
      ]
    }
  end

  def initial_with_predefined_weights do
    %Layer{
      bias: 0.2,
      neurons: [
        %Neuron{
          delta: nil,
          output: nil,
          signals: [
            %Signal{value: 0.5, weight: 0.45},
            %Signal{value: 0.2, weight: 0.4}
          ],
          sum: nil
        },
        %Neuron{
          delta: nil,
          output: nil,
          signals: [
            %Signal{value: 0.5, weight: 0.5},
            %Signal{value: 0.2, weight: 0.55}
          ],
          sum: nil
        }
      ]
    }
  end

  def after_processing do
    %Layer{
      bias: 0.2,
      neurons: [
        %Neuron{
          delta: nil,
          output: 0.623633628298226,
          signals: [%Signal{value: 0.5, weight: 0.45}, %Signal{value: 0.2, weight: 0.4}],
          sum: 0.5050000000000001
        },
        %Neuron{
          delta: nil,
          output: 0.6364525402815664,
          signals: [%Signal{value: 0.5, weight: 0.5}, %Signal{value: 0.2, weight: 0.55}],
          sum: 0.56
        }
      ]
    }
  end
end
