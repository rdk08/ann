defmodule ANNEx.Test.Values.Network do
  alias ANNEx.{Network, Math, Neuron, Layer, Signal}

  def initial do
    %Network{
      activation_fn: Math.Sigmoid,
      layers: [
        %Layer{
          neurons: [
            %Neuron{signals: [], sum: 0},
            %Neuron{signals: [], sum: 0}
          ],
          bias: 0.15
        },
        %Layer{
          neurons: [
            %Neuron{signals: [], sum: 0},
            %Neuron{signals: [], sum: 0},
            %Neuron{signals: [], sum: 0}
          ],
          bias: 0.15
        },
        %Layer{
          neurons: [
            %Neuron{signals: [], sum: 0},
            %Neuron{signals: [], sum: 0}
          ],
          bias: 0.15
        }
      ]
    }
  end

  def initial_with_predefined_weights do
    %Network{
      activation_fn: Math.Sigmoid,
      layers: [
        %Layer{
          neurons: [
            %Neuron{
              signals: [
                %Signal{value: 0.0, weight: 0.25},
                %Signal{value: 0.0, weight: 0.30}
              ],
              sum: 0
            },
            %Neuron{
              signals: [
                %Signal{value: 0.0, weight: 0.35},
                %Signal{value: 0.0, weight: 0.40}
              ],
              sum: 0
            }
          ],
          bias: 0.15
        },
        %Layer{
          neurons: [
            %Neuron{
              signals: [
                %Signal{value: 0.0, weight: 0.45},
                %Signal{value: 0.0, weight: 0.50}
              ],
              sum: 0
            },
            %Neuron{
              signals: [
                %Signal{value: 0.0, weight: 0.45},
                %Signal{value: 0.0, weight: 0.40}
              ],
              sum: 0
            },
            %Neuron{
              signals: [
                %Signal{value: 0.0, weight: 0.35},
                %Signal{value: 0.0, weight: 0.30}
              ],
              sum: 0
            }
          ],
          bias: 0.15
        },
        %Layer{
          neurons: [
            %Neuron{
              signals: [
                %Signal{value: 0.0, weight: 0.25},
                %Signal{value: 0.0, weight: 0.20},
                %Signal{value: 0.0, weight: 0.25}
              ],
              sum: 0
            },
            %Neuron{
              signals: [
                %Signal{value: 0.0, weight: 0.30},
                %Signal{value: 0.0, weight: 0.35},
                %Signal{value: 0.0, weight: 0.40}
              ],
              sum: 0
            }
          ],
          bias: 0.15
        }
      ]
    }
  end

  def processed do
     %Network{
       activation_fn: Math.Sigmoid,
       layers: [
         %Layer{
           bias: 0.15,
           neurons: [
             %Neuron{
               delta: nil,
               output: 0.5866175789173301,
               signals: [
                 %Signal{value: 0.2, weight: 0.25},
                 %Signal{value: 0.5, weight: 0.3}
               ],
               sum: 0.35
             },
             %Neuron{
               delta: nil,
               output: 0.6034832498647263,
               signals: [
                 %Signal{value: 0.2, weight: 0.35},
                 %Signal{value: 0.5, weight: 0.4}
               ],
               sum: 0.42000000000000004
             }
           ]
         },
         %Layer{
           bias: 0.15,
           neurons: [
             %Neuron{
               delta: nil,
               output: 0.6716637340967927,
               signals: [
                 %Signal{value: 0.5866175789173301, weight: 0.45},
                 %Signal{value: 0.6034832498647263, weight: 0.5}
               ],
               sum: 0.7157195354451618
             },
             %Neuron{
               delta: nil,
               output: 0.6582198298643718,
               signals: [
                 %Signal{value: 0.5866175789173301, weight: 0.45},
                 %Signal{value: 0.6034832498647263, weight: 0.4}
               ],
               sum: 0.6553712104586892
             },
             %Neuron{
               delta: nil,
               output: 0.6309655179893461,
               signals: [
                 %Signal{value: 0.5866175789173301, weight: 0.35},
                 %Signal{value: 0.6034832498647263, weight: 0.3}
               ],
               sum: 0.5363611275804835
             }
           ]
         },
         %Layer{
           bias: 0.15,
           neurons: [
             %Neuron{
               delta: nil,
               output: 0.6473249418260394,
               signals: [
                 %Signal{value: 0.6716637340967927, weight: 0.25},
                 %Signal{value: 0.6582198298643718, weight: 0.2},
                 %Signal{value: 0.6309655179893461, weight: 0.25}
               ],
               sum: 0.607301278994409
             },
             %Neuron{
               delta: nil,
               output: 0.6972554089636754,
               signals: [
                 %Signal{value: 0.6716637340967927, weight: 0.3},
                 %Signal{value: 0.6582198298643718, weight: 0.35},
                 %Signal{value: 0.6309655179893461, weight: 0.4}
               ],
               sum: 0.8342622678773064
             }
           ]
         }
       ]
     }
  end
end
