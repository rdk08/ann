defmodule ANN.Test.Values.Network do
  alias ANN.{Network, Math, Neuron, Layer, Signal}

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

  def before_backpropagation do
    %Network{
      activation_fn: Math.Sigmoid,
      layers: [
        %Layer{
          neurons: [
            %Neuron{
              signals: [
                %Signal{value: 0, weight: 0.15},
                %Signal{value: 0, weight: 0.20}
              ],
              sum: 0
            },
            %Neuron{
              signals: [
                %Signal{value: 0, weight: 0.25},
                %Signal{value: 0, weight: 0.30}
              ],
              sum: 0
            }
          ],
          bias: 0.35
        },
        %Layer{
          neurons: [
            %Neuron{
              signals: [
                %Signal{value: 0, weight: 0.40},
                %Signal{value: 0, weight: 0.45}
              ],
              sum: 0
            },
            %Neuron{
              signals: [
                %Signal{value: 0, weight: 0.50},
                %Signal{value: 0, weight: 0.55}
              ],
              sum: 0
            }
          ],
          bias: 0.60
        }
      ]
    }
  end

  def after_backpropagation do
    %Network{
      activation_fn: Math.Sigmoid,
      layers: [
        %Layer{
          bias: 0.35,
          neurons: [
            %Neuron{
              delta: -0.03635030639314468,
              output: 0.5932699921071872,
              signals: [
                %Signal{value: 0.05, weight: 0.1497807161327628},
                %Signal{value: 0.1, weight: 0.19956143226552567}
              ],
              sum: 0.3774451790331907
            },
            %Neuron{
              delta: -0.041370322648744705,
              output: 0.596884378259767,
              signals: [
                %Signal{value: 0.05, weight: 0.24975114363236958},
                %Signal{value: 0.1, weight: 0.29950228726473915}
              ],
              sum: 0.39243778590809236
            }
          ]
        },
        %Layer{
          bias: 0.6,
          neurons: [
            %Neuron{
              delta: -0.7413650695523157,
              output: 0.7513650695523157,
              signals: [
                %Signal{value: 0.5932699921071872, weight: 0.35891647971788465},
                %Signal{value: 0.596884378259767, weight: 0.4086661860762334}
              ],
              sum: 1.0568608394812715
            },
            %Neuron{
              delta: 0.21707153467853746,
              output: 0.7729284653214625,
              signals: [
                %Signal{value: 0.5932699921071872, weight: 0.5113012702387375},
                %Signal{value: 0.596884378259767, weight: 0.5613701211079891}
              ],
              sum: 1.2384127562700828
            }
          ]
        }
      ]
    }
  end

  def after_backpropagation_multiple_datasets do
    %Network{
      activation_fn: Math.Sigmoid,
      layers: [
        %Layer{
          bias: 0.35,
          neurons: [
            %Neuron{
              delta: 0.03231787995962694,
              output: 0.6661186564381759,
              signals: [
                %Signal{value: 0.99, weight: 0.15061957434520243},
                %Signal{value: 0.99, weight: 0.20061957434520244}
              ],
              sum: 0.6977267572035007
            },
            %Neuron{
              delta: 0.036144592885120425,
              output: 0.7084996369695935,
              signals: [
                %Signal{value: 0.99, weight: 0.2504669755676112},
                %Signal{value: 0.99, weight: 0.3004669755676112}
              ],
              sum: 0.8954246116238702
            }
          ]
        },
        %Layer{
          bias: 0.6,
          neurons: [
            %Neuron{
              delta: 0.23123473629035374,
              output: 0.7587652637096463,
              signals: [
                %Signal{value: 0.6661186564381759, weight: 0.3858026729511511},
                %Signal{value: 0.7084996369695935, weight: 0.4360550774084207}
              ],
              sum: 1.1659352221990917
            },
            %Neuron{
              delta: 0.20707487787537104,
              output: 0.782925122124629,
              signals: [
                %Signal{value: 0.6661186564381759, weight: 0.4829876244006613},
                %Signal{value: 0.7084996369695935, weight: 0.5331102753134456}
              ],
              sum: 1.2994355039663712
            }
          ]
        }
      ]
    }
  end

  def after_training do
    %Network{
      activation_fn: Math.Sigmoid,
      layers: [
        %Layer{
          bias: 0.35,
          neurons: [
            %Neuron{
              delta: 0.057529979019108064,
              output: 0.5957322480179468,
              signals: [
                %Signal{value: 0.05, weight: 0.1912020091466572},
                %Signal{value: 0.1, weight: 0.2824040182933144}
              ],
              sum: 0.3878005022866643
            },
            %Neuron{
              delta: 0.0572869001446031,
              output: 0.5992549579790772,
              signals: [
                %Signal{value: 0.05, weight: 0.2897908430448617},
                %Signal{value: 0.1, weight: 0.37958168608972226}
              ],
              sum: 0.40244771076121527
            }
          ]
        },
        %Layer{
          bias: 0.6,
          neurons: [
            %Neuron{
              delta: -0.16708103133414512,
              output: 0.17708103133414513,
              signals: [
                %Signal{value: 0.5957322480179468, weight: -1.8133583581138766},
                %Signal{value: 0.5992549579790772, weight: -1.7766501168669944}
              ],
              sum: -1.544942442267966
            },
            %Neuron{
              delta: 0.11272791667004434,
              output: 0.8772720833299557,
              signals: [
                %Signal{value: 0.5957322480179468, weight: 1.1205048465622143},
                %Signal{value: 0.5992549579790772, weight: 1.1742237076265867}
              ],
              sum: 1.971180249729319
            }
          ]
        }
      ]
    }
  end
end
