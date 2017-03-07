defmodule ANNEx.Network.Config do
  @enforce_keys [:layers, :activation_fn]
  defstruct layers: [],
            activation_fn: nil
end
