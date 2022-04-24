import torch.nn as nn

class MLPAE(nn.Module):
  """
  Similar implementations can be found in the ECE 685 lecture slides
  An MLP undercomplete autoencoder that 
  can reduce data dimension by extracting nonlinear features
  """  

  def __init__(self, in_components, out_components):
    """
    Constructor of this 

    Parameters
    ----------
    in_components : int
      Dimension of input signals.
    out_components : int
      Dimension of extracted features.

    Returns
    -------
    None.

    """

    super(MLPAE, self).__init__()

    self.in_features = in_components
    self.out_features = out_components

    self.Encoder = nn.Sequential(
      nn.Linear(in_features=self.in_features, out_features=256),
      nn.ReLU(inplace=False),
      nn.Linear(in_features=256, out_features=64),
      nn.ReLU(inplace=False),      
      nn.Linear(in_features=64, out_features=self.out_features),
      
    )

    self.Decoder = nn.Sequential(

      nn.Linear(in_features=self.out_features, out_features=64),
      nn.ReLU(inplace=False),      
      nn.Linear(in_features=64, out_features=256),
      nn.ReLU(inplace=False),      
      nn.Linear(in_features=256, out_features=self.in_features),
    )

  def forward(self, x):

    encoder = self.Encoder(x)
    decoder = self.Decoder(encoder)
    return encoder, decoder
