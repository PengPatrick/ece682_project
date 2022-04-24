import torch.nn as nn

# Similar implementations can be found in the ECE 685 lecture slides
class MLPAE(nn.Module):
  def __init__(self, in_components, out_components):

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
