import torch
import torch.nn as nn


class Rs_GCN(nn.Module):
    """
    Relational Spatial Graph Convolutional Network (Rs-GCN) for feature enhancement.

    Args:
        in_channels (int): Number of input channels (feature dimension D).
        inter_channels (int or None): Number of intermediate channels. If None, set to in_channels // 2.
        bn_layer (bool, optional): Whether to use batch normalization (default: True).
    """

    def __init__(self, in_channels, inter_channels, bn_layer=True):
        super(Rs_GCN, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels if inter_channels is not None else max(in_channels // 2, 1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Define convolutional layers
        conv_nd = nn.Conv1d
        bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        # Output layer with optional batch normalization
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)  # BN weight
            nn.init.constant_(self.W[1].bias, 0)    # BN bias
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        # Initialize convolutional weights
        nn.init.xavier_normal_(self.g.weight)
        nn.init.xavier_normal_(self.theta.weight)
        nn.init.xavier_normal_(self.phi.weight)
        if not bn_layer:
            nn.init.xavier_normal_(self.W.weight)

    def forward(self, v):
        """
        Forward pass of the Rs-GCN.

        Args:
            v (torch.Tensor): Input tensor, shape [batch_size, in_channels, num_nodes] (B, D, N).

        Returns:
            torch.Tensor: Enhanced feature tensor, shape [batch_size, in_channels, num_nodes].
        """
        v = v.to(self.device)
        batch_size = v.size(0)

        # Graph convolution components
        g_v = self.g(v).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)  # [B, N, inter_channels]
        theta_v = self.theta(v).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)  # [B, N, inter_channels]
        phi_v = self.phi(v).view(batch_size, self.inter_channels, -1)  # [B, inter_channels, N]

        # Compute relation matrix and normalize
        R = torch.matmul(theta_v, phi_v)  # [B, N, N]
        N = R.size(-1)
        R_div_C = R / N  # Normalize by number of nodes

        # Apply relation to features
        y = torch.matmul(R_div_C, g_v)  # [B, N, inter_channels]
        y = y.permute(0, 2, 1).contiguous().view(batch_size, self.inter_channels, *v.size()[2:])  # [B, inter_channels, N]
        W_y = self.W(y)  # [B, in_channels, N]

        # Residual connection
        v_star = W_y + v
        return v_star


if __name__ == '__main__':
    # Simple test
    model = Rs_GCN(in_channels=2048, inter_channels=512)
    v = torch.randn(32, 2048, 7)  # batch_size=32, in_channels=2048, num_nodes=7
    output = model(v)
    print(f"Input shape: {v.shape}")
    print(f"Output shape: {output.shape}")  # Expected: [32, 2048, 7]