class NeRFModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeRFModel, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, encoded):
        features = self.backbone(encoded)
        return features

def main():
    model = NeRFModel(input_dim=60, output_dim=63)
    sample_points = torch.randn(65536, 60)  # Adjust input shape as needed
    densities, colors = model(sample_points)

if __name__ == "__main__":
    main()