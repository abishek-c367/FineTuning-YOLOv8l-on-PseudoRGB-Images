# Import PyTorch for neural network operations
import torch
# Import torch.nn for building neural networks
import torch.nn as nn
# Import torch.nn.functional for functional operations like activation functions
import torch.nn.functional as F


# Define a residual block for the generator, which helps in training deeper networks
class ResidualBlock(nn.Module):
    # Initialize the residual block with the number of features (channels)
    def __init__(self, features):
        # Call the parent class constructor
        super(ResidualBlock, self).__init__()
        # Define the sequential block with two convolutional layers and normalization
        self.block = nn.Sequential(
            # Add reflection padding to maintain spatial dimensions
            nn.ReflectionPad2d(1),
            # First convolutional layer with same input and output features
            nn.Conv2d(features, features, 3),
            # Instance normalization to normalize across each channel
            nn.InstanceNorm2d(features),
            # ReLU activation function
            nn.ReLU(inplace=True),
            # Add reflection padding again
            nn.ReflectionPad2d(1),
            # Second convolutional layer
            nn.Conv2d(features, features, 3),
            # Instance normalization
            nn.InstanceNorm2d(features)
        )

    # Define the forward pass, adding the input to the block output (residual connection)
    def forward(self, x):
        return x + self.block(x)


# Define the Generator class for CycleGAN
class Generator(nn.Module):
    # Initialize the generator with input channels, output channels, and number of residual blocks
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        # Call the parent class constructor
        super(Generator, self).__init__()

        # Initial convolution block to process input images
        model = [
            # Reflection padding for the initial 7x7 convolution
            nn.ReflectionPad2d(3),
            # 7x7 convolution to extract features, increasing channels to 64
            nn.Conv2d(input_nc, 64, 7),
            # Instance normalization
            nn.InstanceNorm2d(64),
            # ReLU activation
            nn.ReLU(inplace=True)
        ]

        # Downsampling blocks to reduce spatial dimensions and increase channels
        in_features = 64
        out_features = in_features * 2
        # Two downsampling steps
        for _ in range(2):
            model += [
                # 3x3 convolution with stride 2 for downsampling
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                # Instance normalization
                nn.InstanceNorm2d(out_features),
                # ReLU activation
                nn.ReLU(inplace=True)
            ]
            # Update feature counts
            in_features = out_features
            out_features = in_features * 2

        # Add residual blocks to learn complex transformations
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling blocks to restore spatial dimensions
        out_features = in_features // 2
        # Two upsampling steps
        for _ in range(2):
            model += [
                # Transposed convolution for upsampling
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                # Instance normalization
                nn.InstanceNorm2d(out_features),
                # ReLU activation
                nn.ReLU(inplace=True)
            ]
            # Update feature counts
            in_features = out_features
            out_features = in_features // 2

        # Output layer to produce the final image
        model += [
            # Reflection padding
            nn.ReflectionPad2d(3),
            # 7x7 convolution to output channels
            nn.Conv2d(64, output_nc, 7),
            # Tanh activation to scale outputs to [-1, 1]
            nn.Tanh()
        ]

        # Create the sequential model from the list
        self.model = nn.Sequential(*model)

    # Define the forward pass through the generator
    def forward(self, x):
        return self.model(x)


# Define the Discriminator class for CycleGAN
class Discriminator(nn.Module):
    # Initialize the discriminator with input channels
    def __init__(self, input_nc):
        # Call the parent class constructor
        super(Discriminator, self).__init__()

        # Build the discriminator model as a sequence of convolutional layers
        model = [
            # First convolution: 4x4 kernel, stride 2, padding 1
            nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
            # Leaky ReLU with negative slope 0.2
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # Add more convolutional blocks
        model += [
            # Second block
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            # Third block
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            # Fourth block
            nn.Conv2d(256, 512, 4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # Final convolution to output a single channel (patch prediction)
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        # Create the sequential model
        self.model = nn.Sequential(*model)

    # Define the forward pass, applying average pooling to get a scalar output
    def forward(self, x):
        # Pass through the model
        x = self.model(x)
        # Apply average pooling over the spatial dimensions and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)