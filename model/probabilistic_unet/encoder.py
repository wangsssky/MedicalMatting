import torch.nn as nn

from model.utils import init_weights


class Encoder(nn.Module):
    """
    A convolutional neural network, consisting of len(num_filters) times
    a block of no_convs_per_block convolutional layers, after each block
    a pooling operation is performed. And after each convolutional layer
    a non-linear (ReLU) activation function is applied.
    """

    def __init__(self, input_channels, num_filters, no_convs_per_block,
                 padding=True, posterior=False):
        super(Encoder, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.num_filters = num_filters
        # To accommodate for the mask that is concatenated at the channel
        # axis, we increase the input_channels.

        if posterior:
            self.input_channels += 1

        layers = []
        for i in range(len(self.num_filters)):
            """
            Determine input_dim and output_dim of conv layers in this block. 
            The first layer is input x output, All the subsequent layers are 
            output x output.
            """
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]

            if i != 0:
                layers.append(
                    nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))
                # layers.append(
                #     nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

            layers.append(nn.Conv2d(input_dim, output_dim,
                                    kernel_size=(3, 3),
                                    padding=(int(padding), int(padding))))
            layers.append(nn.ReLU())

            for _ in range(no_convs_per_block - 1):
                layers.append(nn.Conv2d(output_dim, output_dim,
                                        kernel_size=(3, 3),
                                        padding=(int(padding), int(padding))))
                layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)
        self.layers.apply(init_weights)

    def forward(self, inputs):
        output = self.layers(inputs)
        return output
