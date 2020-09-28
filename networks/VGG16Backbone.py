import torch
import torch.nn as nn


class VGG16Backbone(nn.Module):

    """
    5 convolutional blocks of the VGG-16 network
    """
    
    def __init__(self):
        super().__init__()

        # Convolution and pooling modules
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv14 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv15 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

        # Activation functions
        self.relu = nn.ReLU() 

    def forward(self, X):
        """
        - Tensor shape in (C,H,W) format --- Input Tensor: (3,256,320)
        - Output shapes for each operation are mentioned in the comments
        """
        
        ## VGG block 1 - (64,128,160)
        pool1_out = self.pool1(
                                self.relu(self.conv2(
                                                    self.relu(self.conv1(X))
                                                    )  
                                        )
                                )    
        
        ## VGG block 2 - (128,64,80)
        pool2_out = self.pool2(
                                self.relu(self.conv4(
                                                    self.relu(self.conv3(pool1_out))
                                                    )
                                        )   
                                )   

        ## VGG block 3 - (256,32,40) 
        pool3_out = self.pool3(
                                self.relu(self.conv7(
                                                    self.relu(self.conv6(
                                                                            self.relu(self.conv5(pool2_out))
                                                                        )
                                                                )
                                                    )
                                        )
                                )   

        ## VGG clock 4 - (512,16,20)
        pool4_out = self.pool4(
                                self.relu(self.conv10(
                                                        self.relu(self.conv9(
                                                                            self.relu(self.conv8(pool3_out))
                                                                            )
                                                                )
                                                    )
                                        )
                                )   

        ## VGG block 5 - (512,8,10)
        pool5_out = self.pool5(
                                self.relu(self.conv13(
                                                        self.relu(self.conv12(
                                                                            self.relu(self.conv11(pool4_out))
                                                                            )
                                                                )
                                                    )
                                        )
                                )
        
        return pool5_out, pool4_out, pool3_out