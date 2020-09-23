import torch
import torch.nn as nn


class FCN_VGG16(nn.Module):
	"""
	Simplified version of FCN-VGG16 with less parameters
	"""
	def __init__(self, mode='FCN-32s'):
		'''
		mode options: 'fcn-32s', '3-stream'
		'''
		
		super().__init__()

		self.mode = mode

		# Convolution and pooling modules
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
		self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
		self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.conv5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
		self.conv6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
		self.conv7 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
		self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.conv8 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
		self.conv9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
		self.conv10 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
		self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.conv11 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
		self.conv12 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
		self.conv13 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
		self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.conv14 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
		self.conv15 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)


		# Activation functions
		self.relu = nn.ReLU() 

		# Upconvolution module for FCN-32s
		self.upconv_32s = nn.ConvTranspose2d(in_channels=64, out_channels=21, kernel_size=3, stride=32, padding=0, output_padding=29)


		if self.mode == '3-stream':
			# Modules for FCN-16s
			self.upconv15_2x = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
			self.scoring_conv_16s = nn.Conv2d(in_channels=128, out_channels=21, kernel_size=1, stride=1, padding=0)
			self.upconv_16s = nn.ConvTranspose2d(in_channels=21, out_channels=21, kernel_size=3, stride=16, padding=0, output_padding=13)


			# Modules for FCN-8s
			self.upconv15_4x = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=4, padding=0, output_padding=1)
			self.scoring_conv_8s = nn.Conv2d(in_channels=64, out_channels=21, kernel_size=1, stride=1, padding=0)
			self.upconv_8s = nn.ConvTranspose2d(in_channels=21, out_channels=21, kernel_size=3, stride=8, padding=0, output_padding=5)

					

	def forward(self, X):
		'''
		- Tensor shape in (C,H,W) format --- Input Tensor: (3,256,320)
		- Output shapes for each operation are mentioned in the comments
		- Final outputs are as logits. Need to be converted to probabilites explicitly using softmax during training/inference
		'''

		# Simplified VGG Net section --
		
		## VGG block 1 - (8,128,160)
		pool1_out = self.pool1(
							   self.relu(self.conv2(
											        self.relu(self.conv1(X))
										           )  
										)
							  )    
		
		## VGG block 2 - (16,64,80)
		pool2_out = self.pool2(
							   self.relu(self.conv4(
											        self.relu(self.conv3(pool1_out))
								                   )
								        )   
		                      )   

		## VGG block 3 - (32,32,40) 
		pool3_out = self.pool3(
			                   self.relu(self.conv7(
								                    self.relu(self.conv6(
											                             self.relu(self.conv5(pool2_out))
													                    )
													         )
										           )
							            )
							  )   

		## VGG clock 4 - (64,16,20)
		pool4_out = self.pool4(
			                   self.relu(self.conv10(
								                     self.relu(self.conv9(
											                              self.relu(self.conv8(pool3_out))
													                     )
															  )
													)
										)
							  )   

		## VGG block 5 - (64,8,10)
		pool5_out = self.pool5(
							   self.relu(self.conv13(
								                     self.relu(self.conv12(
														                   self.relu(self.conv11(pool4_out))
																		  )
															  )
													)
										)
							  )


		# Extra conv layers --
		conv15_out = self.conv15(self.conv14(pool5_out))  # (64,8,10)


		# FCN-32s --
		out_32s = self.upconv_32s(conv15_out) # 32x upsampling of Conv15 output to image size - (21,256,320)  

		if self.mode == 'fcn-32s':
			return out_32s


		if self.mode == '3-stream':
			# FCN-16s --
			conv15_2x_out = self.upconv15_2x(conv15_out)   # 2x upsampling of Conv15 output - (64,16,20)
			out_16s = torch.cat([pool4_out, conv15_2x_out], dim=1) # Concatenate with Pool4 output - (128,16,20) 
			out_16s = self.scoring_conv_16s(out_16s)  # "Scoring" using 1x1 convolution - (21,16,20)
			out_16s = self.upconv_16s(out_16s)   # Upsampling to image size - (21,256,320)    
			

			# FCN-8s --
			conv15_4x_out = self.upconv15_4x(conv15_out)   # 4x upsampling of Conv15 output - (32,32,40)
			out_8s = torch.cat([pool3_out, conv15_4x_out], dim=1) # Concatenate with Pool3 output - (64,32,40)
			out_8s = self.scoring_conv_8s(out_8s) # "Scoring" using 1x1 convolution - (21,32,40)
			out_8s = self.upconv_8s(out_8s)  # Upsampling to image size - (21,256,320)


			return out_32s, out_16s, out_8s
			


##################################################

if __name__ == '__main__':
	model = FCN_VGG16(mode='3-stream')
	softmax_fn = nn.Softmax(dim=1)

	X = torch.rand((1,3,256,320))  # Random pixel image
	out_32s, out_16s, out_8s = model.forward(X)

	out_32s = softmax_fn(out_32s)
	out_16s = softmax_fn(out_16s)
	out_8s = softmax_fn(out_8s)

	print(out_32s.shape)
	print(out_16s.shape)
	print(out_8s.shape)