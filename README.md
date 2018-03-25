# PyTorch Module Probe

[PyTorch](http://pytorch.org/) is a well-known deep learning framework. A common problem using it to develop deep network is to anticipate certain aspects of the intermediate results before running it. This becomes important in the following example:

	import torch
	import torch.nn as nn
	class ConvNet(nn.Module):
	    def __init__(self):
	        self.features = nn.Sequential(
	            nn.Conv2d(3, 64, 3, padding=1),
	            nn.Conv2d(64, 256, 3),
	            # given the dimensions of the input, what's the dimension here?
	        )
	    def forward(self, x):
	        x = self.features(x)
	        return x

It often requires some calculation or experiments to get the exact dimension of the intermediate value. It soon becomes an annoying distraction when designing complicated network architecture.

This repository aims to solve this problem. The author thought up this method and made use of it in his final project of CSE253 Neural Networks and Pattern Recognition course as a Master student in University of California San Diego.

## Usage

	import torch
	import torch.nn as nn
	from probe import SizeProbe
	from inst import instrumented_sequential as instseq
	class ConvNet(nn.Module):
	    def __init__(self):
	        self.features = instseq(nn.Sequential)(
	            nn.Conv2d(3, 64, 3, padding=1),
	            SizeProbe('conv1', echo=True),
	            nn.Conv2d(64, 256, 3),
	            SizeProbe('conv2', echo=True),
	        )
	    def forward(self, x):
	        x = self.features(x)
	        return x

Then at Python console:

	>>> import torch
	>>> from torch.autograd import Variable
	>>> net = ConvNet()
	>>>
	>>> #1
	>>> outputs = net(Variable(torch.zeros(128, 3, 32, 32)))
	conv1: torch.Size([128, 64, 32, 32])
	conv2: torch.Size([128, 256, 30, 30])
	>>>
	>>> #2
	>>> net.state_dict().keys()
	['features.0.weight', 'features.0.bias', 'features.1.weight', 'features.1.bias']
	>>>
	>>> #3
	>>> net
	ConvNet(
	  (features): InstrumentedSequential(
	    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (__probe_conv1): SizeProbe(
	    )
	    (1): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1))
	    (__probe_conv2): SizeProbe(
	    )
	  )
	)

where

- `#1` shows the inspection of the sizes of the intermediate values
- `#2` shows the injected probe modules does not affect `load_state_dict` and `torch.save(net.state_dict())` functions
- `#3` shows the internal structure of the instrumented `ConvNet`, where the probe modules are injected

## Extension

The implementation of `SizeProbe` gives an example application of this probing framework. The end users are expected to write more probe modules that suit their needs.

