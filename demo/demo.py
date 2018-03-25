import torch
import torch.nn as nn
from torch.autograd import Variable
import probe
from probe import SizeProbe
from inst import instrumented_sequential as instseq
# for demo we need `tqdm`: https://pypi.python.org/pypi/tqdm
# a command line progress bar utility
from tqdm import tqdm

sizes = dict()
norms = dict()


# Demo extending `probe.ProbeModule` -- calculating the norm of
# the intermediate values
class NormProbe(probe.ProbeModule):
    def __init__(self, key, out):
        probe.ProbeModule.__init__(self, key)
        self.out = out
    def do_probe(self, x):
        self.out[self.key] = torch.norm(x.data)


class ConvNet(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        # note the `instseq` here
        self.features = instseq(nn.Sequential)(
            nn.Conv2d(3, 64, 3),
            SizeProbe('conv1', echo=tqdm.write, out=sizes),
            NormProbe('conv1-norm', out=norms),
            nn.Conv2d(64, 256, 3),
        )
    def forward(self, x):
        return self.features(x)


if __name__ == '__main__':
    net = ConvNet()
    size_list = list()
    norm_list = list()
    dataset = [torch.randn(8, 3, 32, 32) for _ in range(10)]
    for inputs in tqdm(dataset, ascii=True):
        inputs = Variable(inputs)
        outputs = net(inputs)
        size_list.append(sizes.copy())
        norm_list.append(norms.copy())
    print '---------------'
    print 'Probing results'
    print '---------------'
    print '\n'.join(map(str, size_list))
    print '\n'.join(map(str, norm_list))
