from __future__ import print_function
import torch.nn as nn


class ProbeModule(nn.Module):
    """
    The super class of the probe modules
    """
    def __init__(self, key):
        """
        :param key: the unique name of the probe
        """
        nn.Module.__init__(self)
        self.key = key

    def forward(self, x):
        self.do_proeb(x)
        return x
    
    def do_probe(self, x):
        raise NotImplementedError()


class SizeProbe(ProbeModule):
    """
    Inspect the size of upstream data.
    """
    def __init__(self, key, echo=True, out=None):
        """
        :param key: the unique name of the probe
        :param echo: False or None to suppress printing to stdout; True or 
               a callable object (the print function to invoke instead of
               the native `print`) to print the size message
        :param out: the container to hold the size info, expecting a dict
               or similar type, where the key would be `key` and value
               the size info, i.e. a list of tuples
        """
        ProbeModule.__init__(self, key)
        self.echo = print if echo is True else echo
        self.out = out

    def do_probe(self, x):
        size_info = x.shape if hasattr(x, 'shape') else None
        if self.echo:
            self.echo('{}: {}'.format(self.key, size_info))
        if self.out is not None:
            self.out[self.key] = size_info
