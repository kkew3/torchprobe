import unittest
import sys
import os
cd = os.path.dirname(__file__)
sys.path.append(os.path.join(cd, '..'))
import probe
import torch
import torch.nn as nn
from torch.autograd import Variable
from inst import instrumented_sequential


class IdleSentinelProbe(probe.SentinelProbeModule):
    def __init__(self, key):
        probe.SentinelProbeModule.__init__(self, key)
        self.sz = None

    def do_probe(self, x):
        self.sz = x.size()
        return x

_Seqential = instrumented_sequential(nn.Sequential)

class ConvNet(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.features = _Seqential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            IdleSentinelProbe('linear1'),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            IdleSentinelProbe('linear2'),
        )
    def forward(self, x):
        z = self.features(x)
        y = torch.sum(z)
        return y

__net = ConvNet()
assert isinstance(list(__net.features.children())[2], IdleSentinelProbe)
assert list(__net.features.children())[2].key == 'linear1'
assert isinstance(list(__net.features.children())[5], IdleSentinelProbe)
assert list(__net.features.children())[5].key == 'linear2'

def get_convnet_probes(module):
    c = list(module.features.children())
    return c[2], c[5]

class DataPrep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expected_size = (128, 3, 32, 32)
        cls.expected_size1 = (128, 6, 28, 28)
        cls.expected_size2 = (128, 16, 24, 24)
        cls.data = torch.rand(*cls.expected_size)

class SentinelProbeTest(DataPrep):
    def setUp(self):
        self.sp = IdleSentinelProbe('idle')

    def test_init(self):
        self.assertFalse(self.sp.sentinel)

    def test_unauthorized_pass(self):
        with self.assertRaises(probe.UnauthorizedProbeAccessError):
            self.sp(type(self).data)

    def test_authorized_pass(self):
        self.sp.authorize_pass()
        self.assertTrue(self.sp.sentinel)
        self.sp(type(self).data)
        self.assertEqual(type(self).expected_size, self.sp.sz)

    def test_reentrant_unauthorized_pass(self):
        self.sp.authorize_pass()
        self.assertTrue(self.sp.sentinel)
        self.sp(type(self).data)
        self.assertEqual(type(self).expected_size, self.sp.sz)

        with self.assertRaises(probe.UnauthorizedProbeAccessError):
            self.sp(type(self).data)

    def test_reentrant_authorized_pass(self):
        self.sp.authorize_pass()
        self.assertTrue(self.sp.sentinel)
        self.sp(type(self).data)
        self.assertEqual(type(self).expected_size, self.sp.sz)

        self.sp.authorize_pass()
        self.assertTrue(self.sp.sentinel)
        self.sp(type(self).data)
        self.assertEqual(type(self).expected_size, self.sp.sz)

class SentinelModuleTest(DataPrep):
    def setUp(self):
        self.net = ConvNet()

    def test_init(self):
        sp1, sp2 = get_convnet_probes(self.net)
        self.assertFalse(sp1.sentinel)
        self.assertFalse(sp2.sentinel)

    def test_unauthorized_pass_all(self):
        with self.assertRaises(probe.UnauthorizedProbeAccessError):
            self.net(Variable(type(self).data))

    def test_unauthorized_pass_any1(self):
        sp1, sp2 = get_convnet_probes(self.net)
        sp1.authorize_pass()
        sp1, sp2 = get_convnet_probes(self.net)
        with self.assertRaises(probe.UnauthorizedProbeAccessError):
            self.net(Variable(type(self).data))

    def test_unauthorized_pass_any2(self):
        sp1, sp2 = get_convnet_probes(self.net)
        sp2.authorize_pass()
        with self.assertRaises(probe.UnauthorizedProbeAccessError):
            self.net(Variable(type(self).data))

    def test_authorized_pass(self):
        sp1, sp2 = get_convnet_probes(self.net)
        sp1.authorize_pass()
        sp2.authorize_pass()
        y = self.net(Variable(type(self).data))
        self.assertEqual(type(self).expected_size1, sp1.sz)
        self.assertEqual(type(self).expected_size2, sp2.sz)

    def test_reentrant_unauthorized_pass_all(self):
        sp1, sp2 = get_convnet_probes(self.net)
        sp1.authorize_pass()
        sp2.authorize_pass()
        y = self.net(Variable(type(self).data))
        self.assertEqual(type(self).expected_size1, sp1.sz)
        self.assertEqual(type(self).expected_size2, sp2.sz)

        with self.assertRaises(probe.UnauthorizedProbeAccessError):
            self.net(Variable(type(self).data))

    def test_reentrant_unauthorized_pass_any1(self):
        sp1, sp2 = get_convnet_probes(self.net)
        sp1.authorize_pass()
        sp2.authorize_pass()
        y = self.net(Variable(type(self).data))
        self.assertEqual(type(self).expected_size1, sp1.sz)
        self.assertEqual(type(self).expected_size2, sp2.sz)

        sp1.authorize_pass()
        with self.assertRaises(probe.UnauthorizedProbeAccessError):
            self.net(Variable(type(self).data))

    def test_reentrant_unauthorized_pass_any2(self):
        sp1, sp2 = get_convnet_probes(self.net)
        sp1.authorize_pass()
        sp2.authorize_pass()
        y = self.net(Variable(type(self).data))
        self.assertEqual(type(self).expected_size1, sp1.sz)
        self.assertEqual(type(self).expected_size2, sp2.sz)

        sp2.authorize_pass()
        with self.assertRaises(probe.UnauthorizedProbeAccessError):
            self.net(Variable(type(self).data))

    def test_reentrant_authorized_pass(self):
        sp1, sp2 = get_convnet_probes(self.net)
        sp1.authorize_pass()
        sp2.authorize_pass()
        y = self.net(Variable(type(self).data))
        self.assertEqual(type(self).expected_size1, sp1.sz)
        self.assertEqual(type(self).expected_size2, sp2.sz)

        sp1.authorize_pass()
        sp2.authorize_pass()
        y = self.net(Variable(type(self).data))
        self.assertEqual(type(self).expected_size1, sp1.sz)
        self.assertEqual(type(self).expected_size2, sp2.sz)

    def test_authorize_all(self):
        sp1, sp2 = get_convnet_probes(self.net)
        probe.authorize_all(self.net)
        self.assertTrue(sp1.sentinel)
        self.assertTrue(sp2.sentinel)

        y = self.net(Variable(type(self).data))
        self.assertEqual(type(self).expected_size1, sp1.sz)
        self.assertEqual(type(self).expected_size2, sp2.sz)
