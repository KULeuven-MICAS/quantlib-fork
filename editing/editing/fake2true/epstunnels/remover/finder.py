import torch
import torch.fx as fx
from typing import List

from .applicationpoint import EpsTunnelNode
from quantlib.editing.editing.editors import Finder
from quantlib.editing.graphs.fx import FXOpcodeClasses
from quantlib.editing.graphs.nn import EpsTunnel


class EpsTunnelRemoverFinder(Finder):

    @staticmethod
    def is_identity_epstunnel(g: fx.GraphModule, n: fx.Node) -> bool:
        m = g.get_submodule(target=n.target)
        return torch.all(m.eps_in == m.eps_out)

    @staticmethod
    def is_integerised_placeholder(g: fx.GraphModule, n: fx.Node) -> bool:

        # TODO: copy here my handwritten notes (5.5.2022) justifying why this is a valid application point

        assert len(n.all_input_nodes) == 1
        predecessor = next(iter(n.all_input_nodes))

        m = g.get_submodule(target=n.target)

        return (predecessor.op in FXOpcodeClasses.PLACEHOLDER.value) and torch.all(m.eps_out == 1.0)
    # if output return False
    def is_output_node(g , n):
        users = [u for u in n.users]
        if len(users)==1:
            if(users[0].op in FXOpcodeClasses.OUTPUT.value) :
                return False
        return True
    def find(self, g: fx.GraphModule) -> List[EpsTunnelNode]:

        # find `EpsTunnel` `fx.Node`s
        module_nodes = filter(lambda n: (n.op in FXOpcodeClasses.CALL_MODULE.value), g.graph.nodes)
        # since we consume the `filter` generator twice in the next lines, we must ensure that it does not get empty after the first consumption
        epstunnels   = list(filter(lambda n: isinstance(g.get_submodule(target=n.target), EpsTunnel), module_nodes))

        # filter out those `fx.Node`s that do not represent the identity or integerised inputs
        identitytunnels = filter(lambda n: EpsTunnelRemoverFinder.is_identity_epstunnel(g, n), epstunnels)
        integerisedplhl = filter(lambda n: EpsTunnelRemoverFinder.is_integerised_placeholder(g, n), epstunnels)
        ls = list(set(list(identitytunnels) + list(integerisedplhl)))
        not_output_node = filter(lambda n: EpsTunnelRemoverFinder.is_output_node(g, n), ls)
        return [EpsTunnelNode(n) for n in list(not_output_node)]

    def check_aps_commutativity(self, aps: List[EpsTunnelNode]) -> bool:

        return len(aps) == len(set(map(lambda ap: ap.node, aps)))
