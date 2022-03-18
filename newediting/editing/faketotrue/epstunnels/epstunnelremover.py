import torch
import torch.fx as fx
from typing import List

from quantlib.newediting.graphs.nn.epstunnel import EpsTunnel
from ..epspropagation.epspropagator import EpsPropagator, is_eps_annotated
from ...editors.editors import ApplicationPoint, Rewriter
from ....graphs import FXOPCODE_CALL_MODULE, nnmodule_from_fxnode


class EpsTunnelRemover(Rewriter):

    def __init__(self, force: bool = True):
        name = 'EpsTunnelRemover'
        super(EpsTunnelRemover, self).__init__(name)
        self._force = force

    def find(self, g: fx.GraphModule) -> List[ApplicationPoint]:
        if self._force:
            apcores = list(filter(lambda n:
                                  (n.op in FXOPCODE_CALL_MODULE) and
                                  isinstance(nnmodule_from_fxnode(n, g), EpsTunnel), g.graph.nodes))
        else:
            apcores = list(filter(lambda n:
                                  (n.op in FXOPCODE_CALL_MODULE) and
                                  isinstance(nnmodule_from_fxnode(n, g), EpsTunnel) and
                                  torch.all(nnmodule_from_fxnode(n, g)._eps_in == nnmodule_from_fxnode(n, g)._eps_out), g.graph.nodes))

        aps = [ApplicationPoint(rewriter=self, graph=g, apcore=a) for a in apcores]
        return aps

    def _check_aps(self, g: fx.GraphModule, aps: List[ApplicationPoint]) -> None:
        pass  # TODO: verify that application points do not overlap and that they were generated by this Rewriter

    def _apply(self, g: fx.GraphModule, ap: ApplicationPoint) -> fx.GraphModule:

        n = ap.apcore

        upstream_nodes = {p for p in n.all_input_nodes}
        assert len(upstream_nodes) == 1
        un = next(iter(upstream_nodes))

        downstream_nodes = {s for s in n.users}
        assert len(downstream_nodes) == 1
        dn = next(iter(downstream_nodes))

        dn.replace_input_with(n, un)

        g.delete_submodule(n.target)
        g.graph.erase_node(n)

        return g
