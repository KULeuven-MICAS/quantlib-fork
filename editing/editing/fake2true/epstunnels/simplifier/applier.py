import torch
import torch.fx as fx

from .applicationpoint import CandidateEpsTunnelConstruct
from quantlib.editing.editing.editors import Applier


class EpsTunnelConstructApplier(Applier):

    def _apply(self, g: fx.GraphModule, ap: CandidateEpsTunnelConstruct, id_: str) -> fx.GraphModule:

        # integerise the arrays entering the construct
        for node in ap.backward: 
            predecessor = [p for p in node.all_input_nodes]
            users = [u for u in node.users] 
            print(f'For {node} with construct', " and users : " ,*users ) 
        inbound_eps_tunnels = tuple(map(lambda n: g.get_submodule(target=n.target), ap.backward))
        for m in inbound_eps_tunnels:
            m.set_eps_out(torch.ones_like(m.eps_out))

        # do not integerise the arrays exiting the construct (they are already integerised!)
        for node in ap.backward: 
            predecessor = [p for p in node.all_input_nodes]
            users = [u for u in node.users] 
            print(f'For forward {node} has predecessors ', *predecessor  ) 
        outbound_eps_tunnels = tuple(map(lambda n: g.get_submodule(target=n.target), ap.forward))
        for m in outbound_eps_tunnels:
            m.set_eps_in(torch.ones_like(m.eps_in))

        return g
