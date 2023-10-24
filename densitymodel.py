from typing import List, Dict
import math
import ase

import mindspore as ms
import mindspore.nn as nn

import layer
from layer import ShiftedSoftplus


class DensityModel(nn.Cell):
    def __init__(
        self,
        num_interactions,
        hidden_state_size,
        cutoff,
        gaussian_expansion_step=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.atom_model = AtomRepresentationModel(
            num_interactions,
            hidden_state_size,
            cutoff,
            gaussian_expansion_step,
        )

        self.probe_model = ProbeMessageModel(
            num_interactions,
            hidden_state_size,
            cutoff,
            gaussian_expansion_step,
        )

    def construct(self, input_dict):
        atom_representation = self.atom_model(input_dict)
        probe_result = self.probe_model(input_dict, atom_representation)
        return probe_result

class PainnDensityModel(nn.Cell):
    def __init__(
        self,
        num_interactions,
        hidden_state_size,
        cutoff,
        distance_embedding_size=30,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.atom_model = PainnAtomRepresentationModel(
            num_interactions,
            hidden_state_size,
            cutoff,
            distance_embedding_size,
        )

        self.probe_model = PainnProbeMessageModel(
            num_interactions,
            hidden_state_size,
            cutoff,
            distance_embedding_size,
        )

    def construct(self, input_dict):
        atom_representation_scalar, atom_representation_vector = self.atom_model(input_dict)
        probe_result = self.probe_model(input_dict, atom_representation_scalar, atom_representation_vector)
        return probe_result


class ProbeMessageModel(nn.Cell):
    def __init__(
        self,
        num_interactions,
        hidden_state_size,
        cutoff,
        gaussian_expansion_step,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.num_interactions = num_interactions
        self.hidden_state_size = hidden_state_size
        self.cutoff = cutoff
        self.gaussian_expansion_step = gaussian_expansion_step

        edge_size = int(math.ceil(self.cutoff / self.gaussian_expansion_step))

        # Setup interaction networks
        self.messagesum_layers = nn.CellList(
            [
                layer.MessageSum(
                    hidden_state_size, edge_size, self.cutoff, include_receiver=True
                )
                for _ in range(num_interactions)
            ]
        )

        # Setup transitions networks
        self.probe_state_gate_functions = nn.CellList(
            [
                nn.SequentialCell(
                    nn.Dense(hidden_state_size, hidden_state_size),
                    ShiftedSoftplus(),
                    nn.Dense(hidden_state_size, hidden_state_size),
                    nn.Sigmoid(),
                )
                for _ in range(num_interactions)
            ]
        )
        self.probe_state_transition_functions = nn.CellList(
            [
                nn.SequentialCell(
                    nn.Dense(hidden_state_size, hidden_state_size),
                    ShiftedSoftplus(),
                    nn.Dense(hidden_state_size, hidden_state_size),
                )
                for _ in range(num_interactions)
            ]
        )

        # Setup readout function
        self.readout_function = nn.SequentialCell(
            nn.Dense(hidden_state_size, hidden_state_size),
            ShiftedSoftplus(),
            nn.Dense(hidden_state_size, 1),
        )

    def construct(
            self,
            input_dict: Dict[str, ms.Tensor],
            atom_representation: List[ms.Tensor],
            compute_iri=False,
            compute_dori=False,
            compute_hessian=False,
    ):
        # if compute_iri or compute_dori or compute_hessian:
        # input_dict["probe_xyz"].requires_grad_()

        # Unpad and concatenate edges and features into batch (0th) dimension
        atom_xyz = layer.unpad_and_cat(input_dict["atom_xyz"], input_dict["num_nodes"])
        probe_xyz = layer.unpad_and_cat(input_dict["probe_xyz"], input_dict["num_probes"])
        edge_offset = ms.ops.cumsum(ms.ops.cat((ms.tensor([0]), input_dict["num_nodes"][:-1],)), axis=0)
        edge_offset = edge_offset[:, None, None]

        # Unpad and concatenate probe edges into batch (0th) dimension
        probe_edges_displacement = layer.unpad_and_cat(input_dict["probe_edges_displacement"],
                                                       input_dict["num_probe_edges"])
        edge_probe_offset = ms.ops.cumsum(ms.ops.cat((ms.tensor([0]), input_dict["num_probes"][:-1],)), axis=0, )
        edge_probe_offset = edge_probe_offset[:, None, None]
        edge_probe_offset = ms.ops.cat((edge_offset, edge_probe_offset), axis=2)
        probe_edges = input_dict["probe_edges"] + edge_probe_offset
        probe_edges = layer.unpad_and_cat(probe_edges, input_dict["num_probe_edges"])

        # Compute edge distances
        probe_edges_features = layer.calc_distance_to_probe(
            atom_xyz,
            probe_xyz,
            input_dict["cell"],
            probe_edges,
            probe_edges_displacement,
            input_dict["num_probe_edges"],
        )

        # Expand edge features in Gaussian basis
        probe_edge_state = layer.gaussian_expansion(
            probe_edges_features, [(0.0, self.gaussian_expansion_step, self.cutoff)]
        )

        probe_state = ms.ops.zeros(
            (ms.ops.sum(input_dict["num_probes"]), self.hidden_state_size)
        )
        for msg_layer, gate_layer, state_layer, nodes in zip(
                self.messagesum_layers,
                self.probe_state_gate_functions,
                self.probe_state_transition_functions,
                atom_representation,
        ):
            msgsum = msg_layer(
                nodes,
                probe_edges,
                probe_edge_state,
                probe_edges_features,
                probe_state,
            )
            gates = gate_layer(probe_state)
            probe_state = probe_state * gates + (1 - gates) * state_layer(msgsum)

        # Restack probe states
        probe_output = self.readout_function(probe_state).squeeze(1)
        probe_output = layer.pad_and_stack(
            ms.ops.split(
                probe_output,
                list(input_dict["num_probes"].detach().cpu().numpy()),
                axis=0,
            )
            # torch.split(probe_output, input_dict["num_probes"], dim=0)
            # probe_output.reshape((-1, input_dict["num_probes"][0]))
        )

        if compute_iri or compute_dori or compute_hessian:

            def dp_dxyz_graph(x):
                input_dict_ = input_dict
                atom_xyz = layer.unpad_and_cat(input_dict_["atom_xyz"], input_dict_["num_nodes"])
                probe_xyz = layer.unpad_and_cat(x, input_dict_["num_probes"])
                edge_offset = ms.ops.cumsum(ms.ops.cat((ms.tensor([0]), input_dict_["num_nodes"][:-1],)), axis=0)
                edge_offset = edge_offset[:, None, None]

                # Unpad and concatenate probe edges into batch (0th) dimension
                probe_edges_displacement = layer.unpad_and_cat(input_dict_["probe_edges_displacement"],
                                                               input_dict_["num_probe_edges"])
                edge_probe_offset = ms.ops.cumsum(ms.ops.cat((ms.tensor([0]), input_dict_["num_probes"][:-1],)),
                                                  axis=0, )
                edge_probe_offset = edge_probe_offset[:, None, None]
                edge_probe_offset = ms.ops.cat((edge_offset, edge_probe_offset), axis=2)
                probe_edges = input_dict_["probe_edges"] + edge_probe_offset
                probe_edges = layer.unpad_and_cat(probe_edges, input_dict_["num_probe_edges"])

                # Compute edge distances
                probe_edges_features = layer.calc_distance_to_probe(
                    atom_xyz,
                    probe_xyz,
                    input_dict_["cell"],
                    probe_edges,
                    probe_edges_displacement,
                    input_dict_["num_probe_edges"],
                )

                # Expand edge features in Gaussian basis
                probe_edge_state = layer.gaussian_expansion(
                    probe_edges_features, [(0.0, self.gaussian_expansion_step, self.cutoff)]
                )

                probe_state = ms.ops.zeros(
                    (ms.ops.sum(input_dict_["num_probes"]), self.hidden_state_size)
                )
                for msg_layer, gate_layer, state_layer, nodes in zip(
                        self.messagesum_layers,
                        self.probe_state_gate_functions,
                        self.probe_state_transition_functions,
                        atom_representation,
                ):
                    msgsum = msg_layer(
                        nodes,
                        probe_edges,
                        probe_edge_state,
                        probe_edges_features,
                        probe_state,
                    )
                    gates = gate_layer(probe_state)
                    probe_state = probe_state * gates + (1 - gates) * state_layer(msgsum)

                # Restack probe states
                probe_output = self.readout_function(probe_state).squeeze(1)
                probe_output = layer.pad_and_stack(
                    ms.ops.split(
                        probe_output,
                        list(input_dict_["num_probes"].asnumpy()),
                        axis=0,
                    )
                )
                return probe_output

            dp_dxyz = ms.grad(dp_dxyz_graph, grad_position=0)(input_dict["probe_xyz"])[0]


        grad_probe_outputs = {}

        if compute_iri:
            iri = ms.ops.norm(dp_dxyz, dim=2)/(ms.ops.pow(probe_output, 1.1))
            grad_probe_outputs["iri"] = iri

        if compute_dori:
            ##
            ## DORI(r) = phi(r) / (1 + phi(r))
            ## phi(r) = ||grad(||grad(rho(r))/rho||^2)||^2 / ||grad(rho(r))/rho(r)||^6
            ##
            norm_grad_2 = ms.ops.norm(dp_dxyz/ms.ops.unsqueeze(probe_output, 2), dim=2)**2

            def grad_norm_grad_2_graph(x):
                input_dict_ = input_dict
                atom_xyz = layer.unpad_and_cat(input_dict_["atom_xyz"], input_dict_["num_nodes"])
                probe_xyz = layer.unpad_and_cat(x, input_dict_["num_probes"])
                edge_offset = ms.ops.cumsum(ms.ops.cat((ms.tensor([0]), input_dict_["num_nodes"][:-1],)), axis=0)
                edge_offset = edge_offset[:, None, None]

                # Unpad and concatenate probe edges into batch (0th) dimension
                probe_edges_displacement = layer.unpad_and_cat(input_dict_["probe_edges_displacement"],
                                                               input_dict_["num_probe_edges"])
                edge_probe_offset = ms.ops.cumsum(ms.ops.cat((ms.tensor([0]), input_dict_["num_probes"][:-1],)),
                                                  axis=0, )
                edge_probe_offset = edge_probe_offset[:, None, None]
                edge_probe_offset = ms.ops.cat((edge_offset, edge_probe_offset), axis=2)
                probe_edges = input_dict_["probe_edges"] + edge_probe_offset
                probe_edges = layer.unpad_and_cat(probe_edges, input_dict_["num_probe_edges"])

                # Compute edge distances
                probe_edges_features = layer.calc_distance_to_probe(
                    atom_xyz,
                    probe_xyz,
                    input_dict_["cell"],
                    probe_edges,
                    probe_edges_displacement,
                    input_dict_["num_probe_edges"],
                )

                # Expand edge features in Gaussian basis
                probe_edge_state = layer.gaussian_expansion(
                    probe_edges_features, [(0.0, self.gaussian_expansion_step, self.cutoff)]
                )

                probe_state = ms.ops.zeros(
                    (ms.ops.sum(input_dict_["num_probes"]), self.hidden_state_size)
                )
                for msg_layer, gate_layer, state_layer, nodes in zip(
                        self.messagesum_layers,
                        self.probe_state_gate_functions,
                        self.probe_state_transition_functions,
                        atom_representation,
                ):
                    msgsum = msg_layer(
                        nodes,
                        probe_edges,
                        probe_edge_state,
                        probe_edges_features,
                        probe_state,
                    )
                    gates = gate_layer(probe_state)
                    probe_state = probe_state * gates + (1 - gates) * state_layer(msgsum)

                # Restack probe states
                probe_output = self.readout_function(probe_state).squeeze(1)
                probe_output = layer.pad_and_stack(
                    ms.ops.split(
                        probe_output,
                        list(input_dict_["num_probes"].asnumpy()),
                        axis=0,
                    )
                )
                norm_grad_2 = ms.ops.norm(dp_dxyz / ms.ops.unsqueeze(probe_output, 2), dim=2) ** 2
                return norm_grad_2

            grad_norm_grad_2 = ms.grad(grad_norm_grad_2_graph, grad_position=0)(input_dict["probe_xyz"])[0]

            phi_r = ms.ops.norm(grad_norm_grad_2, dim=2)**2 / (norm_grad_2**3)

            dori = phi_r / (1 + phi_r)
            grad_probe_outputs["dori"] = dori

        if compute_hessian:
            hessian_shape = (input_dict["probe_xyz"].shape[0], input_dict["probe_xyz"].shape[1], 3, 3)
            hessian = ms.ops.zeros(hessian_shape, dtype=probe_xyz.dtype)
            for dim_idx, grad_out in enumerate(ms.ops.unbind(dp_dxyz, dim=-1)):

                def dp2_dxyz2_graph(x):
                    input_dict_ = input_dict
                    atom_xyz = layer.unpad_and_cat(input_dict_["atom_xyz"], input_dict_["num_nodes"])
                    probe_xyz = layer.unpad_and_cat(x, input_dict_["num_probes"])
                    edge_offset = ms.ops.cumsum(ms.ops.cat((ms.tensor([0]), input_dict_["num_nodes"][:-1],)), axis=0)
                    edge_offset = edge_offset[:, None, None]

                    # Unpad and concatenate probe edges into batch (0th) dimension
                    probe_edges_displacement = layer.unpad_and_cat(input_dict_["probe_edges_displacement"],
                                                                   input_dict_["num_probe_edges"])
                    edge_probe_offset = ms.ops.cumsum(ms.ops.cat((ms.tensor([0]), input_dict_["num_probes"][:-1],)),
                                                      axis=0, )
                    edge_probe_offset = edge_probe_offset[:, None, None]
                    edge_probe_offset = ms.ops.cat((edge_offset, edge_probe_offset), axis=2)
                    probe_edges = input_dict_["probe_edges"] + edge_probe_offset
                    probe_edges = layer.unpad_and_cat(probe_edges, input_dict_["num_probe_edges"])

                    # Compute edge distances
                    probe_edges_features = layer.calc_distance_to_probe(
                        atom_xyz,
                        probe_xyz,
                        input_dict_["cell"],
                        probe_edges,
                        probe_edges_displacement,
                        input_dict_["num_probe_edges"],
                    )

                    # Expand edge features in Gaussian basis
                    probe_edge_state = layer.gaussian_expansion(
                        probe_edges_features, [(0.0, self.gaussian_expansion_step, self.cutoff)]
                    )

                    probe_state = ms.ops.zeros(
                        (ms.ops.sum(input_dict_["num_probes"]), self.hidden_state_size)
                    )
                    for msg_layer, gate_layer, state_layer, nodes in zip(
                            self.messagesum_layers,
                            self.probe_state_gate_functions,
                            self.probe_state_transition_functions,
                            atom_representation,
                    ):
                        msgsum = msg_layer(
                            nodes,
                            probe_edges,
                            probe_edge_state,
                            probe_edges_features,
                            probe_state,
                        )
                        gates = gate_layer(probe_state)
                        probe_state = probe_state * gates + (1 - gates) * state_layer(msgsum)

                    # Restack probe states
                    probe_output = self.readout_function(probe_state).squeeze(1)
                    probe_output = layer.pad_and_stack(
                        ms.ops.split(
                            probe_output,
                            list(input_dict_["num_probes"].asnumpy()),
                            axis=0,
                        )
                    )
                    dp_dxyz = ms.grad(dp_dxyz_graph, grad_position=0)(input_dict["probe_xyz"])[0]
                    grad_out = ms.ops.unbind(dp_dxyz, dim=-1)[dim_idx]
                    return grad_out

                dp2_dxyz2 = ms.grad(dp2_dxyz2_graph, grad_position=0)(input_dict["probe_xyz"])[0]
                hessian[:, :, dim_idx] = dp2_dxyz2
            grad_probe_outputs["hessian"] = hessian


        if grad_probe_outputs:
            return probe_output, grad_probe_outputs
        else:
            return probe_output


class AtomRepresentationModel(nn.Cell):
    def __init__(
        self,
        num_interactions,
        hidden_state_size,
        cutoff,
        gaussian_expansion_step,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.num_interactions = num_interactions
        self.hidden_state_size = hidden_state_size
        self.cutoff = cutoff
        self.gaussian_expansion_step = gaussian_expansion_step

        edge_size = int(math.ceil(self.cutoff / self.gaussian_expansion_step))

        # Setup interaction networks
        self.interactions = nn.CellList(
            [
                layer.Interaction(
                    hidden_state_size, edge_size, self.cutoff, include_receiver=True
                )
                for _ in range(num_interactions)
            ]
        )

        # Atom embeddings
        self.atom_embeddings = nn.Embedding(
            len(ase.data.atomic_numbers), self.hidden_state_size
        )

    def construct(self, input_dict):
        # Unpad and concatenate edges and features into batch (0th) dimension
        edges_displacement = layer.unpad_and_cat(
            input_dict["atom_edges_displacement"], input_dict["num_atom_edges"]
        )
        edge_offset = ms.ops.cumsum(
            ms.ops.cat(
                (
                    ms.tensor([0]),
                    input_dict["num_nodes"][:-1],
                )
            ),
            axis=0,
        )
        edge_offset = edge_offset[:, None, None]
        edges = input_dict["atom_edges"] + edge_offset
        edges = layer.unpad_and_cat(edges, input_dict["num_atom_edges"])

        # Unpad and concatenate all nodes into batch (0th) dimension
        atom_xyz = layer.unpad_and_cat(input_dict["atom_xyz"], input_dict["num_nodes"])
        nodes = layer.unpad_and_cat(input_dict["nodes"], input_dict["num_nodes"])
        nodes = self.atom_embeddings(nodes)

        # Compute edge distances
        edges_features = layer.calc_distance(
            atom_xyz,
            input_dict["cell"],
            edges,
            edges_displacement,
            input_dict["num_atom_edges"],
        )

        # Expand edge features in Gaussian basis
        edge_state = layer.gaussian_expansion(
            edges_features, [(0.0, self.gaussian_expansion_step, self.cutoff)]
        )

        nodes_list = []
        # Apply interaction layers
        for int_layer in self.interactions:
            nodes = int_layer(nodes, edges, edge_state, edges_features)
            nodes_list.append(nodes)

        return nodes_list


class PainnAtomRepresentationModel(nn.Cell):
    def __init__(
        self,
        num_interactions,
        hidden_state_size,
        cutoff,
        distance_embedding_size,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.num_interactions = num_interactions
        self.hidden_state_size = hidden_state_size
        self.cutoff = cutoff
        self.distance_embedding_size = distance_embedding_size

        # Setup interaction networks
        self.interactions = nn.CellList(
            [
                layer.PaiNNInteraction(
                    hidden_state_size, self.distance_embedding_size, self.cutoff
                )
                for _ in range(num_interactions)
            ]
        )
        self.scalar_vector_update = nn.CellList(
            [layer.PaiNNUpdate(hidden_state_size) for _ in range(num_interactions)]
        )

        # Atom embeddings
        self.atom_embeddings = nn.Embedding(
            len(ase.data.atomic_numbers), self.hidden_state_size
        )

    def construct(self, input_dict):
        # Unpad and concatenate edges and features into batch (0th) dimension
        edges_displacement = layer.unpad_and_cat(
            input_dict["atom_edges_displacement"], input_dict["num_atom_edges"]
        )
        edge_offset = ms.ops.cumsum(
            ms.ops.cat(
                (
                    ms.tensor([0]),
                    input_dict["num_nodes"][:-1],
                )
            ),
            axis=0,
        )
        edge_offset = edge_offset[:, None, None]
        edges = input_dict["atom_edges"] + edge_offset
        edges = layer.unpad_and_cat(edges, input_dict["num_atom_edges"])

        # Unpad and concatenate all nodes into batch (0th) dimension
        atom_xyz = layer.unpad_and_cat(input_dict["atom_xyz"], input_dict["num_nodes"])
        nodes_scalar = layer.unpad_and_cat(input_dict["nodes"], input_dict["num_nodes"])
        nodes_scalar = self.atom_embeddings(nodes_scalar)
        nodes_vector = ms.ops.zeros(
            (nodes_scalar.shape[0], 3, self.hidden_state_size),
            dtype=nodes_scalar.dtype
        )

        # Compute edge distances
        edges_distance, edges_diff = layer.calc_distance(
            atom_xyz,
            input_dict["cell"],
            edges,
            edges_displacement,
            input_dict["num_atom_edges"],
            return_diff=True,
        )

        # Expand edge features in sinc basis
        edge_state = layer.sinc_expansion(
            edges_distance, [(self.distance_embedding_size, self.cutoff)]
        )

        nodes_list_scalar = []
        nodes_list_vector = []
        # Apply interaction layers
        for int_layer, update_layer in zip(
            self.interactions, self.scalar_vector_update
        ):
            nodes_scalar, nodes_vector = int_layer(
                nodes_scalar,
                nodes_vector,
                edge_state,
                edges_diff,
                edges_distance,
                edges,
            )
            nodes_scalar, nodes_vector = update_layer(nodes_scalar, nodes_vector)
            nodes_list_scalar.append(nodes_scalar)
            nodes_list_vector.append(nodes_vector)

        return nodes_list_scalar, nodes_list_vector


class PainnProbeMessageModel(nn.Cell):
    def __init__(
        self,
        num_interactions,
        hidden_state_size,
        cutoff,
        distance_embedding_size,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.num_interactions = num_interactions
        self.hidden_state_size = hidden_state_size
        self.cutoff = cutoff
        self.distance_embedding_size = distance_embedding_size

        # Setup interaction networks
        self.message_layers = nn.CellList(
            [
                layer.PaiNNInteractionOneWay(
                    hidden_state_size, self.distance_embedding_size, self.cutoff
                )
                for _ in range(num_interactions)
            ]
        )
        self.scalar_vector_update = nn.CellList(
            [layer.PaiNNUpdate(hidden_state_size) for _ in range(num_interactions)]
        )

        # Setup readout function
        self.readout_function = nn.SequentialCell(
            nn.Dense(hidden_state_size, hidden_state_size),
            nn.SiLU(),
            nn.Dense(hidden_state_size, 1),
        )

    def construct(
        self,
        input_dict: Dict[str, ms.Tensor],
        atom_representation_scalar: List[ms.Tensor],
        atom_representation_vector: List[ms.Tensor],
        compute_iri=False,
        compute_dori=False,
        compute_hessian=False,
    ):


        # Unpad and concatenate edges and features into batch (0th) dimension
        atom_xyz = layer.unpad_and_cat(input_dict["atom_xyz"], input_dict["num_nodes"])
        probe_xyz = layer.unpad_and_cat(
            input_dict["probe_xyz"], input_dict["num_probes"]
        )
        edge_offset = ms.ops.cumsum(
            ms.ops.cat(
                (
                    ms.tensor([0]),
                    input_dict["num_nodes"][:-1],
                )
            ),
            axis=0,
        )
        edge_offset = edge_offset[:, None, None]

        # Unpad and concatenate probe edges into batch (0th) dimension
        probe_edges_displacement = layer.unpad_and_cat(
            input_dict["probe_edges_displacement"], input_dict["num_probe_edges"]
        )
        edge_probe_offset = ms.ops.cumsum(
            ms.ops.cat(
                (
                    ms.tensor([0]),
                    input_dict["num_probes"][:-1],
                )
            ),
            axis=0,
        )
        edge_probe_offset = edge_probe_offset[:, None, None]
        edge_probe_offset = ms.ops.cat((edge_offset, edge_probe_offset), axis=2)
        probe_edges = input_dict["probe_edges"] + edge_probe_offset
        probe_edges = layer.unpad_and_cat(probe_edges, input_dict["num_probe_edges"])

        # Compute edge distances
        probe_edges_distance, probe_edges_diff = layer.calc_distance_to_probe(
            atom_xyz,
            probe_xyz,
            input_dict["cell"],
            probe_edges,
            probe_edges_displacement,
            input_dict["num_probe_edges"],
            return_diff=True,
        )

        # Expand edge features in sinc basis
        edge_state = layer.sinc_expansion(
            probe_edges_distance, [(self.distance_embedding_size, self.cutoff)]
        )

        # Apply interaction layers
        probe_state_scalar = ms.ops.zeros(
            (ms.ops.sum(input_dict["num_probes"]), self.hidden_state_size),
        )
        probe_state_vector = ms.ops.zeros(
            (ms.ops.sum(input_dict["num_probes"]), 3, self.hidden_state_size),
        )

        for msg_layer, update_layer, atom_nodes_scalar, atom_nodes_vector in zip(
            self.message_layers,
            self.scalar_vector_update,
            atom_representation_scalar,
            atom_representation_vector,
        ):
            probe_state_scalar, probe_state_vector = msg_layer(
                atom_nodes_scalar,
                atom_nodes_vector,
                probe_state_scalar,
                probe_state_vector,
                edge_state,
                probe_edges_diff,
                probe_edges_distance,
                probe_edges,
            )
            probe_state_scalar, probe_state_vector = update_layer(
                probe_state_scalar, probe_state_vector
            )

        # Restack probe states
        probe_output = self.readout_function(probe_state_scalar).squeeze(1)
        probe_output = layer.pad_and_stack(
            ms.ops.split(
                probe_output,
                list(input_dict["num_probes"].detach().cpu().numpy()),
                axis=0,
            )
            # torch.split(probe_output, input_dict["num_probes"], dim=0)
            # probe_output.reshape((-1, input_dict["num_probes"][0]))
        )

        if compute_iri or compute_dori or compute_hessian:

            def dp_dxyz_graph(x):
                # Unpad and concatenate edges and features into batch (0th) dimension
                atom_xyz = layer.unpad_and_cat(input_dict["atom_xyz"], input_dict["num_nodes"])
                probe_xyz = layer.unpad_and_cat(
                    x, input_dict["num_probes"]
                )
                edge_offset = ms.ops.cumsum(
                    ms.ops.cat(
                        (
                            ms.tensor([0]),
                            input_dict["num_nodes"][:-1],
                        )
                    ),
                    axis=0,
                )
                edge_offset = edge_offset[:, None, None]

                # Unpad and concatenate probe edges into batch (0th) dimension
                probe_edges_displacement = layer.unpad_and_cat(
                    input_dict["probe_edges_displacement"], input_dict["num_probe_edges"]
                )
                edge_probe_offset = ms.ops.cumsum(
                    ms.ops.cat(
                        (
                            ms.tensor([0]),
                            input_dict["num_probes"][:-1],
                        )
                    ),
                    axis=0,
                )
                edge_probe_offset = edge_probe_offset[:, None, None]
                edge_probe_offset = ms.ops.cat((edge_offset, edge_probe_offset), axis=2)
                probe_edges = input_dict["probe_edges"] + edge_probe_offset
                probe_edges = layer.unpad_and_cat(probe_edges, input_dict["num_probe_edges"])

                # Compute edge distances
                probe_edges_distance, probe_edges_diff = layer.calc_distance_to_probe(
                    atom_xyz,
                    probe_xyz,
                    input_dict["cell"],
                    probe_edges,
                    probe_edges_displacement,
                    input_dict["num_probe_edges"],
                    return_diff=True,
                )

                # Expand edge features in sinc basis
                edge_state = layer.sinc_expansion(
                    probe_edges_distance, [(self.distance_embedding_size, self.cutoff)]
                )

                # Apply interaction layers
                probe_state_scalar = ms.ops.zeros(
                    (ms.ops.sum(input_dict["num_probes"]), self.hidden_state_size),
                )
                probe_state_vector = ms.ops.zeros(
                    (ms.ops.sum(input_dict["num_probes"]), 3, self.hidden_state_size),
                )

                for msg_layer, update_layer, atom_nodes_scalar, atom_nodes_vector in zip(
                        self.message_layers,
                        self.scalar_vector_update,
                        atom_representation_scalar,
                        atom_representation_vector,
                ):
                    probe_state_scalar, probe_state_vector = msg_layer(
                        atom_nodes_scalar,
                        atom_nodes_vector,
                        probe_state_scalar,
                        probe_state_vector,
                        edge_state,
                        probe_edges_diff,
                        probe_edges_distance,
                        probe_edges,
                    )
                    probe_state_scalar, probe_state_vector = update_layer(
                        probe_state_scalar, probe_state_vector
                    )

                # Restack probe states
                probe_output = self.readout_function(probe_state_scalar).squeeze(1)
                probe_output = layer.pad_and_stack(
                    ms.ops.split(
                        probe_output,
                        list(input_dict["num_probes"].detach().cpu().numpy()),
                        axis=0,
                    )
                )
                return probe_output

            dp_dxyz = ms.grad(dp_dxyz_graph, grad_position=0)(input_dict["probe_xyz"])[0]

        grad_probe_outputs = {}

        if compute_iri:
            iri = ms.ops.norm(dp_dxyz, dim=2)/(ms.ops.pow(probe_output, 1.1))
            grad_probe_outputs["iri"] = iri

        if compute_dori:
            ##
            ## DORI(r) = phi(r) / (1 + phi(r))
            ## phi(r) = ||grad(||grad(rho(r))/rho||^2)||^2 / ||grad(rho(r))/rho(r)||^6
            ##
            norm_grad_2 = ms.ops.norm(dp_dxyz/(ms.ops.unsqueeze(probe_output, 2)), dim=2)**2

            def grad_norm_grad_2_graph(x):
                # Unpad and concatenate edges and features into batch (0th) dimension
                atom_xyz = layer.unpad_and_cat(input_dict["atom_xyz"], input_dict["num_nodes"])
                probe_xyz = layer.unpad_and_cat(
                    x, input_dict["num_probes"]
                )
                edge_offset = ms.ops.cumsum(
                    ms.ops.cat(
                        (
                            ms.tensor([0]),
                            input_dict["num_nodes"][:-1],
                        )
                    ),
                    axis=0,
                )
                edge_offset = edge_offset[:, None, None]

                # Unpad and concatenate probe edges into batch (0th) dimension
                probe_edges_displacement = layer.unpad_and_cat(
                    input_dict["probe_edges_displacement"], input_dict["num_probe_edges"]
                )
                edge_probe_offset = ms.ops.cumsum(
                    ms.ops.cat(
                        (
                            ms.tensor([0]),
                            input_dict["num_probes"][:-1],
                        )
                    ),
                    axis=0,
                )
                edge_probe_offset = edge_probe_offset[:, None, None]
                edge_probe_offset = ms.ops.cat((edge_offset, edge_probe_offset), axis=2)
                probe_edges = input_dict["probe_edges"] + edge_probe_offset
                probe_edges = layer.unpad_and_cat(probe_edges, input_dict["num_probe_edges"])

                # Compute edge distances
                probe_edges_distance, probe_edges_diff = layer.calc_distance_to_probe(
                    atom_xyz,
                    probe_xyz,
                    input_dict["cell"],
                    probe_edges,
                    probe_edges_displacement,
                    input_dict["num_probe_edges"],
                    return_diff=True,
                )

                # Expand edge features in sinc basis
                edge_state = layer.sinc_expansion(
                    probe_edges_distance, [(self.distance_embedding_size, self.cutoff)]
                )

                # Apply interaction layers
                probe_state_scalar = ms.ops.zeros(
                    (ms.ops.sum(input_dict["num_probes"]), self.hidden_state_size),
                )
                probe_state_vector = ms.ops.zeros(
                    (ms.ops.sum(input_dict["num_probes"]), 3, self.hidden_state_size),
                )

                for msg_layer, update_layer, atom_nodes_scalar, atom_nodes_vector in zip(
                        self.message_layers,
                        self.scalar_vector_update,
                        atom_representation_scalar,
                        atom_representation_vector,
                ):
                    probe_state_scalar, probe_state_vector = msg_layer(
                        atom_nodes_scalar,
                        atom_nodes_vector,
                        probe_state_scalar,
                        probe_state_vector,
                        edge_state,
                        probe_edges_diff,
                        probe_edges_distance,
                        probe_edges,
                    )
                    probe_state_scalar, probe_state_vector = update_layer(
                        probe_state_scalar, probe_state_vector
                    )

                # Restack probe states
                probe_output = self.readout_function(probe_state_scalar).squeeze(1)
                probe_output = layer.pad_and_stack(
                    ms.ops.split(
                        probe_output,
                        list(input_dict["num_probes"].detach().cpu().numpy()),
                        axis=0,
                    )
                )
                norm_grad_2 = ms.ops.norm(dp_dxyz / (ms.ops.unsqueeze(probe_output, 2)), dim=2) ** 2
                return norm_grad_2

            grad_norm_grad_2 = ms.grad(grad_norm_grad_2_graph, grad_position=0)(input_dict["probe_xyz"])[0]

            phi_r = ms.ops.norm(grad_norm_grad_2, dim=2)**2 / (norm_grad_2**3)

            dori = phi_r / (1 + phi_r)
            grad_probe_outputs["dori"] = dori

        if compute_hessian:
            hessian_shape = (input_dict["probe_xyz"].shape[0], input_dict["probe_xyz"].shape[1], 3, 3)
            hessian = ms.ops.zeros(hessian_shape, dtype=probe_xyz.dtype)
            for dim_idx, grad_out in enumerate(ms.ops.unbind(dp_dxyz, dim=-1)):

                def dp2_dxyz2_graph(x):
                    # Unpad and concatenate edges and features into batch (0th) dimension
                    atom_xyz = layer.unpad_and_cat(input_dict["atom_xyz"], input_dict["num_nodes"])
                    probe_xyz = layer.unpad_and_cat(
                        x, input_dict["num_probes"]
                    )
                    edge_offset = ms.ops.cumsum(
                        ms.ops.cat(
                            (
                                ms.tensor([0]),
                                input_dict["num_nodes"][:-1],
                            )
                        ),
                        axis=0,
                    )
                    edge_offset = edge_offset[:, None, None]

                    # Unpad and concatenate probe edges into batch (0th) dimension
                    probe_edges_displacement = layer.unpad_and_cat(
                        input_dict["probe_edges_displacement"], input_dict["num_probe_edges"]
                    )
                    edge_probe_offset = ms.ops.cumsum(
                        ms.ops.cat(
                            (
                                ms.tensor([0]),
                                input_dict["num_probes"][:-1],
                            )
                        ),
                        axis=0,
                    )
                    edge_probe_offset = edge_probe_offset[:, None, None]
                    edge_probe_offset = ms.ops.cat((edge_offset, edge_probe_offset), axis=2)
                    probe_edges = input_dict["probe_edges"] + edge_probe_offset
                    probe_edges = layer.unpad_and_cat(probe_edges, input_dict["num_probe_edges"])

                    # Compute edge distances
                    probe_edges_distance, probe_edges_diff = layer.calc_distance_to_probe(
                        atom_xyz,
                        probe_xyz,
                        input_dict["cell"],
                        probe_edges,
                        probe_edges_displacement,
                        input_dict["num_probe_edges"],
                        return_diff=True,
                    )

                    # Expand edge features in sinc basis
                    edge_state = layer.sinc_expansion(
                        probe_edges_distance, [(self.distance_embedding_size, self.cutoff)]
                    )

                    # Apply interaction layers
                    probe_state_scalar = ms.ops.zeros(
                        (ms.ops.sum(input_dict["num_probes"]), self.hidden_state_size),
                    )
                    probe_state_vector = ms.ops.zeros(
                        (ms.ops.sum(input_dict["num_probes"]), 3, self.hidden_state_size),
                    )

                    for msg_layer, update_layer, atom_nodes_scalar, atom_nodes_vector in zip(
                            self.message_layers,
                            self.scalar_vector_update,
                            atom_representation_scalar,
                            atom_representation_vector,
                    ):
                        probe_state_scalar, probe_state_vector = msg_layer(
                            atom_nodes_scalar,
                            atom_nodes_vector,
                            probe_state_scalar,
                            probe_state_vector,
                            edge_state,
                            probe_edges_diff,
                            probe_edges_distance,
                            probe_edges,
                        )
                        probe_state_scalar, probe_state_vector = update_layer(
                            probe_state_scalar, probe_state_vector
                        )

                    # Restack probe states
                    probe_output = self.readout_function(probe_state_scalar).squeeze(1)
                    probe_output = layer.pad_and_stack(
                        ms.ops.split(
                            probe_output,
                            list(input_dict["num_probes"].detach().cpu().numpy()),
                            axis=0,
                        )
                    )
                    dp_dxyz = ms.grad(dp_dxyz_graph, grad_position=0)(input_dict["probe_xyz"])[0]
                    grad_out = ms.ops.unbind(dp_dxyz, dim=-1)[dim_idx]
                    return grad_out

                dp2_dxyz2 = ms.grad(dp2_dxyz2_graph, grad_position=0)(input_dict["probe_xyz"])[0]
                hessian[:, :, dim_idx] = dp2_dxyz2
            grad_probe_outputs["hessian"] = hessian


        if grad_probe_outputs:
            return probe_output, grad_probe_outputs
        else:
            return probe_output
