from typing import Tuple, List
import itertools

import mindspore as ms
import mindspore.nn as nn

import numpy as np


def pad_and_stack(tensors: List[ms.Tensor]):
    """Pad list of tensors if tensors are arrays and stack if they are scalars"""

    if len(tensors[0]) > 0:
        # get the padding paras
        len_list = []
        for t in tensors:
            len_list.append(t.shape[0])
        m = max(len_list)

        padding_list = []
        for p in len_list:
            padding_list.append(m - p)

        # process tensors
        padded_tensors = []
        for i in range(len(tensors)):
            padded_tensor = ms.ops.Pad(((0, padding_list[i]), (0, 0)))(tensors[i])
            padded_tensors.append(padded_tensor)
            i += 1

        # stack tensors
        return ms.ops.Stack(axis=0)(padded_tensors)

    else:
        # stack tensors
        return ms.ops.Stack(axis=0)(tensors)



def shifted_softplus(x):
    """
    Compute shifted soft-plus activation function.
    .. math::
       y = \ln\left(1 + e^{-x}\right) - \ln(2)
    Args:
        x (minspore.Tensor): input tensor.
    Returns:
        minspore.Tensor: shifted soft-plus of input.
    """

    y = ms.ops.log(1 + ms.ops.exp(-x)) - np.log(2.0)
    return y


class ShiftedSoftplus(nn.Cell):
    def construct(self, x):
        return shifted_softplus(x)


def unpad_and_cat(stacked_seq: ms.Tensor, seq_len: ms.Tensor):
    """
    Unpad and concatenate by removing batch dimension

    Args:
        stacked_seq: (batch_size, max_length, *) Tensor
        seq_len: (batch_size) Tensor with length of each sequence

    Returns:
        (prod(seq_len), *) Tensor

    """

    unstacked = ms.ops.Unstack(num=stacked_seq.shape[0])(stacked_seq)

    unpadded = []
    for t, l in zip(unstacked, ms.ops.Unstack()(seq_len)):
        unpadded.append(t[:l])

    return ms.ops.Concat(axis=0)(unpadded)


def sum_splits(values: ms.Tensor, splits: ms.Tensor):
    """
    Sum across dimension 0 of the tensor `values` in chunks
    defined in `splits`

    Args:
        values: Tensor of shape (`prod(splits)`, *)
        splits: 1-dimensional tensor with size of each chunk

    Returns:
        Tensor of shape (`splits.shape[0]`, *)

    """

    # prepare an index vector for summation
    ind = ms.ops.zeros(splits.sum(), dtype=splits.dtype)
    ind[splits.cumsum(axis=0)[:-1]] = 1
    ind = ind.cumsum(axis=0)
    # prepare the output
    sum_y = ms.ops.zeros(splits.shape + values.shape[1:], dtype=values.dtype)

    # do the actual summation
    sum_y = ms.ops.index_add(sum_y, ind, values, 0)

    return sum_y


def calc_distance(
    positions: ms.Tensor,
    cells: ms.Tensor,
    edges: ms.Tensor,
    edges_displacement: ms.Tensor,
    splits: ms.Tensor,
    return_diff=False,
):
    """
    Calculate distance of edges

    Args:
        positions: Tensor of shape (num_nodes, 3) with xyz coordinates inside cell
        cells: Tensor of shape (num_splits, 3, 3) with one unit cell for each split
        edges: Tensor of shape (num_edges, 2)
        edges_displacement: Tensor of shape (num_edges, 3) with the offset (in number of cell vectors) of the sending node
        splits: 1-dimensional tensor with the number of edges for each separate graph
        return_diff: If non-zero return vector corresponding to edges
    """
    unitcell_repeat = ms.ops.repeat_interleave(cells, splits, axis=0)
    displacement = ms.ops.matmul(
        ms.ops.unsqueeze(edges_displacement, 1), unitcell_repeat
    )  # num_edges, 1, 3
    displacement = ms.ops.squeeze(displacement, axis=1)
    neigh_pos = positions[edges[:, 0]]  # num_edges, 3
    neigh_abs_pos = neigh_pos + displacement  # num_edges, 3
    this_pos = positions[edges[:, 1]]  # num_edges, 3
    diff = this_pos - neigh_abs_pos  # num_edges, 3
    dist = ms.ops.sqrt(
        ms.ops.sum(ms.ops.square(diff), dim=1, keepdim=True)
    )  # num_edges, 1

    if return_diff:
        return dist, diff
    else:
        return dist


def calc_distance_to_probe(
    positions: ms.Tensor,
    positions_probe: ms.Tensor,
    cells: ms.Tensor,
    edges: ms.Tensor,
    edges_displacement: ms.Tensor,
    splits: ms.Tensor,
    return_diff=False,
):
    """
    Calculate distance of edges

    Args:
        positions: Tensor of shape (num_nodes, 3) with xyz coordinates inside cell
        positions_probe: Tensor of shape (num_probes, 3) with xyz coordinates of probes inside cell
        cells: Tensor of shape (num_splits, 3, 3) with one unit cell for each split
        edges: Tensor of shape (num_edges, 2)
        edges_displacement: Tensor of shape (num_edges, 3) with the offset (in number of cell vectors) of the sending node
        splits: 1-dimensional tensor with the number of edges for each separate graph
        return_diff: If non-zero return vector corresponding to edges
    """
    unitcell_repeat = ms.ops.repeat_interleave(cells, splits, axis=0)

    displacement = ms.ops.matmul(
        ms.ops.unsqueeze(edges_displacement, 1), unitcell_repeat
    )  # num_edges, 1, 3

    displacement = ms.ops.squeeze(displacement, 1)

    neigh_pos = positions[edges[:, 0]]  # num_edges, 3
    neigh_abs_pos = neigh_pos + displacement  # num_edges, 3
    this_pos = positions_probe[edges[:, 1]]  # num_edges, 3
    diff = this_pos - neigh_abs_pos  # num_edges, 3

    dist = ms.ops.sqrt(ms.ops.sum(ms.ops.square(diff), dim=1, keepdim=True))  # num_edges, 1
    if return_diff:
        return dist, diff
    else:
        return dist


def gaussian_expansion(input_x: ms.Tensor, expand_params: List[Tuple]):
    """
    Expand each feature in a number of Gaussian basis function.
    Expand_params is a list of length input_x.shape[1]

    Args:
        input_x: (num_edges, num_features) tensor
        expand_params: list of None or (start, step, stop) tuples

    Returns:
        (num_edges, ``ceil((stop - start)/step)``) tensor

    """
    feat_list = ms.ops.unbind(input_x, dim=1)
    expanded_list = []
    for step_tuple, feat in itertools.zip_longest(expand_params, feat_list):
        assert feat is not None, "Too many expansion parameters given"
        if step_tuple:
            start, step, stop = step_tuple
            feat_expanded = ms.ops.unsqueeze(feat, dim=1)
            sigma = step
            basis_mu = ms.ops.arange(
                start, stop, step, dtype=input_x.dtype
            )
            expanded_list.append(
                ms.ops.exp(-((feat_expanded - basis_mu) ** 2) / (2.0 * sigma ** 2))
            )
        else:
            expanded_list.append(ms.ops.unsqueeze(feat, dim=1))
    return ms.ops.cat(expanded_list, axis=1)


class SchnetMessageFunction(nn.Cell):
    def __init__(self, input_size, edge_size, output_size, hard_cutoff):
        super().__init__()
        self.msg_function_edge = nn.SequentialCell(
            nn.Dense(edge_size, output_size),
            ShiftedSoftplus(),
            nn.Dense(output_size, output_size),
        )
        self.msg_function_node = nn.SequentialCell(
            nn.Dense(input_size, input_size),
            ShiftedSoftplus(),
            nn.Dense(input_size, output_size),
        )

        self.soft_cutoff_func = lambda x: 1.0 - ms.ops.sigmoid(
            5 * (x - (hard_cutoff - 1.5))
        )

    def construct(self, node_state, edge_state, edge_distance):
        gates = self.msg_function_edge(edge_state) * self.soft_cutoff_func(edge_distance)
        nodes = self.msg_function_node(node_state)
        return nodes * gates


class MessageSum(nn.Cell):
    def __init__(self, node_size, edge_size, cutoff, include_receiver):
        super().__init__()

        self.include_receiver = include_receiver

        if include_receiver:
            input_size = node_size * 2
        else:
            input_size = node_size

        self.message_function = SchnetMessageFunction(
            input_size, edge_size, node_size, cutoff
        )

    def construct(
        self, node_state, edges, edge_state, edges_distance, receiver_nodes=None
    ):
        """
        Args:
            node_state: [num_nodes, n_node_features] State of input nodes
            edges: [num_edges, 2] array of sender and receiver indices
            edge_state: [num_edges, n_features] array of edge features
            edges_distance: [num_edges, 1] array of distances
            receiver_nodes: If given, use these nodes as receiver nodes instead of node_state

        Returns:
            sum of messages to each node

        """
        # Compute all messages
        if self.include_receiver:
            if receiver_nodes is not None:
                senders = node_state[edges[:, 0]]
                receivers = receiver_nodes[edges[:, 1]]
                nodes = ms.ops.cat((senders, receivers), axis=1)
            else:
                num_edges = edges.shape[0]
                nodes = ms.ops.reshape(node_state[edges], (num_edges, -1))
        else:
            nodes = node_state[edges[:, 0]]  # Only include sender in messages
        messages = self.message_function(nodes, edge_state, edges_distance)

        # Sum messages
        if receiver_nodes is not None:
            message_sum = ms.ops.zeros_like(receiver_nodes)
        else:
            message_sum = ms.ops.zeros_like(node_state)
        message_sum.index_add_(0, edges[:, 1], messages)

        return message_sum


class Interaction(nn.Cell):
    def __init__(self, node_size, edge_size, cutoff, include_receiver=False):
        super().__init__()

        self.message_sum_module = MessageSum(
            node_size, edge_size, cutoff, include_receiver
        )

        self.state_transition_function = nn.SequentialCell(
            nn.Dense(node_size, node_size),
            ShiftedSoftplus(),
            nn.Dense(node_size, node_size),
        )

    def construct(self, node_state, edges, edge_state, edges_distance):

        # Compute sum of messages
        message_sum = self.message_sum_module(
            node_state, edges, edge_state, edges_distance
        )

        # State transition
        new_state = node_state + self.state_transition_function(message_sum)

        return new_state


class EdgeUpdate(nn.Cell):
    def __init__(self, edge_size, node_size):
        super().__init__()

        self.node_size = node_size
        self.edge_update_mlp = nn.SequentialCell(
            nn.Dense(2 * node_size + edge_size, 2 * edge_size),
            ShiftedSoftplus(),
            nn.Dense(2 * edge_size, edge_size),
        )

    def construct(self, edge_state, edges, node_state):
        combined = ms.ops.cat(
            (node_state[edges].view(-1, 2 * self.node_size), edge_state), axis=1
        )
        return self.edge_update_mlp(combined)


class PaiNNUpdate(nn.Cell):
    """PaiNN style update network. Models the interaction between scalar and vectorial part"""

    def __init__(self, node_size):
        super().__init__()

        self.linearU = nn.Dense(node_size, node_size, bias=False)
        self.linearV = nn.Dense(node_size, node_size, bias=False)
        self.combined_mlp = nn.SequentialCell(
            nn.Dense(2 * node_size, node_size),
            nn.SiLU(),
            nn.Dense(node_size, 3 * node_size),
        )

    def construct(self, node_state_scalar, node_state_vector):
        """
        Args:
            node_state_scalar (tensor): Node states (num_nodes, node_size)
            node_state_vector (tensor): Node states (num_nodes, 3, node_size)

        Returns:
            Tuple of 2 tensors:
                updated_node_state_scalar (num_nodes, node_size)
                updated_node_state_vector (num_nodes, 3, node_size)
        """

        Uv = self.linearU(node_state_vector)  # num_nodes, 3, node_size
        Vv = self.linearV(node_state_vector)  # num_nodes, 3, node_size

        Vv_norm = ms.ops.norm(Vv, dim=1, keepdim=False)  # num_nodes, node_size

        mlp_input = ms.ops.cat(
            (node_state_scalar, Vv_norm), axis=1
        )  # num_nodes, node_size*2
        mlp_output = self.combined_mlp(mlp_input)

        a_ss, a_sv, a_vv = ms.ops.split(
            mlp_output, node_state_scalar.shape[1], axis=1
        )  # num_nodes, node_size

        inner_prod = ms.ops.sum(Uv * Vv, dim=1)  # num_nodes, node_size

        delta_v = ms.ops.unsqueeze(a_vv, 1) * Uv  # num_nodes, 3, node_size

        delta_s = a_ss + a_sv * inner_prod  # num_nodes, node_size

        return node_state_scalar + delta_s, node_state_vector + delta_v


class PaiNNInteraction(nn.Cell):
    """Interaction network"""

    def __init__(self, node_size, edge_size, cutoff):
        """
        Args:
            node_size (int): Size of node state
            edge_size (int): Size of edge state
            cutoff (float): Cutoff distance
        """
        super().__init__()

        self.filter_layer = nn.Dense(edge_size, 3 * node_size)

        self.cutoff = cutoff

        self.scalar_message_mlp = nn.SequentialCell(
            nn.Dense(node_size, node_size),
            nn.SiLU(),
            nn.Dense(node_size, 3 * node_size),
        )

    def construct(
        self,
        node_state_scalar,
        node_state_vector,
        edge_state,
        edge_vector,
        edge_distance,
        edges,
    ):
        """
        Args:
            node_state_scalar (tensor): Node states (num_nodes, node_size)
            node_state_vector (tensor): Node states (num_nodes, 3, node_size)
            edge_state (tensor): Edge states (num_edges, edge_size)
            edge_vector (tensor): Edge vector difference between nodes (num_edges, 3)
            edge_distance (tensor): l2-norm of edge_vector (num_edges, 1)
            edges (tensor): Directed edges with node indices (num_edges, 2)

        Returns:
            Tuple of 2 tensors:
                updated_node_state_scalar (num_nodes, node_size)
                updated_node_state_vector (num_nodes, 3, node_size)
        """
        # Compute all messages
        edge_vector_normalised = edge_vector / ms.ops.maximum(
            ms.ops.norm(edge_vector, dim=1, keepdim=True), ms.tensor(1e-12)
        )  # num_edges, 3

        filter_weight = self.filter_layer(edge_state)  # num_edges, 3*node_size
        filter_weight = filter_weight * cosine_cutoff(edge_distance, self.cutoff)

        scalar_output = self.scalar_message_mlp(
            node_state_scalar
        )  # num_nodes, 3*node_size
        scalar_output = scalar_output[edges[:, 0]]  # num_edges, 3*node_size
        filter_output = filter_weight * scalar_output  # num_edges, 3*node_size

        gate_state_vector, gate_edge_vector, gate_node_state = ms.ops.split(
            filter_output, node_state_scalar.shape[1], axis=1
        )

        gate_state_vector = ms.ops.unsqueeze(
            gate_state_vector, 1
        )  # num_edges, 1, node_size
        gate_edge_vector = ms.ops.unsqueeze(
            gate_edge_vector, 1
        )  # num_edges, 1, node_size

        # Only include sender in messages
        messages_scalar = node_state_scalar[edges[:, 0]] * gate_node_state
        messages_state_vector = node_state_vector[
            edges[:, 0]
        ] * gate_state_vector + gate_edge_vector * ms.ops.unsqueeze(
            edge_vector_normalised, 2
        )

        # Sum messages
        message_sum_scalar = ms.ops.zeros_like(node_state_scalar)
        message_sum_scalar.index_add_(0, edges[:, 1], messages_scalar)
        message_sum_vector = ms.ops.zeros_like(node_state_vector)
        message_sum_vector.index_add_(0, edges[:, 1], messages_state_vector)

        # State transition
        new_state_scalar = node_state_scalar + message_sum_scalar
        new_state_vector = node_state_vector + message_sum_vector

        return new_state_scalar, new_state_vector


class PaiNNInteractionOneWay(nn.Cell):
    """Sasme as Interaction network, but the receiving nodes are differently indexed from the sending nodes"""

    def __init__(self, node_size, edge_size, cutoff):
        """
        Args:
            node_size (int): Size of node state
            edge_size (int): Size of edge state
            cutoff (float): Cutoff distance
        """
        super().__init__()

        self.filter_layer = nn.Dense(edge_size, 3 * node_size)

        self.cutoff = cutoff

        self.scalar_message_mlp = nn.SequentialCell(
            nn.Dense(node_size, node_size),
            nn.SiLU(),
            nn.Dense(node_size, 3 * node_size),
        )

        # Ignore messages gate (not part of original PaiNN network)
        self.update_gate_mlp = nn.SequentialCell(
            nn.Dense(node_size, 2 * node_size),
            nn.SiLU(),
            nn.Dense(2 * node_size, 2 * node_size),
            nn.Sigmoid(),
        )

    def construct(
        self,
        sender_node_state_scalar,
        sender_node_state_vector,
        receiver_node_state_scalar,
        receiver_node_state_vector,
        edge_state,
        edge_vector,
        edge_distance,
        edges,
    ):
        """
        Args:
            sender_node_state_scalar (tensor): Node states (num_nodes, node_size)
            sender_node_state_vector (tensor): Node states (num_nodes, 3, node_size)
            receiver_node_state_scalar (tensor): Node states (num_nodes, node_size)
            receiver_node_state_vector (tensor): Node states (num_nodes, 3, node_size)
            edge_state (tensor): Edge states (num_edges, edge_size)
            edge_vector (tensor): Edge vector difference between nodes (num_edges, 3)
            edge_distance (tensor): l2-norm of edge_vector (num_edges, 1)
            edges (tensor): Directed edges with node indices (num_edges, 2)

        Returns:
            Tuple of 2 tensors:
                updated_node_state_scalar (num_nodes, node_size)
                updated_node_state_vector (num_nodes, 3, node_size)
        """
        # Compute all messages
        edge_vector_normalised = edge_vector / ms.ops.maximum(
            ms.ops.norm(edge_vector, dim=1, keepdim=True), ms.tensor(1e-12)
        )  # num_edges, 3

        filter_weight = self.filter_layer(edge_state)  # num_edges, 3*node_size
        filter_weight = filter_weight * cosine_cutoff(edge_distance, self.cutoff)

        scalar_output = self.scalar_message_mlp(
            sender_node_state_scalar
        )  # num_nodes, 3*node_size
        scalar_output = scalar_output[edges[:, 0]]  # num_edges, 3*node_size
        filter_output = filter_weight * scalar_output  # num_edges, 3*node_size

        gate_state_vector, gate_edge_vector, gate_node_state = ms.ops.split(
            filter_output, sender_node_state_scalar.shape[1], axis=1
        )

        gate_state_vector = ms.ops.unsqueeze(
            gate_state_vector, 1
        )  # num_edges, 1, node_size
        gate_edge_vector = ms.ops.unsqueeze(
            gate_edge_vector, 1
        )  # num_edges, 1, node_size

        # Only include sender in messages
        messages_scalar = sender_node_state_scalar[edges[:, 0]] * gate_node_state
        messages_state_vector = sender_node_state_vector[
            edges[:, 0]
        ] * gate_state_vector + gate_edge_vector * ms.ops.unsqueeze(
            edge_vector_normalised, 2
        )

        # Sum messages
        message_sum_scalar = ms.ops.zeros_like(receiver_node_state_scalar)
        message_sum_scalar.index_add_(0, edges[:, 1], messages_scalar)
        message_sum_vector = ms.ops.zeros_like(receiver_node_state_vector)
        message_sum_vector.index_add_(0, edges[:, 1], messages_state_vector)

        # State transition
        update_gate_scalar, update_gate_vector = ms.ops.split(
            self.update_gate_mlp(message_sum_scalar),
            receiver_node_state_scalar.shape[1],
            axis=1,
        )
        update_gate_vector = ms.ops.unsqueeze(
            update_gate_vector, 1
        )  # num_nodes, 1, node_size
        new_state_scalar = (
            update_gate_scalar * receiver_node_state_scalar
            + (1.0 - update_gate_scalar) * message_sum_scalar
        )
        new_state_vector = (
            update_gate_vector * receiver_node_state_vector
            + (1.0 - update_gate_vector) * message_sum_vector
        )

        return new_state_scalar, new_state_vector


def sinc_expansion(input_x: ms.Tensor, expand_params: List[Tuple]):
    """
    Expand each feature in a sinc-like basis function expansion.
    Based on [1].
    sin(n*pi*f/rcut)/f

    [1] arXiv:2003.03123 - Directional Message Passing for Molecular Graphs

    Args:
        input_x: (num_edges, num_features) tensor
        expand_params: list of None or (n, cutoff) tuples

    Return:
        (num_edges, n1+n2+...) tensor
    """
    feat_list = ms.ops.unbind(input_x, dim=1)
    expanded_list = []
    for step_tuple, feat in itertools.zip_longest(expand_params, feat_list):
        assert feat is not None, "Too many expansion parameters given"
        if step_tuple:
            n, cutoff = step_tuple
            feat_expanded = ms.ops.unsqueeze(feat, dim=1)
            n_range = ms.ops.arange(n, dtype=input_x.dtype) + 1
            # multiplication by pi n_range / cutoff is done in original painn for some reason
            out = ms.ops.sinc(n_range/cutoff*feat_expanded)*np.pi*n_range/cutoff
            expanded_list.append(out)
        else:
            expanded_list.append(ms.ops.unsqueeze(feat, 1))
    return ms.ops.cat(expanded_list, axis=1)


def cosine_cutoff(distance: ms.Tensor, cutoff: float):
    """
    Calculate cutoff value based on distance.
    This uses the cosine Behler-Parinello cutoff function:

    f(d) = 0.5*(cos(pi*d/d_cut)+1) for d < d_cut and 0 otherwise
    """

    return ms.ops.where(
        distance < cutoff,
        0.5 * (ms.ops.cos(np.pi * distance / cutoff) + 1),
        ms.tensor(0.0, dtype=distance.dtype),
    )
