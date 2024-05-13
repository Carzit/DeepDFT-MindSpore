import numpy
import mindspore as ms
import mindspore.common.dtype as mstype 

def convert_batch_int32(batch:dict):
    batch32 = {}  
    for key, value in batch.items():  
        if isinstance(value, ms.Tensor) and value.dtype == mstype.int64:   
            converted_value = ms.Tensor(value.asnumpy().astype('int32'), mstype.int32)  
                    #key: [nodes, num_nodes, num_atom_edges, num_probe_edges, num_probes]
        else:   
            converted_value = value
        batch32[key] = converted_value
    return batch32
            