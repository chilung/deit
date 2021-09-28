import torch
import torch.nn as nn

'''
map_name:
    'attention': named buffer '_attention_map' will be used
    'feature': named buffer '_feature_map' will be used
inter_layer:
    True: cross layer similarity will be calculated
    False: intra layer similarity will be calculated
similarity_type:
    'head': head similarity will be calculated
    'token': token similarity will be calculated
'''
def cross_layer_similaity(model, map_name='attention', inter_layer=False, similarity_type='token', threshold=0.5):
    assert map_name in ['attention', 'feature']
    
    map_name = '_attention_similarity' if map_name=='attention' else '_feature_similarity'
    map_buf = [buf for name, buf in model.named_buffers() if map_name in name]
    cos_sim = 0
    for buf in map_buf:
        cos_sim += buf
    
    print('Gradient function for final cos_sim =', cos_sim.grad_fn)
    return cos_sim

'''
map_name:
    'attention': named buffer '_attention_map' will be used
    'feature': named buffer '_feature_map' will be used
inter_layer:
    True: cross layer similarity will be calculated
    False: intra layer similarity will be calculated
similarity_type:
    'head': head similarity will be calculated
    'token': token similarity will be calculated
'''
def cross_layer_similaity_old(model, map_name='attention', inter_layer=False, similarity_type='token', threshold=0.5):
    assert map_name in ['attention', 'feature']
    assert similarity_type in ['head', 'token']
    
    # print([name for name, _ in model.named_buffers() if map_name in name])
    
    map_name = '_attention_map' if map_name=='attention' else '_feature_map'
    map_buf = torch.tensor(
        torch.stack(tuple([buf for name, buf in model.named_buffers() if map_name in name]))).cuda()
        
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    if similarity_type == 'head':
        cos_sim = cos(
            map_buf.flatten(-2)[..., None, :, :], map_buf.roll(shifts=-1, dims=0).flatten(-2)[..., :, None, :]) if inter_layer else cos(map_buf[..., None, :, :], map_buf[..., :, None, :])
    else:
        cos_sim = cos(map_buf[..., None, :, :], map_buf.roll(shifts=-1, dims=0)[..., :, None, :]) if inter_layer else cos(map_buf[..., None, :, :], map_buf[..., :, None, :])

    # print(cos_sim)
    # cos_sim[cos_sim > threshold] = 1
    # cos_sim[cos_sim <= threshold] = 0

    # if similarity_type == 'head':
    #     cos_sim = torch.sum(torch.count_nonzero(cos_sim, dim=2),dim=1) / (cos_sim.shape[1]*cos_sim.shape[2])
    # else:
    #     cos_sim = torch.sum(torch.sum(torch.count_nonzero(cos_sim, dim=3),dim=2),dim=1) / (cos_sim.shape[1]*cos_sim.shape[2]*cos_sim.shape[3])
    # print(cos_sim)

    cos_sim = torch.mean(cos_sim)
    # print(cos_sim)

    return cos_sim
