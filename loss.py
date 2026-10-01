import torch
import torch.nn.functional as F


LOSS_WEIGHTS = {
    'cluster': 0.8,
    'separation': -0.08,
    'orthogonality': 1.0,
    'subspace': -1e-7,
}


def nll_loss(logits, labels):
    if labels.dim() == 1:
        labels = F.one_hot(labels, num_classes=logits.size(1)).float()
    else:
        labels = labels.float()
    loss = -labels * F.logsigmoid(logits) - (1 - labels) * F.logsigmoid(-logits)
    return loss.sum(dim=1).mean()


def tesnet_cluster_separation(min_distances, labels, model):
    if labels.dim() == 2:
        labels = labels.argmax(dim=1)
    max_dist = float(model.prototype_shape[1] * model.prototype_shape[2] * model.prototype_shape[3])
    identity = model.prototype_class_identity.to(labels.device)
    pos = identity[:, labels].transpose(0, 1)
    neg = 1.0 - pos
    inv_pos, _ = torch.max((max_dist - min_distances) * pos, dim=1)
    cluster = torch.mean(max_dist - inv_pos)
    inv_neg, _ = torch.max((max_dist - min_distances) * neg, dim=1)
    separation = torch.mean(max_dist - inv_neg)
    return cluster, separation


def orth_and_subspace(model):
    basis = model.prototype_vectors.view(model.num_prototypes, -1)
    subspace = basis.view(model.num_classes, model.num_prototypes_per_class, -1)
    subspace_t = subspace.transpose(1, 2)
    gram = torch.matmul(subspace, subspace_t)
    eye = torch.eye(model.num_prototypes_per_class, device=basis.device, dtype=basis.dtype)
    orth = torch.sum(torch.relu(torch.norm(gram - eye, p=1, dim=[1, 2])))
    proj = torch.matmul(subspace_t, subspace)
    diff = proj.unsqueeze(1) - proj.unsqueeze(0)
    pairwise = torch.norm(diff + 1e-10, p='fro', dim=[2, 3])
    subspace_sep = (
        0.5 * torch.norm(pairwise, p=1, dim=[0, 1], dtype=torch.double)
        / torch.sqrt(torch.tensor(2.0, dtype=torch.double, device=basis.device))
    )
    return orth, subspace_sep.to(dtype=basis.dtype)


def gbtesnet_loss(logits, labels, min_distances, model, weights=None):
    if weights is None:
        weights = LOSS_WEIGHTS
    nll = nll_loss(logits, labels)
    cluster, separation = tesnet_cluster_separation(min_distances, labels, model)
    orth, subspace_sep = orth_and_subspace(model)
    total = (nll
             + weights['cluster'] * cluster
             + weights['separation'] * separation
             + weights['orthogonality'] * orth
             + weights['subspace'] * subspace_sep)
    return total, {
        'nll': nll,
        'cluster': cluster,
        'separation': separation,
        'orth': orth,
        'subspace': subspace_sep,
    }
