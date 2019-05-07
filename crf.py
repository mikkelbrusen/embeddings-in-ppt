import torch
from torch.autograd import Variable
from torch import nn

def broadcast(*inputs):
    '''Assumes that all inputs have the same or singleton dimensions'''
    # get the N incoming shapes of dim D
    shp = [(x.size()) for x in inputs]
    # stack the shapes as an N x D matrix and max over the N dim
    shp = torch.IntTensor(shp).max(0)[0].squeeze(0)
    return [x.expand(*shp) for x in inputs]


def log_sum_exp(x, axis=-1, keepdims=False):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    axis = x.ndimension() - 1 if axis == -1 else axis
    m = x.max(axis)[0].unsqueeze(-1)
    out = m.squeeze(-1) + (x - m.expand(x.size())).exp().sum(axis).log()
    if not keepdims:
        out = out.squeeze(axis)
    return out


def onehot(class_vec, num_class):
    if class_vec.is_cuda:
        onehot = Variable(
            torch.eye(num_class)[class_vec.data.view(-1)].cuda())
    else:
        onehot = Variable(torch.eye(num_class)[class_vec.data.view(-1)])

    shp = list(class_vec.size())
    return onehot.view(shp + [num_class])


def forward_step(nu, f, g):
    g, nu = broadcast(g, nu.unsqueeze(-1))
    return f + log_sum_exp(g + nu, axis=-2)


#def forward_step(nu, f, g):
#    g, nu = broadcast(g, nu.unsqueeze(-1))
#    return f + log_sum_exp(g + nu, axis=-1)


def forward_pass(f, g, mask):
    if f.is_cuda:
        nu = Variable(torch.zeros(f.size()).cuda())
    else:
        nu = Variable(torch.zeros(f.size()))

    nu[0] = f[0]  # n_1 = f_(y_1)
    sequence_length = f.size(0)
    for index in range(1, sequence_length):
        nu[index] = forward_step(nu[index - 1], f[index], g[
            index]) * mask.permute(1,0).unsqueeze(-1).expand(f.size()).float()[index]  # nu_i = nu_{i-1}*f_(y_i)*g(y_{i-1},y_{i})
    return nu


def backward_step(nu, f, g):
    nu, f, g = broadcast(nu.unsqueeze(-2), f.unsqueeze(-2), g)
    return log_sum_exp(nu + f + g, axis=-1, keepdims=True)

#def backward_step(nu, f, g):
#    nu, f, g = broadcast(nu.unsqueeze(-1), f.unsqueeze(-1), g)
#    return log_sum_exp(nu + f + g, axis=-1)


def backward_pass(f, g, mask):
    if f.is_cuda:
        nu = Variable(torch.zeros(f.size()).cuda())
    else:
        nu = Variable(torch.zeros(f.size()))

    sequence_length = f.size(0)
    for index in range(1, sequence_length)[::-1]:  # the [::-1] reverses the list to [N ... 1]
        nu[index - 1] = backward_step(nu[index], f[index], g[index]) * mask.permute(1,0).unsqueeze(-1).expand(f.size()).float()[index]  # nu_{i-1} = nu_i*f_(y_i)*g(y_{i-1},y_{i})
    return nu


def logZ(nu_alp, nu_bet, index=0):
    return log_sum_exp(nu_alp[index] + nu_bet[index], axis=-1, keepdims=True)


def log_marginal(nu_alp, nu_bet):
    z_term  = logZ(nu_alp, nu_bet).unsqueeze(0).unsqueeze(2)  # 1, bs, 1
    z_term, nu_alp, nu_bet = broadcast(z_term, nu_alp, nu_bet)
    res = nu_alp + nu_bet - z_term
    return res


def log_likelihood(y, f, g, nu_alp, nu_bet):
    num_class = f.size(2)
    y_oh = onehot(y, num_class).float().permute(1,0,2)
    f_term = (f * y_oh).sum(0).sum(1)

    y_i = y_oh[:-1].unsqueeze(3)
    y_plus = y_oh[1:].unsqueeze(2)

    g, y_i, y_plus = broadcast(g, y_i, y_plus)

    g_term = (g * y_i * y_plus).sum(0).sum(1).sum(1)
    z_term = logZ(nu_alp, nu_bet)
    log_like = f_term + g_term - z_term
    return log_like


def forward_step_max(nu, f, g, class_dimension):
    g, nu = broadcast(g, nu.unsqueeze(class_dimension + 1))
    m, _ = (g + nu).max(class_dimension)
    return f + m



def viterbi(f, g, time_major=False):
    num_class = f.size(2)
    if not time_major:
        f = f.permute(1, 0, 2)
        g = g.permute(1, 0, 2, 3)

    sequence_length = f.size(0)
    class_dimension = 1 

    if f.is_cuda:
        nu = Variable(torch.zeros(f.size()).cuda())
        nu_label = Variable(torch.zeros(f.size()).long().cuda())
        viterbi_seq = Variable(torch.zeros(f[:, :, 0].size()).long().cuda())
    else:
        nu = Variable(torch.zeros(f.size()))
        nu_label = Variable(torch.zeros(f.size()).long())
        viterbi_seq = Variable(torch.zeros(f[:, :, 0].size()).long())

    nu[0] = f[0]

    for index in range(1, sequence_length):
        nu[index] = forward_step_max(nu[index - 1], f[index], g[index - 1],
                                     class_dimension)

        _, _idx = sum(broadcast(nu[index - 1].unsqueeze(class_dimension + 1),
                                g[index - 1])).max(class_dimension)
        nu_label[index] = _idx

    _, _idx = nu[-1].max(class_dimension)
    viterbi_seq[-1] = _idx

    for index in range(1, sequence_length):
        viterbi_seq[-index - 1] = (
            nu_label[-index] * onehot(viterbi_seq[-index],
                                      num_class).long()).sum(1).squeeze(1)
    if not time_major:
        viterbi_seq = viterbi_seq.transpose(1, 0)
    return viterbi_seq



