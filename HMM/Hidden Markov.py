#%%

import torch

#%%

def forward(X, A, B, pi):
    """
    前向算法。本函数没有考虑 underflow 问题，如果考虑 underflow，则可以使用 logsumexp 技巧。
    T 是每个 batch data 中数据长度。N 是 state 数量。M 是所有可能的观测值数量。
    :param X: 输入数据。尺寸为 (batch_size, T)
    :param A: 转移矩阵。尺寸为 (N, N)。
    :param B: 观测矩阵。尺寸为 (N, M)。
    :param pi: 初始状态概率。尺寸为 (N,)
    :return: 每条数据的每一步的每个状态出现概率。一个尺寸为 (batch_size, N, T) 的矩阵
    """
    batch_size = X.shape[0]
    T, N, M = X.shape[1], A.shape[1], B.shape[1]

    alpha = torch.zeros([batch_size, N, T])

    # 初始化 alpha_0
    X_0 =  X[:, 0] # 每条数据第一步的观测值。尺寸为 (batch_size, )
    # alpha[:,:,0] 是尺寸为 (batch_size, N) 的矩阵
    alpha[:,:,0] = pi * (B[:, X_0]).T  # (B[:, X_1]).T 的尺寸为 (batch_size, N)

    # 计算剩余的 alpha
    for t in range(1, T):
        # 这里不是必须要使用 einsum 函数，可以使用普通的 PyTorch api 函数组合。
        alpha[:,:,t] = torch.einsum('bi, ij, bj->bj', alpha[:,:,t-1], A, (B[:, X[:,t]]).T)
    return alpha

#%%

def backward(X, A, B, pi):
    """
    后向算法。本函数没有考虑 underflow 问题，如果考虑 underflow，则可以使用 logsumexp 技巧。
    T 是每个 batch data 中数据长度。N 是 state 数量。M 是所有可能的观测值数量。
    :param X: 输入数据。尺寸为 (batch_size, T)
    :param A: 转移矩阵。尺寸为 (N, N)。
    :param B: 观测矩阵。尺寸为 (N, M)。
    :param pi: 初始状态概率。尺寸为 (N,)
    :return: 每条数据的每一步的每个状态后向概率。一个尺寸为 (batch_size, N, T) 的 Tensor
    """
    batch_size = X.shape[0]
    T, N, M = X.shape[1], A.shape[1], B.shape[1]

    beta = torch.ones([batch_size, N, T])

    for t in range(T-2, -1, -1):
        beta[:,:,t] = (B[:, X[:, t+1]]).T * beta[:,:,t+1] @ A.T
    return beta

#%%

def viterbi(X, A, B, pi):
    """维特比算法。"""
    batch_size = X.shape[0]
    T, N, M = X.shape[1], A.shape[1], B.shape[1]

    path_probs = torch.zeros((batch_size, N, T))
    paths = torch.zeros([batch_size, N, T]).type(torch.int16)

    # initialization
    X_0 =  X[:, 0] # 每条数据第一步的观测值。尺寸为 (batch_size, )
    path_probs[:,:,0] = pi * (B[:, X_0]).T  # (B[:, X_0]).T 的尺寸为 (batch_size, N)

    for t in range(1, T):
        # states_prob 是在 t 步位于各个状态的概率，是 (batch_size, N, N) 的 Tensor
        states_prob = torch.einsum('bi, ij->bij', path_probs[:,:,t-1], A)
        max_state_prob, max_state_prob_indices = torch.max(states_prob, dim=1)
        path_probs[:, :, t] = max_state_prob * (B[:, X[:,t]]).T
        paths[:,:,t] = max_state_prob_indices
    best_path_prob, best_path_last_steps = torch.max(path_probs[:, :, T-1], dim=1)
    paths = paths.roll(-1,2)
    paths[range(len(best_path_last_steps)), best_path_last_steps, -1] = best_path_last_steps.type(torch.int16)
    return best_path_prob, paths[range(len(best_path_last_steps)), best_path_last_steps]

#%%
def _estimate_transition_prob(X, A, B, alpha, beta):
    """
    :param X: 输入数据。尺寸为 (batch_size, T)
    :param A: 转移矩阵。尺寸为 (N, N)。
    :param B: 观测矩阵。尺寸为 (N, M)。
    :param alpha: forward 函数的返回结果，(batch_size, N, T) 的 Tensor
    :param beta: backward 函数的返回结果，(batch_size, N, T) 的 Tensor
    :return: (N, N) 的转移矩阵
    """
    batch_size = X.shape[0]
    T, N, M = X.shape[1], A.shape[1], B.shape[1]
    observation_prob = alpha[:,:,X.shape[1]-1].sum(dim=1)
    epsilon = torch.zeros([batch_size, T-1, N, N])  # 存储每条数据在 t 步从状态 i 的状态 j 的概率

    for t in range(T-1):
        # epsilon[:,t,:,:] 尺寸是 (batch_size, N, N)
        # alpha[:,:,t] 尺寸为(batch_size, N)的矩阵; * A 后是 (batch_size, N, N)
        # (B[:, X[:, t]]).T 的尺寸为 (batch_size, N); beta[:,t+1,:] 是 (batch_size, N)
        epsilon[:,t,:,:] = alpha[:,:,t].view([batch_size, N, 1]) * A * ((B[:, X[:, t+1]]).T * beta[:,:,t+1]).view([batch_size, N, 1])
        epsilon[:,t,:,:] = torch.einsum('bij,b->bij', epsilon[:,t,:,:], 1 / observation_prob)

    return epsilon.sum(dim=[0,1]) / epsilon.sum(dim=[0,1, 3]).view([-1, 1])

def _estimate_emission_initial_prob(X, A, B, alpha, beta):
    batch_size = X.shape[0]
    T, N, M = X.shape[1], A.shape[1], B.shape[1]

    # 长度为 batch_size 的向量
    observation_prob = alpha[:, :, X.shape[1] - 1].sum(dim=1)

    # gamma 是每条数据在 t 时刻处于状态 i 的概率。gamma, alpha, beta 都是 (batch_size, N, T) 矩阵
    gamma = (alpha * beta) / observation_prob.view([-1,1,1])

    initial_prob = gamma[:,:,0].mean(0)

    one_hot_x = torch.zeros([batch_size, T, M])
    one_hot_x.scatter_(2, X.unsqueeze(2), 1)

    # torch.bmm(gamma, one_hot_x) 是 (batch_size, N, M)
    emission_prob = torch.bmm(gamma, one_hot_x) / gamma.sum(2).unsqueeze(2)
    emission_prob = emission_prob.mean(0)
    return emission_prob, initial_prob

def forward_backward(X, state_num, obs_num, iter_num=50):
    # 随机初始化参数值
    A = torch.nn.functional.softmax(torch.rand([state_num, state_num]), dim=1)
    B = torch.nn.functional.softmax(torch.rand([state_num, obs_num]), dim=1)
    pi = torch.nn.functional.softmax(torch.rand([state_num]), dim=0)
    for i in range(iter_num):
        alpha = forward(X, A, B, pi)
        beta = backward(X, A, B, pi)
        A = _estimate_transition_prob(X, A, B, alpha, beta)
        B, pi = _estimate_emission_initial_prob(alpha, beta)
    return A, B, pi



#%%

def check_forward_backward(X, A, B, pi):
    """使用 forward 算法与 backward 算法得到的 P(O|λ) 应该是一样的"""
    alpha = forward(X, A, B, pi)
    beta = backward(X, A, B, pi)
    alpha_prob = alpha[:,:,X.shape[1]-1].sum(dim=1)
    beta_prob = (beta[:,:,0] * pi * (B[:, X[:, 0]]).T).sum(dim=1)
    assert torch.allclose(alpha_prob, beta_prob)

#%%
if __name__ == '__main__':
    torch.manual_seed(0)
    N = 3  # 状态可选值 :[0,2]
    batch_size = 2  # 一个 batch 的数据数
    T = 4  # 最长步数
    M = 4  # 观测值可选值: [0, 3]
    X = torch.randint(0, M, [batch_size,T])
    A = torch.nn.functional.softmax(torch.rand([N, N]), dim=1)
    B = torch.nn.functional.softmax(torch.rand([N, M]), dim=1)
    pi = torch.nn.functional.softmax(torch.rand([N]), dim=0)
    # print(backward(X, A, B, pi))
    a = torch.Tensor([[0.5, 0.2, 0.3], [0.3,0.5,0.2], [0.2,0.3,0.5]])
    b = torch.tensor([[0.5,0.5], [0.4,0.6],[0.7,0.3]])
    p = torch.tensor([0.2,0.4,0.4])
    x = torch.tensor([[0,1,0,0], [0,1,1,1]])
    # print(viterbi(X, A, B, pi))
    alpha = forward(x, a, b, p)
    beta = backward(x, a, b, p)
    _estimate_emission_initial_prob(x, a, b, alpha, beta)