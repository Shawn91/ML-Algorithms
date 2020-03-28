import numpy as np


def _cross_entropy_error(scores, labels):
    if scores.ndim == 1:
        labels = labels.reshape(1, labels.size)
        scores = scores.reshape(1, scores.size)
    delta = 1e-7
    return -np.sum(labels * np.log(scores + delta)) / labels.shape[0]


def _softmax(x):
    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


def _sample_data(data, sample_size, seed=42):
    np.random.RandomState(seed)
    mask = np.random.choice(data, sample_size)


class AffineLayer:
    def __init__(self, W=None, B=None, pre_layer_node_num=None, cur_layer_node_num=None):
        if W is None or B is None:
            W = 0.01 * np.random.randn(pre_layer_node_num, cur_layer_node_num)
            B = np.zeros(1, cur_layer_node_num)
        self.W = W
        self.B = B
        self.input = None
        self.dW = None
        self.dB = None
        self.learning_rate = None

    def forward(self, input):
        """
        以第一个 Affine 层（也就是整个网络出了输入层外的第 0 层）为例
        输入 Input 应该是 m 个数据点， shape 为 (n, m)，即每一个数据点是行向量
        W 是 (m, k)，k 是本层的 nodes 数量
        B 是 (1, k)
        """
        self.input = input
        return np.dot(self.input, self.W) + self.B

    def backward(self, input):
        dInput = np.dot(input, self.W.T)
        self.dW = np.dot(self.X.T, input)
        self.dB = np.sum(input, axis=0)
        self.W -= self.learning_rate * self.dW
        self.B -= self.learning_rate * self.dW
        return dInput

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate


class SigmoidLayer:
    """以 sigmoid 函数作为 activation function 的层"""
    def __init__(self):
        self.result = None

    def forward(self, input):
        self.result = 1 / (1 + np.exp(-1 * input))
        return self.result

    # 计算 sigmoid 的梯度
    def backward(self, input):
        return self.result * (1 - self.result) * input


class ReluLayer:
    def __init__(self):
        self.mask = None  # 记录哪些元素小于等于 0

    def forward(self, input):
        # input 是 wx+b 的值，如果只有一个数据点，input 就是标量；如果同时传递多个数据点，就是向量
        self.mask = input <= 0
        result = input.copy()
        result[self.mask] = 0
        return result

    def backward(self, input):
        # input 是从下一层反向传回来的输入
        input[self.mask] = 0
        return input


class SoftmaxLayer:
    """以 softmax 函数作为 activation function 的层。
    由于这一般都是输出层，因此同时使用 cross entropy error 计算总的损失函数
    """
    def __init__(self):
        self.result = None  # softmax 的输出
        self.cost = None
        self.labels = None  # 实际标注数据

    def forward(self, input, labels):
        self.labels = labels
        self.result = _softmax(input)
        self.cost = _cross_entropy_error(self.result, labels)
        return self.cost

    def backward(self, input):
        num_data = input.shape[0]
        return (self.result - self.labels) / num_data


class NeuralNetwork:
    """输入数据的 shape 为 数据量 * features数
    label 的 shape 为 数据量 * 1
    """
    LAYER_NAMES = {'sigmoid': SigmoidLayer, 'softmax': SoftmaxLayer, 'relu': ReluLayer}

    def __init__(self, input_size, hidden_layer_sizes, output_size, activate_function='sigmoid',
                 activate_function_outputlayer='softmax', learning_rate=0.01):
        """
        :param input_size: int, features 数量
        :param hidden_layer_sizes: list of int, 每一层 hidden layer 的 nodes 数; None, 0, [] 都表示无 hidden layer
        :param output_size: int, 输出层 nodes 数
        """
        if not hidden_layer_sizes:
            hidden_layer_sizes = []

        self.layers = []
        for i in range(len(hidden_layer_sizes)+1):
            if i == 0:
                self.layers.extend([AffineLayer(pre_layer_node_num=input_size, cur_layer_node_num=hidden_layer_sizes[0]),
                                    self.LAYER_NAMES[activate_function]])
            elif i == len(hidden_layer_sizes):
                self.layers.extend([AffineLayer(pre_layer_node_num=hidden_layer_sizes[-1], cur_layer_node_num=output_size),
                                    self.LAYER_NAMES[activate_function_outputlayer]])
            else:
                self.layers.extend([AffineLayer(pre_layer_node_num=hidden_layer_sizes[i-1], cur_layer_node_num=hidden_layer_sizes[i]),
                                    self.LAYER_NAMES[activate_function]])


    def _loss(self, X, labels):
        scores = self.predict(X=X)
        # 最后一层的 softmax 层在 forward 函数中计算了损失，因此直接调用 forward 即可
        return self.layers[-1].forward(scores, labels)

    def _update_parameters(self, X, labels):
        loss = self._loss(X, labels)
        for layer in self.layers.reverse():
            # backward 调用时，会自动更新每个 affine 层的参数
            loss = layer.backward(loss)

    def predict(self, X):
        # 预测时无需计算 softmax 层，因此取 [:-1]
        for layer in self.layers[:-1]:
            X = layer.forward(X)
        return X

    def cal_accuracy(self, X, labels):
        predictions = self.predict(X)
        return np.sum(predictions == labels) / labels.shape[0]

    def train(self, X, labels, learning_rate=None, num_iter=2000):
        for layer in self.layers:
            if isinstance(layer, AffineLayer):
                layer.set_learning_rate(learning_rate)

        for i in range(num_iter):
            self._update_parameters(X, labels)



if __name__ == '__main__':
    pass

