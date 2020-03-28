class Percepton:
    def __init__(self,X,y,lam):
        self.X = X
        self.y = y
        self.w = [0 for _ in range(len(X[0]))] # 初始参数定为 0
        self.b = 0
        self.lam = lam

    def update_term(self, X_one_point,y_one_point):
        for i in range(len(self.w)):
            self.w[i] += self.lam*y_one_point*X_one_point[i]
        self.b += self.lam*y_one_point

    def classify(self,X_one_point, y_one_point):
        result = 0
        for x,w in zip(X_one_point, self.w):
            result += x*w
        result += self.b
        return y_one_point*result>0 # 返回是否判断正确
    
    def train(self):
        num_iter = 0
        while num_iter < 1000:
            num_misclassified = 0
            for i in range(len(X)):
                if self.classify(self.X[i],self.y[i]):
                    continue
                else: #判断错误时
                    num_misclassified += 1
                    self.update_term(self.X[i],self.y[i])
            if num_misclassified == 0:
                break
            num_iter += 1

if __name__ == '__main__':
    X = [[3, 3], [4, 3],[1, 1]]
    y = [1, 1,-1]

    percepton = Percepton(X = [(3, 3), (4, 3),(1, 1)],y = [1, 1,-1],lam=1)
    percepton.train()
    print(percepton.w) # [1,1]
    print(percepton.b) # -3
    # 最后训练出的模型为 y=x1+x2-3