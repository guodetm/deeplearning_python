


# 乘法层
class MulLayer:
    def __init__(self):
        self.x = None  # 正向传播的初始值x
        self.y = None  # 正向传播的初始值y

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y  # 正向传播的输出值out
        return out

    def backward(self, dout):
        # dout为上游传过来的导数
        dx = dout * self.y  # 翻转值
        dy = dout * self.x  # 翻转值
        return dx, dy


# 加法层
class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy



