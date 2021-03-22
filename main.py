import numpy as np
import scipy.special as spy  # for sigmoid func; scipy.expit()

errors = []


class NeuralNet:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # 입력, 은닉, 출력 계층의 노드 개수 설정
        self.iNodes = input_nodes
        self.hNodes = hidden_nodes
        self.oNodes = output_nodes

        # 학습률
        self.lr = learning_rate

        # 가중치
        self.wih = np.random.normal(0.0, pow(self.hNodes, -0.5), (self.hNodes, self.iNodes))
        self.who = np.random.normal(0.0, pow(self.oNodes, -0.5), (self.oNodes, self.hNodes))

        # 시그모이드 함수를 activation func로 설정
        self.activation_func = lambda x: spy.expit(x)
        pass

    def train(self, input_list, target_list):
        # 주어진 학습데이터에 대한 결과를 계산하는 단계
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_func(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_func(final_inputs)

        errors.append(np.mean(abs(final_outputs - targets)))

        # 위 결과값과 실제 값을 비교해 가중치를 update하는 단계
        # 1. 오차를 계산하는 부분
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        # 2. 계산된 오차를 이용해 가중치를 update
        self.who += self.lr * np.dot(output_errors * final_outputs * (1.0 - final_outputs), np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot(hidden_errors * hidden_outputs * (1.0 - hidden_outputs), np.transpose(inputs))

        pass

    def query(self, inputList):
        inputs = np.array(inputList, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_func(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_func(final_inputs)

        return final_outputs
        pass

    pass


input_nodes = 2
hidden_nodes = 7
output_nodes = 1

learning_rate = 0.1

xor = NeuralNet(input_nodes, hidden_nodes, output_nodes, learning_rate)

print(xor.query([0, 1]))
print(xor.query([1, 0]))
print(xor.query([1, 1]))
print(xor.query([0, 0]))

input_list = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
target_list = np.array([[1], [1], [0], [0]])

i = 0
while True:
    i += 1
    xor.train(input_list, target_list)

    if i % 1000 == 0: print("error of ", i, "th iteration is ", errors[i - 1])

    if errors[i - 1] < 0.05:
        break

print(xor.query([0, 1]))
print(xor.query([1, 0]))
print(xor.query([1, 1]))
print(xor.query([0, 0]))
