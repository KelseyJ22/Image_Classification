import matplotlib.pyplot as plt

accuracies = [0.1, 0.69999999, 0.68000001, 0.89999998, 0.75999999, 0.88, 0.88, 0.83999997, 0.81999999, 0.88, 0.86000001, 0.86000001]
iterations = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]


plt.plot(iterations, accuracies)
plt.title('Accuracy over iterations on train data')
plt.show()