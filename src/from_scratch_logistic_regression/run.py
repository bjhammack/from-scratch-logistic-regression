import matplotlib.pyplot as plt
from model import Model
import numpy as np


model = Model()

model_results = model.model(
    model.X,
    model.Y,
    iterations=2000,
    learning_rate=0.05,
    verbose=True
    )

costs = np.squeeze(model_results['costs'])
plt.plot(costs)
plt.title(
    f'Params: Iter - {model_results["iterations"]}; '
    f'LR - {model_results["learning_rate"]}'
    )
plt.ylabel('Cost')
plt.xlabel('Iteration/100')
plt.show()