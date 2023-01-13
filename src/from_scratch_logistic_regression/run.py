from model import Model


model = Model()
X = model.X[:, :500]
Y = model.Y[:, :500]
model_results = model.model(X, Y, verbose=True)