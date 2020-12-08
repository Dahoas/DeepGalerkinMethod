import matplotlib.pyplot as plt

def plot_loss(losses,filename):
	plt.clf()
	plt.plot(losses)
	plt.savefig(filename)