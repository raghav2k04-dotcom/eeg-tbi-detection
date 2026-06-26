import matplotlib.pyplot as plt

labels = ["Control", "TBI"]
values = [4150, 4400]

plt.bar(labels, values)

plt.title("Prediction Distribution")
plt.ylabel("Number of Samples")

plt.show()
