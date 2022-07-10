import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import csv

training_losses = []
validation_losses = []

with open('results/losses.csv', 'r') as file:
    lines = csv.reader(file)
    for row in lines:
        if row[0] == 'training':
            training_losses.append(abs(int(row[1])))
        elif row[0] == 'validation':
            validation_losses.append(abs(int(row[1])))

fig, ax = plt.subplots()
ax.plot(range(len(training_losses)), training_losses, color="red", marker="o")
ax.set_xlabel("Epoch", fontsize=14)
ax.set_ylim([0, 5000])
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_ylabel("Training Loss / batch", color="red", fontsize=14)
ax2 = ax.twinx()
ax2.plot(range(len(validation_losses)), validation_losses, color="blue", marker="o")
ax2.set_ylabel("Validation Loss / batch", color="blue", fontsize=14)
ax2.set_ylim([0, 10000000000])
ax2.ticklabel_format(useOffset=False, style='plain')
plt.show()
fig.savefig('results/lossesOverTime.jpg', format='jpeg', dpi=100, bbox_inches='tight')
