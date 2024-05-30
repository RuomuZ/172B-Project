import matplotlib.pyplot as plt

# Initialize lists to store the metrics
losses = []
ious = []
val_losses = []

# Open the log file and read the lines
with open('trainingvitDeep_3ch.log', 'r') as f:
    lines = f.readlines()

# Parse the lines and extract the metrics
for line in lines:
    parts = line.split(',')
    losses.append(float(parts[1].split(': ')[1]))
    val_losses.append(float(parts[2].split(': ')[1]))
    ious.append(float(parts[3].split(': ')[1]))

# Create a figure and axes
fig, ax1 = plt.subplots()

# Plot the loss and validation loss
ax1.plot(losses, label='Loss', color='blue')
ax1.plot(val_losses, label='Val Loss', color='green')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')

# Create a second y-axis to plot the IoU
ax2 = ax1.twinx()
ax2.plot(ious, label='IoU', color='red')
ax2.set_ylabel('IoU')

# Add a legend
fig.legend(loc='upper right')

# Save the plot
plt.savefig('trainingvitDeep_3ch.png')


# Show the plot
plt.show()