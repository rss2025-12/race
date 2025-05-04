import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
filename = 'crosstrack_two'
df = pd.read_csv(f'../{filename}.csv')
df['timestamp'] = df['timestamp'] - df['timestamp'].iloc[0]

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(df['timestamp'], df['cross_track_error'], label='Cross-Track Error')
plt.xlabel('Time (s)')
plt.ylabel('CTE (m)')
plt.title('Cross-Track Error Over Time (2 m/s)') # Remember to change this
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(f'{filename}.png')
plt.show()
