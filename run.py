from covid_model import CovidModel
import matplotlib.pyplot as plt
import seaborn as sns

model = CovidModel(100, 10, 10)

for i in range(100):
    model.step()

# Convert data to a DataFrame
data = model.datacollector.get_agent_vars_dataframe()

# Display the first few rows
print(data.head())



# Count the number of agents in each state at each step
state_counts = data.reset_index().groupby(["Step", "State"]).size().unstack().fillna(0)

# Plot the data
state_counts.plot()
plt.xlabel("Step")
plt.ylabel("Number of Agents")
plt.title("COVID-19 Propagation")
plt.show()