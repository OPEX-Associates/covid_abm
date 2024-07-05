import mesa
from mesa.datacollection import DataCollector

# Data visualization tools.
import seaborn as sns

# Has multi-dimensional arrays and matrices. Has a large collection of
# mathematical functions to operate on these arrays.
import numpy as np

# Data manipulation and analysis.
import pandas as pd

class CovidAgent(mesa.Agent):
    """An agent with a health state in the COVID-19 model."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.state = "Susceptible"  # Initial state

    def step(self):
        if self.state == "Infected":
            self.spread_infection()

    def spread_infection(self):
        # Infect susceptible neighbors
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        for neighbor in neighbors:
            if neighbor.state == "Susceptible":
                neighbor.state = "Infected"
        # Recover the agent
        self.state = "Recovered"

class CovidModel(mesa.Model):
    def __init__(self, N, width, height):
        self.num_agents = N
        self.grid = mesa.space.MultiGrid(width, height, True)
        self.schedule = mesa.time.RandomActivation(self)

        # Create agents
        for i in range(self.num_agents):
            a = CovidAgent(i, self)
            self.schedule.add(a)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        # Infect one agent at the start
        initial_infected = self.random.choice(self.schedule.agents)
        initial_infected.state = "Infected"

        self.datacollector = DataCollector(
            agent_reporters={"State": "state"}
        )

    def step(self):
        """Advance the model by one step."""
        self.schedule.step()
        self.datacollector.collect(self)
