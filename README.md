# Artificial Bee Colony algorithm to solve the Traveling Salesman Problem written in Python.

## Bee Algorithm

### General Principles

In the Artificial Bee Colony (ABC) algorithm, bees are categorized into three groups: worker bees, bee observers, and scout bees. The bees collaborate to optimize solutions in a manner inspired by the foraging behavior of real bees:

https://en.wikipedia.org/wiki/Bees_algorithm#:~:text=The%20bees%20algorithm%20mimics%20the,to%20search%20the%20solution%20space.

### Bee Roles

1. **Worker Bees:**
   - **Responsibility:** Visiting forage sources and collecting information on location and quality.
   - **Memory:** Worker bees have memory, enabling them to recall previously visited locations.
   - **Local Search:** Perform local searches using their knowledge of nearby food source locations.
   
2. **Observer Bees:**
   - **Responsibility:** Wait on the "dance floor" to decide the best food source based on information provided by employed bees.
   - **Decision-Making:** Evaluate the information from worker bees for global optimization.
   
3. **Scout Bees:**
   - **Responsibility:** Conduct random searches to discover new areas.
   - **Random Nature:** Completely random search operations to explore uncharted territories.
   - **Avoid Local Minima:** Scouts help in avoiding convergence to local minima by exploring diverse areas.

### Colony Structure

- The colony is divided into two halves:
  - The first half comprises worker bees.
  - The second half consists of observer bees.
- The total number of bees equals the number of food sources around the hive.

### Bee Transitions

- A worker bee whose food source is depleted transforms into a scout bee.
- The scout bee's role is to explore new areas not covered by employed bees.

### Solution Representation

- In the ABC algorithm, each food source's position represents a solution to the optimization problem.
- Each solution is associated with a fitness value, guiding the decision-making process.
  
### Population Size

- The number of worker bees or observer bees is equivalent to the number of solutions in the population.

This structured approach to optimization, inspired by the intricate behaviors of bees, allows the ABC algorithm to efficiently explore and exploit the search space for finding optimal solutions.

## Algorithm Steps

Assuming the problem's solution space is D-dimensional, with S bees foraging and observing, and S honey sources, the standard Artificial Bee Colony (ABC) algorithm navigates the optimization problem in a D-dimensional search space.

1. **Initialization:**
   - Set the number of foraging and observing bees equal to S.
   - Define the number of honey sources equal to the number of variables (D) in the problem.

2. **Search Space Representation:**
   - Consider the optimization problem as a search in a D-dimensional space.
   - Each honey source's position represents a potential solution, and the nectar amount signifies the solution's fitness.

3. **Bee Movement:**
   - Each bee corresponding to the i-th honey source explores a new solution using the formula:
   
     $X_{id}' = X_{id} + \Phi_{id} \cdot (X_{kd} - X_{id})$
     where $i = 1, 2, \ldots, S$ denotes honey sources,  $D = 1, 2, \ldots, D$ represents the number of variables, and $\Phi_{id}$ is a random number between [-1, 1].
   - Compare newly generated solutions ${X_{i1}', X_{i2}', \ldots, X_{iD}'}$ with original solutions ${X_{i1}, X_{i2}, \ldots, X_{iD}}$ and retain the best solution using a greedy strategy.

4. **Probability Calculation:**
   - Calculate the acceptance probability for each foraging bee using the formula mentioned above:
$\text{prob} = \frac{0.9 \cdot \text{fit}(i)}{\max(\text{fit})} + 1$
   - Observer bees accept foraging bees based on the calculated probability.
   - Update foraging bees using the formula and make greedy decisions.

5. **Boundary Control:**
   - If the fitness of a honey source does not increase within a defined step (controlled by the "boundary" parameter) during the entire search process, researchers (bees) search for new solutions.

6. **Researcher Bee Exploration:**
   - Researchers search for new solutions using the formula:
     $X_{id}' = X_{id} + r \cdot (X_{\text{max}} - X_{\text{min}})$
     where $r$ is a random number between [0, 1], and $X_{\text{min}}$ and $X_{\text{max}}$ are the lower and upper limits of the d variable space.

This structured approach allows the ABC algorithm to iteratively explore and update solutions in the search space, balancing exploration and exploitation for efficient optimization.

## Model TSP

https://en.wikipedia.org/wiki/Travelling_salesman_problem

The problem with TSP is starting from one point and then going back to the starting point, asking for the entire shortest route. Specific the mathematical formula looks like this:

![image.png](attachment:afedd292-4f8a-42da-9383-f5097e155e5e.png)

## Summary

The algorithm of the artificial bee colony is mainly divided into three stages: honey collection, observation and reconnaissance.

The whole principle is very similar to the principle of the genetic algorithm. The collection of bees is equivalent to the initialization of the parental chromosome. Watching the bees is equivalent to the offspring after choosing a roulette wheel. Scout bees cannot find the best solution in the limited time. The solution is then randomly initialized.

![image.png](attachment:529ef86d-ede2-4513-a7a8-44caa677100b.png)
