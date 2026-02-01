Introduction: implementation of RoboCup Soccer Simulator

The development on robots may be severely limited by the constrained resources.
This is especially true in the research of multi-robot systems in areas such as
RoboCup. Using simulation for algorithm development and testing makes thing
easier.
SimSpark, a multi-robot simulator based on the generic components of the Spark physical multi-agent simulation system, has been used in the RoboCup
Soccer Simulation League since 2004. The project was registered as open source project in SourceForge in 2004, it has an established code base with development increasing year-over-year. 
As the result, RoboCup soccer simulations have changed significantly over the years, going from rather abstract agent representations to more and more realistic humanoid robot games.
Thanks to the flexibility of the Spark system, these transitions were achieved with little changes to the simulator’s core architecture.
In this project we describe the recent development of RoboCup, which make the simulation possible upto 11 vs. 11 humanoid robot soccer games in real time.
After that, we will describe the development of physical robots using Choregraph to prepare the two teams for real playground.

<img width="1087" height="515" alt="Script_Python" src="https://github.com/user-attachments/assets/7a3432f7-6a8f-4b69-bd70-8fc98a1b497b" />

Key Activities and Uses for SimSpark:
•	RoboCup Simulation League Participation: The primary use is developing AI for 3D robotic soccer teams, simulating robot physics, and sensor data.
•	Robotics & AI Research: It is used to simulate physical agents in a 3D environment, allowing for testing control algorithms, machine learning models, and autonomous behaviors.
•	Running the Simulator: SimSpark is available for Linux, allowing users to run the simulation directly without complex installation.
•	Scenario Development: Users can create custom 3D environments and scenarios for multiagent, physical simulations, kinematics simulation and AI (RL) inegration. 
