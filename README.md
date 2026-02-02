
<img width="396" height="366" alt="11" src="https://github.com/user-attachments/assets/9af5bbad-de34-443f-b64c-054a59a837d8" />

## Introduction: 
implementation of RoboCup Soccer Simulator

The development on robots may be severely limited by the constrained resources.
This is especially true in the research of multi-robot systems in areas such as
RoboCup. Using simulation for algorithm development and testing makes thing
easier.
**SimSpark**, a multi-robot simulator based on the generic components of the Spark physical multi-agent simulation system, has been used in the RoboCup
Soccer Simulation League since 2004. The project was registered as open source project in SourceForge in 2004, it has an established code base with development increasing year-over-year. 
As the result, RoboCup soccer simulations have changed significantly over the years, going from rather abstract agent representations to more and more realistic humanoid robot games.
Thanks to the flexibility of the Spark system, these transitions were achieved with little changes to the simulator’s core architecture.
In this project we describe the recent development of RoboCup, which make the simulation possible upto 11 vs. 11 humanoid robot soccer games in real time.
After that, we will describe the development of physical robots using **Choregraph** to prepare the two teams for real playground.


<img width="1180" height="669" alt="image" src="https://github.com/user-attachments/assets/e668bda1-1f0f-46c7-9528-75b1b2104920" />

## Physical Hardware Simulation:

For testing the physical Humanoid Robots, we have to use the python script functions to Choregraph as a linking process for creating physical and well tested movements in a given workspace.
##SimSpark allow to develop major required motions, team communication, path finding as well as enforce AI by leveraging Reinforcement Learning for Working Robots. 

Choregraphe is a desktop application from SoftBank Robotics (Aldebaran) designed for programming, controlling, and simulating NAO and Pepper robots without necessarily needing to write code. 
## What we can do with Choregraphe:
### Create Robot Behaviors & Animations 
- Drag-and-Drop Programming: Use pre-defined boxes from the library (speech, movement, interaction) to create complex behaviors.
- Animation Creation: Use the Timeline editor to create, edit, and play animations for the robot’s joints.
- Dialog & Interaction: Create engaging interactions, such as dance routines, storytelling, or voice-controlled responses.
- Python Integration: Enhance or create your own custom boxes using Python code for more complex logic. 
### Test and Simulate:
- Virtual Robot: Test your code on a simulated robot within Choregraphe before deploying to a physical robot.
- Simulation Connections: Connect to external simulators like Webots to test navigation and, more advanced, AI interaction. 
### Monitor and Control Real Robots:
- Live Control: Connect to a real NAO or Pepper via IP address to monitor its sensors, camera view, and battery status in real-time.
- Stiffen Motors: Use the "Wake Up" feature to provide power to the motors to enable movement.
- System Maintenance: Update the robot's operating system (NAOqi) and factory reset the device. 


## Typical First Steps (Hello World)...
- Start Choregraphe and click "Connect to" to select your robot.
- Drag and drop a "Say" box from the dialogue library onto the flow diagram.
- Link the input to the box, type your message, and press Play to hear the robot speak. 



<img width="172" height="380" alt="Code" src="https://github.com/user-attachments/assets/4335e22e-840a-4d54-bc9c-ec19b3b5815e" />

