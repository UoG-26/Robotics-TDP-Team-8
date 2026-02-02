# Note1
This is a 4v4 demostration based on FCP, please run

cd ~/"Your Dir Path"
source ~/.venvs/fcp/bin/activate
export PYTHONPATH=.
chmod +x start_two_teams_roles_4v4.sh
./start_two_teams_roles_4v4.sh localhost 3100 Team8 Team_robots

# Note2
By the way, if you want to change the Team name, please modify this part "Team8" "Team_robots" in (./start_two_teams_roles_4v4.sh localhost 3100 Team8 Team_robots)
And "localhost 3100" means I set the port:3100 as Server Communication port, 3200 as Monitor port using Roboviz

# Note3
Finally, there're a lot of details I don't show in my script. Should you have any question, please feel free to contact me or submit the Issue on the top.
