from agent.Base_Agent import Base_Agent as Agent


# Args: Server IP, Agent Port, Monitor Port, Uniform No., Robot Type, Team Name

player = Agent("localhost", 3100, 3200, 7, 1, "TeamName")



w = player.world


while True:

    player.behavior.execute("Walk", w.ball_abs_pos[:2], True, None, True, None)

    player.scom.commit_and_send( w.robot.get_command() )

    player.scom.receive()