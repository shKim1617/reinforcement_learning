from drone_class import create_drones, FollowerDrone, LeaderDrone

leader = LeaderDrone()
follow = FollowerDrone(drone_id=1, init_direction=[-20, 20], offset=[-20, 20])

print(leader.get_state())
print(follow.get_state())
print()

leader.move([-10, 0])
follow.update_all(leader_position=leader.position, leader_direction=leader.direction)

print(leader.get_state())
print(follow.get_state())
print()

follow.move(follow.target_position)

print(follow.get_state())
print()

follow.set_leader_direction()

print(follow.get_state())
print()

follow