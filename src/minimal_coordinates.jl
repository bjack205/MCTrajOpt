import RigidBodyDynamics
const RBD = RigidBodyDynamics 

function builddoublependulum(bodies, joints; gravity=9.81)
    g=SA[0,0,-gravity]
    world = RBD.RigidBody{Float64}("world")
    dp = RBD.Mechanism(world; gravity = g)

    link1_com = RBD.CartesianFrame3D("link1")
    inertia1 = RBD.SpatialInertia(link1_com,
        moment_about_com=bodies[1].J,
        mass=bodies[2].mass,
        com=SA[0,0,0.0]
    )
    link1 = RBD.RigidBody(inertia1)

    shoulder = RBD.Joint("shoulder", RBD.Revolute(joints[1].axis))
    joint_pose = one(RBD.Transform3D, 
        RBD.frame_before(shoulder), RBD.default_frame(world)
    )
    successor_pose = RBD.Transform3D(
        link1_com, RBD.frame_after(shoulder), -joints[1].p2
    )
    RBD.attach!(dp, world, link1, shoulder, 
        joint_pose=joint_pose, successor_pose=successor_pose
    )
    state = RBD.MechanismState(dp)
    RBD.transform(state, RBD.Point3D(link1_com, SA[0,0,0]), RBD.default_frame(world))

    # Create 2nd link
    link2_com = RBD.CartesianFrame3D("link2")
    inertia2 = RBD.SpatialInertia(link2_com,
        moment_about_com=bodies[2].J,
        mass=bodies[2].mass,
        com=SA[0,0,0.0]
    )
    link2 = RBD.RigidBody(inertia2)

    # Add elbow joint
    elbow = RBD.Joint("elbow", RBD.Revolute(joints[2].axis))
    elbow_to_link1 = RBD.Transform3D(
        RBD.frame_before(elbow), RBD.frame_after(shoulder), joints[2].p1*2
    )
    link2_to_elbow = RBD.Transform3D(
        link2_com, RBD.frame_after(elbow), -joints[2].p2
    )
    RBD.attach!(dp, link1, link2, elbow, joint_pose=elbow_to_link1, successor_pose=link2_to_elbow)
    return dp
end