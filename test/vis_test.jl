import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using MCTrajOpt
using StaticArrays
using MeshCat, GeometryBasics, CoordinateTransformations, Rotations, Colors

using MCTrajOpt: L, R, Amat, CylindricalBody, geometry
const MC = MCTrajOpt

densities = Dict(
    :aluminum => 2.7 * 1000,
    :steel => 8.05 * 1000
)

vis = Visualizer()
open(vis)

##
r_0 = SA[1,0,0.]
q_0 = SA[1,0,0,0.]
q_0 = MC.expm2([0,1,0]*deg2rad(00))

b1 = CylindricalBody(densities[:aluminum], 0.15, 0.5)
joint0 = RevoluteJoint([0,0,0], [0,0,-b1.length/2], [0,0,1])
θ1 = deg2rad(00)
r_1, q_1 = MC.joint_kinematics(joint0, r_0, q_0, θ1)

delete!(vis)
setobject!(vis["body1"], geometry(b1), MeshPhongMaterial(color=colorant"green"))
settransform!(vis["body1"], compose(Translation(r_1), LinearMap(UnitQuaternion(q_1))))

b2 = CylindricalBody(densities[:aluminum], 0.10, 0.25)
joint1 = RevoluteJoint([0,b1.radius,b1.length/2], [0,-b2.radius,-b2.length/2],[0,1,0])
θ2 = deg2rad(00)
r_2, q_2 = MC.joint_kinematics(joint1, r_1, q_1, θ2)

setobject!(vis["body2"], geometry(b2), MeshPhongMaterial(color=colorant"green"))
settransform!(vis["body2"], compose(Translation(r_2), LinearMap(UnitQuaternion(q_2))))

b3 = CylindricalBody(densities[:aluminum], 0.10, 0.25)
joint2 = RevoluteJoint([0,0,b2.length/2], [0,0,-b3.length/2],[0,0,1])
θ3 = deg2rad(00)
r_3, q_3 = MC.joint_kinematics(joint2, r_2, q_2, θ3)

setobject!(vis["body3"], geometry(b3), MeshPhongMaterial(color=colorant"blue"))
settransform!(vis["body3"], compose(Translation(r_3), LinearMap(UnitQuaternion(q_3))))

b4 = CylindricalBody(densities[:aluminum], 0.10, 0.25)
joint3 = RevoluteJoint([0,-b3.radius,b3.length/2], [0,b4.radius,-b4.length/2],[0,1,0])
θ4 = deg2rad(00)
r_4, q_4 = MC.joint_kinematics(joint3, r_3, q_3, θ4)

setobject!(vis["body4"], geometry(b4), MeshPhongMaterial(color=colorant"blue"))
settransform!(vis["body4"], compose(Translation(r_4), LinearMap(UnitQuaternion(q_4))))

b5 = CylindricalBody(densities[:aluminum], 0.10, 0.25)
joint4 = RevoluteJoint([0,0,b4.length/2], [0,0,-b5.length/2],[0,0,1])
θ5 = deg2rad(00)
r_5, q_5 = MC.joint_kinematics(joint4, r_4, q_4, θ5)

setobject!(vis["body5"], geometry(b5), MeshPhongMaterial(color=colorant"orange"))
settransform!(vis["body5"], compose(Translation(r_5), LinearMap(UnitQuaternion(q_5))))

b6 = CylindricalBody(densities[:aluminum], 0.10, 0.10)
joint5 = RevoluteJoint([0,0,b5.length/2], [0,0,-b6.length/2],[0,1,0])
θ6 = deg2rad(00)
r_6, q_6 = MC.joint_kinematics(joint5, r_5, q_5, θ6)

setobject!(vis["body6"], geometry(b6), MeshPhongMaterial(color=colorant"red"))
settransform!(vis["body6"], compose(Translation(r_6), LinearMap(UnitQuaternion(q_6))))