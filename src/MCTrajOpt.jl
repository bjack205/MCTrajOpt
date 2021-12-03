module MCTrajOpt

using StaticArrays
using ForwardDiff
using LinearAlgebra

export RigidBody, SimParams
export mass_matrix,
       kinematics, inv_kinematics, D1K, D2K, D1Kinv, D2Kinv,
       Lagrangian_vel, D1L_vel, D2L_vel,
       Lagrangian_dot, D1L_dot, D2L_dot,
       Ld, D1Ld, D2Ld,
       DEL, D3_DEL,
       simulate

export TwoBody, RevoluteJoint
export packZ!

include("rotations.jl")
include("rigid_body.jl")
include("two_body.jl")
include("rigid_body_ipopt.jl")
include("point_mass_ipopt.jl")
include("twobody_ipopt.jl")

end # module
