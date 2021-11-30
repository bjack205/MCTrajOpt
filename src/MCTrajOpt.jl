module MCTrajOpt

using StaticArrays
using ForwardDiff
using LinearAlgebra

export RigidBody, SimParams
export kinematics, inv_kinematics, D1K, D2K, D1Kinv, D2Kinv,
       Lagrangian_vel, D1L_vel, D2L_vel,
       Lagrangian_dot, D1L_dot, D2L_dot,
       Ld, D1Ld, D2Ld,
       DEL, D3_DEL,
       simulate

include("rotations.jl")
include("rigid_body.jl")

end # module
