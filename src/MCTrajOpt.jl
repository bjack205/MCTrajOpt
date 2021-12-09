module MCTrajOpt

using StaticArrays
using ForwardDiff
using LinearAlgebra

export RigidBody, SimParams
export mass_matrix,
       kinematics, inv_kinematics, D1K, D2K, D1Kinv, D2Kinv,
       min2max,
       Lagrangian, D1L, D2L,
    #    Lagrangian_dot, D1L_dot, D2L_dot,
       Ld, D1Ld, D2Ld,
       DEL, D3_DEL,
       simulate

export TwoBody, RevoluteJoint, SpaceBar, DoublePendulum
export packZ!, randstate

include("sparse.jl")
include("joints.jl")
include("rotations.jl")
include("rigid_body.jl")
include("two_body.jl")
include("rigid_body_ipopt.jl")
include("point_mass_ipopt.jl")
include("twobody_ipopt.jl")
include("double_pendulum.jl")
include("dp_ipopt.jl")

include("arm.jl")
include("arm_ipopt.jl")

include("visualization/bodies.jl")

end # module

