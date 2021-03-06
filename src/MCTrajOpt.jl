module MCTrajOpt

using StaticArrays
using ForwardDiff
using FiniteDiff
using LinearAlgebra

export RigidBody, SimParams
export mass_matrix,
       kinematics, inv_kinematics, D1K, D2K, D1Kinv, D2Kinv,
       min2max,
       Lagrangian, D1L, D2L,
    #    Lagrangian_dot, D1L_dot, D2L_dot,
       Ld, D1Ld, D2Ld,
       DEL, D3_DEL,
       simulate,
       DENSITIES

export TwoBody, RevoluteJoint, SpaceBar, DoublePendulum
export packZ!, randstate, visualize!, launchvis


include("ipopt_helpers.jl")
include("sparse.jl")
include("joints.jl")
include("rotations.jl")
include("two_body.jl")
include("double_pendulum.jl")
include("minimal_coordinates.jl")
include("dp_ipopt.jl")

include("visualization/bodies.jl")

include("arm.jl")
include("arm_ipopt.jl")

include("visualization/visualization.jl")

end # module

