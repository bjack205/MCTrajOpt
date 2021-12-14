import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using MCTrajOpt
using MeshCatMechanisms
import RigidBodyDynamics
using ForwardDiff
using StaticArrays
using LinearAlgebra
using Rotations
using Test
using Random
using BenchmarkTools
using FiniteDiff
using SparseArrays
using Plots

using MathOptInterface
const MOI = MathOptInterface 
const MC = MCTrajOpt
const RBD = RigidBodyDynamics

## Generate the model
body1 = RigidBody(1.0, Diagonal([1.0, 1.0, 1.0]*1e-2))
body2 = RigidBody(1.0, Diagonal([1.0, 1.0, 1.0]*1e-2))
model = DoublePendulum(body1, body2, gravity = false)
bodies = [body1, body2]
joints = [model.joint0, model.joint1]

##
dp = MC.builddoublependulum(bodies, joints, gravity=model.gravity)

##
state = RBD.MechanismState(dp)
RBD.set_configuration!(state, deg2rad.([0,90]))
RBD.set_velocity!(state, [0,0])

# Test kinematics
function getlinkcom(state,j)
    body = RBD.findbody(state.mechanism, "link$j")
    com = RBD.center_of_mass(RBD.spatial_inertia(body))
    com_world = RBD.transform(state, com, RBD.root_frame(state.mechanism)) 
    T = RBD.transform_to_root(state, RBD.default_frame(body))
    RBD.Transform3D(com.frame, com_world.frame, RBD.rotation(T), com_world.v)
end

θ = randn(2)
RBD.set_configuration!(state, θ) 
T1 = getlinkcom(state, 1)
T2 = getlinkcom(state, 2)
x = MC.min2max(model, θ)

MC.gettran(model, x, 1) ≈ RBD.translation(T1) 
UnitQuaternion(MC.getquat(model, x, 1)) ≈ UnitQuaternion(RBD.rotation(T1))
MC.gettran(model, x, 2) ≈ RBD.translation(T2)
UnitQuaternion(MC.getquat(model, x, 2)) ≈ UnitQuaternion(RBD.rotation(T2))
