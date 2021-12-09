import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using MCTrajOpt
using ForwardDiff
using StaticArrays
using LinearAlgebra
using Test
using Random
using BenchmarkTools
using FiniteDiff
include("visualization.jl")

using MathOptInterface
const MOI = MathOptInterface 
const MC = MCTrajOpt

## Build the Double Pendulum
body1 = RigidBody(1.0, Diagonal([0.1, 1.0, 1.0]))
body2 = RigidBody(1.0, Diagonal([0.1, 1.0, 1.0]))
pen = DoublePendulum(body1, body2, gravity=true)

## Build the Robot Arm
arm = MC.RobotArm([body1, body2], [pen.joint0, pen.joint1], gravity=true)

θ = randn(2)
r,q = MC.min2max(arm, θ)
@test MC.min2max(pen, θ) ≈ [r[2]; q[2]; r[3]; q[3]]

x = MC.randstate(pen)
v = @SVector randn(12)
@test MC.kinetic_energy(pen, x, v) ≈ MC.kinetic_energy(arm, x, v)
@test MC.potential_energy(pen, x) ≈ MC.potential_energy(arm, x)

# Discrete Legendre Transforms
ypen = zeros(12)
yarm = zeros(12)
x1 = MC.randstate(pen)
x2 = MC.randstate(pen)
h = 0.01
@test MC.D1Ld!(pen, ypen, x1, x2, h) ≈ MC.D1Ld!(arm, yarm, x1, x2, h)

ypen .= 0
yarm .= 0
@test MC.D2Ld!(pen, ypen, x1, x2, h) ≈ MC.D2Ld!(arm, yarm, x1, x2, h)

# Joint Constraints
cpen = zeros(10)
carm = zeros(10)
@test MC.joint_constraints!(pen, cpen, x) ≈ MC.joint_constraints!(arm, carm, x)

Cpen = zeros(10, 14)
Carm = zeros(10, 14)
@test MC.∇joint_constraints!(pen, Cpen, x) ≈ MC.∇joint_constraints!(arm, Carm, x)

λ = @SVector randn(10)
ypen .= 0
yarm .= 0
@test MC.jtvp_joint_constraints!(pen, ypen, x, λ) ≈ MC.jtvp_joint_constraints!(arm, yarm, x, λ)

# Wrenches
ξpen = zeros(12)
ξarm = zeros(12)
u = @SVector randn(2)
@test MC.getwrenches!(pen, ξpen, x, u) ≈ MC.getwrenches!(arm, ξarm, x, u)

Wpen = zeros(12, 16)
Warm = zeros(12, 16)
@test MC.∇getwrenches!(pen, Wpen, x, u) ≈ MC.∇getwrenches!(arm, Warm, x, u)

# Discrete Euler Lagrange
ypen .= 0
yarm .= 0
x3 = MC.randstate(pen)
u1 = @SVector randn(2)
u2 = @SVector randn(2)
@test MC.DEL!(pen, ypen, x1, x2, x3, λ, u1, u2, h) ≈ 
    MC.DEL!(arm, yarm, x1, x2, x3, λ, u1, u2, h)

Jpen = zeros(12, 14*3 + 2 * 3 + 10)
Jarm = zeros(12, 14*3 + 2 * 3 + 10)
@test MC.∇DEL!(pen, Jpen, x1, x2, x3, λ, u1, u2, h) ≈ 
    MC.∇DEL!(arm, Jarm, x1, x2, x3, λ, u1, u2, h)
