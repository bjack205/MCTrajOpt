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


## MOI
pen = DoublePendulum(body1, body2, gravity=false)
arm = MC.RobotArm([body1, body2], [pen.joint0, pen.joint1], gravity=false)

# Generate target trajectory
opt = SimParams(1.0, 0.05)
opt.N
control(t) = SA[0.5 * (t > 0.5), cos(pi*t)*2]
U = control.(opt.thist)
x0 = MC.min2max(pen, [0.0,0])
Xref = MC.simulate(pen, opt, U, x0)
@test MC.simulate(arm, opt, U, x0) ≈ Xref

# Goal position
xgoal = MC.min2max(pen, [-deg2rad(20), deg2rad(40)])
r_2 = MC.gettran(pen, xgoal)[2]
q_2 = MC.getquat(pen, xgoal)[2]
p_ee = SA[0,0,0.5]
r_ee = r_2 + MC.Amat(q_2)*p_ee

# Set up the problems
Qr = Diagonal(SA_F64[1,1,1.])
Qq = Diagonal(SA_F64[1,1,1,1.])
R = Diagonal(SA_F64[1e-3, 1e-3])
ppen = MC.DoublePendulumMOI(pen, opt, Qr, Qq, R, x0, Xref, rf=r_ee, p_ee=p_ee)
parm = MC.ArmMOI(arm, opt, Qr, Qq, R, x0, Xref, rf=r_ee, p_ee=p_ee)

# Test the MOI functions
ztest = MC.randtraj(parm)

# Objective
gpen = zeros(ppen.n_nlp)
garm = zeros(parm.n_nlp)
@test MOI.eval_objective(ppen, ztest) ≈ MOI.eval_objective(parm, ztest)
MOI.eval_objective_gradient(ppen, gpen, ztest)
MOI.eval_objective_gradient(parm, garm, ztest)
@test gpen ≈ garm

# Constraints
cpen = zeros(ppen.m_nlp)
carm = zeros(parm.m_nlp)
MOI.eval_constraint(ppen, cpen, ztest)
MOI.eval_constraint(parm, carm, ztest)
@test cpen ≈ carm

# Constraint Jacobian
rcpen = MOI.jacobian_structure(ppen)
rcarm = MOI.jacobian_structure(parm)
@test [idx[1] for idx in rcpen] ≈ [idx[1] for idx in rcarm]
@test [idx[2] for idx in rcpen] ≈ [idx[2] for idx in rcarm]
row = [idx[1] for idx in rcarm]
col = [idx[2] for idx in rcarm]
Jpen = zeros(length(rcpen))
Jarm = zeros(length(rcarm))

MOI.eval_constraint_jacobian(ppen, Jpen, ztest)
MOI.eval_constraint_jacobian(parm, Jarm, ztest)
@test Jpen ≈ Jarm

# Solve with Ipopt 
z0 = zeros(parm.n_nlp)
U0 = [SA[0.0,0.0] for k = 1:parm.N-1]
λ0 = [@SVector zeros(parm.p) for k = 1:parm.N-1]
for k = 1:parm.N
    z0[parm.xinds[k]] = Xref[k]
    if k < parm.N
        z0[parm.uinds[k]] = U0[k]
        z0[parm.λinds[k]] = λ0[k] 
    end
end
zpen, = MC.ipopt_solve(ppen, z0, tol=1e-4, goal_tol=1e-6)
zarm, = MC.ipopt_solve(parm, z0, tol=1e-4, goal_tol=1e-6)
@test zpen ≈ zarm