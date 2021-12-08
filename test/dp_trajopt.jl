import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using MCTrajOpt
using ForwardDiff
using StaticArrays
using LinearAlgebra
using Test
using Random
using BenchmarkTools
using FiniteDiff
using SparseArrays
using Plots
include("visualization.jl")

using MathOptInterface
const MOI = MathOptInterface 
const MC = MCTrajOpt
##

#############################################
# Trajectory Optimization
#############################################

# Generate the model
body1 = RigidBody(1.0, Diagonal([0.1, 1.0, 1.0]))
body2 = RigidBody(1.0, Diagonal([0.1, 1.0, 1.0]))
model = DoublePendulum(body1, body2, gravity = false)

# Generate target trajectory
opt = SimParams(1.0, 0.05)
opt.N
control(t) = SA[0.5 * (t > 0.5), cos(pi*t)*2]
U = control.(opt.thist)
x0 = MC.min2max(model, [0.0,0])
Xref = MC.simulate(model, opt, U, x0)

# Visualizer
if !isdefined(Main, :vis)
    vis = launchvis(model, x0)
end
visualize!(vis, model, Xref, opt)

# Goal position
xgoal = MC.min2max(model, [-deg2rad(20), deg2rad(40)])
visualize!(vis, model, xgoal)
r_2 = MC.gettran(model, xgoal)[2]
q_2 = MC.getquat(model, xgoal)[2]
p_ee = SA[0,0,0.5]
r_ee = r_2 + MC.Amat(q_2)*p_ee

## Set up the problems
Qr = Diagonal(SA_F64[1,1,1.])
Qq = Diagonal(SA_F64[1,1,1,1.])
R = Diagonal(SA_F64[1e-3, 1e-3])
prob = MC.DoublePendulumMOI(model, opt, Qr, Qq, R, x0, Xref, rf=r_ee, p_ee=p_ee)

# Create initial guess
z0 = zeros(prob.n_nlp)
U0 = [SA[0.0,0.0] for k = 1:prob.N-1]
λ0 = [@SVector zeros(prob.p) for k = 1:prob.N-1]
for k = 1:prob.N
    z0[prob.xinds[k]] = Xref[k]
    if k < prob.N
        z0[prob.uinds[k]] = U0[k]
        z0[prob.λinds[k]] = λ0[k] 
    end
end

# Test functions
grad_f = zeros(prob.n_nlp)
c = zeros(prob.m_nlp)
jac0 = zeros(prob.m_nlp, prob.n_nlp)
rc = MOI.jacobian_structure(prob)
row = [idx[1] for idx in rc]
col = [idx[2] for idx in rc]
jac = zeros(length(rc)) 
jac .= 1
jac_struct = sparse(row, col, jac)
jac_struct

ztest = zero(z0)
Xtest = [MC.randstate(model) for k = 1:prob.N]
Utest = [@SVector randn(2) for k = 1:prob.N-1] .* 0
λtest = [@SVector randn(prob.p) for k = 1:prob.N-1]
for k = 1:prob.N
    ztest[prob.xinds[k]] = Xtest[k]
    if k < prob.N
        ztest[prob.uinds[k]] = Utest[k]
        ztest[prob.λinds[k]] = λtest[k] 
    end
end

J = MOI.eval_objective(prob, ztest)
MOI.eval_objective_gradient(prob, grad_f, z0)
MOI.eval_constraint(prob, c, ztest)

jac .= 0
MOI.eval_constraint_jacobian(prob, jac, ztest)
FiniteDiff.finite_difference_jacobian!(jac0, (c,x)->MOI.eval_constraint(prob, c, x), ztest)
@test sparse(row, col, jac, prob.m_nlp, prob.n_nlp) ≈ jac0
length(row) / (prob.m_nlp * prob.n_nlp)
prob.m_nlp
# @test c[end-2:end] ≈ zeros(3)

zsol, = MC.ipopt_solve(prob, z0, tol=1e-4, goal_tol=1e-6)
Xsol = [zsol[xi] for xi in prob.xinds]
Usol = [zsol[ui] for ui in prob.uinds]

visualize!(vis, model, Xsol, opt)
norm(map(Xsol) do x
    MC.joint_constraints(model, x)
end)

#############################################
# Swing-up
#############################################

# Model
model = DoublePendulum(body1, body2, gravity = true)

# Sim params
opt = SimParams(1.0, 0.05) 
opt.N

# Generate the reference trajectory
x0 = MC.min2max(model, [-pi,0])
xf = MC.min2max(model, [0,0]) 
Xref = map(1:opt.N) do k
    t = (k-1) / (opt.N-1)
    θ1 = -pi + pi * t
    θ2 = 0 
    x0 = MC.min2max(model, SA[θ1, θ2])
end
visualize!(vis, model, x0)
visualize!(vis, model, Xref, opt)

#
Qr = Diagonal(SA_F64[1,1,1.])
Qq = Diagonal(SA_F64[1,1,1,1.])
R = Diagonal(SA_F64[1e-3, 1e-3])
r_ee = SA[0,0,2.0]

prob = MC.DoublePendulumMOI(model, opt, Qr, Qq, R, x0, Xref, rf=r_ee)
prob.goalcon

# Create initial guess
z0 = zeros(prob.n_nlp)
U0 = [SA[0.0,0.0] for k = 1:prob.N-1]
λ0 = [@SVector zeros(prob.p) for k = 1:prob.N-1]
for k = 1:prob.N
    z0[prob.xinds[k]] = Xref[k]
    if k < prob.N
        z0[prob.uinds[k]] = U0[k]
        z0[prob.λinds[k]] = λ0[k] 
    end
end

zsol, = MC.ipopt_solve(prob, z0, tol=1e-4, goal_tol=1e-6)
Xsol = [zsol[xi] for xi in prob.xinds]
Usol = [zsol[ui] for ui in prob.uinds]
visualize!(vis, model, Xsol, opt)

uhist = hcat(Vector.(Usol)...)
plot(uhist')
