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

using MathOptInterface
const MOI = MathOptInterface 
const MC = MCTrajOpt

## Generate the model
body1 = RigidBody(1.0, Diagonal([0.1, 1.0, 1.0]))
body2 = RigidBody(1.0, Diagonal([0.1, 1.0, 1.0]))
model = DoublePendulum(body1, body2, gravity = true, acrobot=true)
opt = SimParams(2.0, 0.05)

# Goal position
x0 = MC.min2max(model, [-pi, 0]) 
xgoal = MC.min2max(model, [-deg2rad(00), deg2rad(00)])
r_2 = MC.gettran(model, xgoal)[2]
q_2 = MC.getquat(model, xgoal)[2]
p_ee = SA[0,0,0.5]
r_ee = r_2 + MC.Amat(q_2)*p_ee

# Reference trajectory is just the goal position
Xref = [copy(xgoal) for k = 1:opt.N]

# Set up the problems
Qr = Diagonal(SA_F64[1,1,1.])
Qq = Diagonal(SA_F64[1,1,1,1.])
R = Diagonal(SA_F64[1e-3])
prob = MC.DoublePendulumMOI(model, opt, Qr, Qq, R, x0, Xref, rf=r_ee, p_ee=p_ee)

# Create initial guess
z0 = zeros(prob.n_nlp)
U0 = [SA[randn()] for k = 1:prob.N-1]
λ0 = [@SVector zeros(prob.p) for k = 1:prob.N-1]
for k = 1:prob.N
    z0[prob.xinds[k]] = x0 
    if k < prob.N
        z0[prob.uinds[k]] = U0[k]
        z0[prob.λinds[k]] = λ0[k] 
    end
end

zsol, = MC.ipopt_solve(prob, z0, tol=1e-4, goal_tol=1e-6)
Xsol = [zsol[xi] for xi in prob.xinds]
Usol = [zsol[ui] for ui in prob.uinds]
λsol = [zsol[λi] for λi in prob.λinds]
if isdefined(Main, :vis)
    visualize!(vis, model, Xsol, opt)
end

## Test Jacobian
ztest = MC.randtraj(prob)

rc = MOI.jacobian_structure(prob)
row = [idx[1] for idx in rc]
col = [idx[2] for idx in rc]
jac = zeros(length(rc)) 
jac0 = zeros(prob.m_nlp, prob.n_nlp)
c = zeros(prob.m_nlp)

MOI.eval_constraint_jacobian(prob, jac, ztest)
FiniteDiff.finite_difference_jacobian!(jac0, (c,x)->MOI.eval_constraint(prob, c, x), ztest)
@test sparse(row, col, jac, prob.m_nlp, prob.n_nlp) ≈ jac0

## Visualizer
if !isdefined(Main, :vis)
    vis = launchvis(model, x0)
end
Xsol = [zsol[xi] for xi in prob.xinds]
Usol = [zsol[ui] for ui in prob.uinds]
visualize!(vis, model, Xsol, opt)

Xsim,λsim = simulate(model, opt, Usol, Xsol[1])
visualize!(vis, model, Xsim, opt)
norm(λsim - λsol)
plot(Usol)

model.joint0.p2
