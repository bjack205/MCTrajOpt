import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using MCTrajOpt
using ForwardDiff
using StaticArrays
using LinearAlgebra
using Test
using Random
using BenchmarkTools
using FiniteDiff
using Colors
using SparseArrays
using Rotations
using Plots
# include("visualization.jl")

using MCTrajOpt: CylindricalBody
using MathOptInterface
const MOI = MathOptInterface 
const MC = MCTrajOpt

## Build the Arm
geom = [
    CylindricalBody(0.15, 0.50, :aluminum),
    CylindricalBody(0.10, 0.25, :aluminum),
    CylindricalBody(0.10, 0.25, :aluminum, color=colorant"green")

]
joints = [
    RevoluteJoint([0,0,0], [0,0,-geom[1].length/2], [0,0,1]),
    RevoluteJoint([0,geom[1].radius, geom[1].length/2], [0,-geom[2].radius,-geom[2].length/2],[0,1,0]),
    RevoluteJoint([0,0,geom[2].length/2], [0,0,-geom[3].length/2],[0,0,1])
]
arm = MC.RobotArm(geom, joints)

## 
x0 = MC.min2max(arm, zeros(3))
MC.getrind(arm, 3)
if !isdefined(Main, :vis)
    vis = launchvis(arm, x0)
end
visualize!(vis, arm, MC.min2max(arm, deg2rad.([0, 45, 90])))

## Simulate a small trajectory
torques(t) = SA[10.0 * (1-t), -10.0 * (t < 1.5), cos(π*t)]
sim = SimParams(1.0, 0.05)
Xref, λref = simulate(arm, sim, torques.(sim.thist), x0)
visualize!(vis, arm, Xref, sim)

# Goal position
xgoal = MC.min2max(arm, deg2rad.([20, -60, 90]))
# visualize!(vis, arm, xgoal)
r_2 = MC.gettran(arm, xgoal, 3)
q_2 = MC.getquat(arm, xgoal, 3)
p_ee = SA[0,0,geom[3].length/2]
r_ee = r_2 + MC.Amat(q_2)*p_ee

## Set up the problem
Qr = Diagonal(SA_F64[1,1,1.])
Qq = Diagonal(SA_F64[1,1,1,1.])
R = Diagonal(SA_F64[1e-3, 1e-3, 1e-3])
prob = MC.ArmMOI(arm, sim, Qr, Qq, R, x0, Xref, rf=r_ee, p_ee=p_ee)

# Form the reference trajectory
Uref = torques.(sim.thist)
zref = MC.buildtraj(prob, Xref, Uref, λref)

#############################################
# Test Interface
#############################################
ztest = MC.randtraj(prob)
grad = zeros(prob.n_nlp)
MOI.eval_objective(prob, ztest)
MOI.eval_objective(prob, zref)
MOI.eval_objective_gradient(prob, grad, ztest)

c = zeros(prob.m_nlp)
MOI.eval_constraint(prob, c, ztest)

rc = MOI.jacobian_structure(prob)
row = [idx[1] for idx in rc]
col = [idx[2] for idx in rc]
jvec = zeros(length(rc))
MOI.eval_constraint_jacobian(prob, jvec, ztest)

jac = zeros(prob.m_nlp, prob.n_nlp)
FiniteDiff.finite_difference_jacobian!(jac, 
    (c,x)->MOI.eval_constraint(prob, c, x), ztest
)
@test sparse(row, col, jvec) ≈ jac

#############################################
## Solve w/ Ipopt
#############################################
Qr = Diagonal(SA_F64[1,1,1.])
Qq = Diagonal(SA_F64[1,1,1,1.])
R = Diagonal(SA_F64[1e-3, 1e-3, 1e-3]) * 10
prob = MC.ArmMOI(arm, sim, Qr, Qq, R, x0, Xref, rf=r_ee, p_ee=p_ee)

opt = SimParams(1.0, 0.05)
z0 = zeros(prob.n_nlp)
U0 = [@SVector zeros(prob.L) for k = 1:prob.N-1]
λ0 = [@SVector zeros(prob.p) for k = 1:prob.N-1]
prob.xinds[1]
Xref[1]
for k = 1:prob.N
    # z0[prob.xinds[k]] = Xref[k]
    z0[prob.xinds[k]] = x0 
    if k < prob.N
        z0[prob.uinds[k]] = U0[k]
        z0[prob.λinds[k]] = λ0[k] 
    end
end

zsol, = MC.ipopt_solve(prob, z0, tol=1e-4)
Xsol = [zsol[xi] for xi in prob.xinds]
Usol = [zsol[ui] for ui in prob.uinds]
visualize!(vis, arm, Xsol, opt)
