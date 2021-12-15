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
using RobotDynamics

using MCTrajOpt: CylindricalBody
using MathOptInterface
const MOI = MathOptInterface 
const MC = MCTrajOpt

## Build the Arm
geom = [
    CylindricalBody(0.15, 0.50, :aluminum),
    CylindricalBody(0.10, 0.25, :aluminum),
    CylindricalBody(0.10, 0.25, :aluminum, color=colorant"green"),
    CylindricalBody(0.10, 0.25, :aluminum),
    CylindricalBody(0.10, 0.25, :aluminum, color=colorant"orange"),
    CylindricalBody(0.10, 0.10, :aluminum, color=colorant"red"),
]
joints = [
    RevoluteJoint([0,0,0], [0,0,-geom[1].length/2], [0,0,1]),
    RevoluteJoint([0,geom[1].radius, geom[1].length/2], [0,-geom[2].radius,-geom[2].length/2],[0,1,0]),
    RevoluteJoint([0,0,geom[2].length/2], [0,0,-geom[3].length/2],[0,0,1]),
    RevoluteJoint([0,-geom[3].radius,geom[3].length/2], [0,geom[4].radius,-geom[4].length/2],[0,1,0]),
    RevoluteJoint([0,0,geom[4].length/2], [0,0,-geom[5].length/2],[0,0,1]),
    RevoluteJoint([0,0,geom[5].length/2], [0,0,-geom[6].length/2],[0,1,0])
]
arm = MC.RobotArm(geom, joints, gravity=false)
@test arm.numlinks == 6

## 
θ0 = zeros(6)
x0 = MC.min2max(arm, θ0)
MC.getrind(arm, 3)
if !isdefined(Main, :vis)
    vis = launchvis(arm, x0)
end

# ## Simulate a small trajectory
# torques(t) = SA[10.0 * (1-t), -10.0 * (t < 1.5), cos(π*t)]
# sim = SimParams(1.0, 0.05)
# Xref, λref = simulate(arm, sim, torques.(sim.thist), x0)
# visualize!(vis, arm, Xref, sim)

# Goal position
θf = deg2rad.([0, 45, 90, -70,90,45])
xgoal = MC.min2max(arm, θf)
visualize!(vis, arm, xgoal)
r_2 = MC.gettran(arm, xgoal, arm.numlinks)
q_2 = MC.getquat(arm, xgoal, arm.numlinks)
p_ee = SA[0,0,geom[end].length/2]
r_ee = r_2 + MC.Amat(q_2)*p_ee

using GeometryBasics, Colors, MeshCat
setobject!(vis["goal"], Sphere(Point3(r_ee...), 0.05), MeshPhongMaterial(color=colorant"green"))


# Generate the reference trajectory
opt = SimParams(1.0, 0.02)
opt.N
Xref = map(1:opt.N) do k
    t = (k-1) / (opt.N-1)
    θ = θ0 + (θf - θ0) * t
    x0 = MC.min2max(arm, θ) 
end
visualize!(vis, arm, Xref, opt)

# Nominal reference
Xnom = [copy(x0) for k = 1:opt.N]

## Set up the problem
eeonly = true
Qr = Diagonal(SA_F64[1,1,1.]) * !eeonly
Qq = Diagonal(SA_F64[1,1,1,1.]) * !eeonly
Qν = Diagonal(@SVector fill(0.1, 3)) * !eeonly
Qω = Diagonal(@SVector fill(0.1, 3)) * !eeonly
Qe = Diagonal(@SVector fill(10.0, 3))
R = Diagonal(@SVector fill(1e-2, arm.numlinks)) * 0
prob = MC.ArmMOI(arm, opt, Qr, Qq, R, x0, Xref, rf=r_ee, p_ee=p_ee, Qν=Qν, Qω=Qω, Qe=Qe)

# # Form the reference trajectory
# Uref = torques.(sim.thist)
# zref = MC.buildtraj(prob, Xref, Uref, λref)

#############################################
# Test Interface
#############################################
ztest = MC.randtraj(prob)
grad = zeros(prob.n_nlp)
MOI.eval_objective(prob, ztest)
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
eeonly = true 
Qr = Diagonal(SA_F64[1,1,1.]) * !eeonly
Qq = Diagonal(SA_F64[1,1,1,1.]) * !eeonly
Qν = Diagonal(@SVector fill(0.1, 3)) * eeonly
Qω = Diagonal(@SVector fill(0.1, 3)) * eeonly
Qe = Diagonal(@SVector fill(10.0, 3))
R = Diagonal(@SVector fill(1e-1, arm.numlinks)) 
prob = MC.ArmMOI(arm, opt, Qr, Qq, R, x0, Xref, rf=r_ee, p_ee=p_ee, Qν=Qν, Qω=Qω, Qe=Qe)

z0 = zeros(prob.n_nlp)
U0 = [@SVector zeros(prob.L) for k = 1:prob.N-1]
λ0 = [@SVector zeros(prob.p) for k = 1:prob.N-1]
for k = 1:prob.N
    # z0[prob.xinds[k]] = Xref[k]
    z0[prob.xinds[k]] = Xnom[k]
    if k < prob.N
        z0[prob.uinds[k]] = U0[k]
        z0[prob.λinds[k]] = λ0[k] 
    end
end

zsol,solver = MC.ipopt_solve(prob, z0, tol=1e-2, c_tol=1e-6)
Xsol = [zsol[xi] for xi in prob.xinds]
Usol = [zsol[ui] for ui in prob.uinds]
visualize!(vis, arm, Xsol, opt)

tsolve = MOI.get(solver, MOI.SolveTimeSec())
data = MC.parseipoptoutput("ipopt.out")
jacnnz = length(MOI.jacobian_structure(prob))
jacdensity = MC.getjacobiandensity(prob)

using LaTeXTabulars
tikzdir = joinpath(dirname(pathof(MCTrajOpt)), "..", "tex", "figs")
topercent(x) = string(round(x*100, digits=1)) * raw"\%"
latex_tabular(joinpath(tikzdir, "arm_data.tex"),
    Tabular("lcl"),
    [
        Rule(:top),
        ["Value", "6dof Arm"],
        Rule(:mid),
        ["Variables", prob.n_nlp],
        ["Constraints", prob.m_nlp],
        ["nnz(jac)", jacnnz],
        ["Jac density", topercent(jacdensity)],
        ["Iters", data["iter"][end]],
        ["Cost", round(data["objective"][end])],
        ["Run time (s)", round(tsolve, digits=2)], 
        Rule(:bottom)
    ]
)

plot(data["iter"], data["objective"])
plot(data["iter"], data["inf_pr"], yscale=:log10)

plot(Usol)
