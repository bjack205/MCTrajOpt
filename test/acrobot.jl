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
using RobotDynamics
import RigidBodyDynamics

using MathOptInterface
const MOI = MathOptInterface 
const MC = MCTrajOpt
const RBD = RigidBodyDynamics

## Generate the model
Random.seed!(1)
const ℓ = 0.5
geom1 = MC.CylindricalBody(0.03, ℓ, :aluminum)
geom2 = MC.CylindricalBody(0.03, ℓ, :aluminum)
body1 = RigidBody(geom1)
body2 = RigidBody(geom2)

# body1 = RigidBody(1.0, Diagonal([1.0, 1.0, 0.01]))
# body2 = RigidBody(1.0, Diagonal([1.0, 1.0, 0.01]))
model = DoublePendulum(body1, body2, gravity = true, acrobot=true, len=ℓ)
opt = SimParams(2.0, 0.02)
opt.N

# Goal position
θ0 = SA[-pi, 0, 0,0]
x0 = MC.min2max(model, θ0[1:2])
θgoal = SA[0,0,0,0]
xgoal = MC.min2max(model, θgoal[1:2]) 
r_2 = MC.gettran(model, xgoal)[2]
q_2 = MC.getquat(model, xgoal)[2]
p_ee = SA[0,0,ℓ/2]
r_ee = r_2 + MC.Amat(q_2)*p_ee

# Reference trajectory is just the goal position
Xref_min = [copy(θgoal) for k = 1:opt.N]
Xref_max = map(x->MC.min2max(model, x), Xref_min)

# Set up the problems
Qr = Diagonal(SA_F64[1,1,1.]) * 1
Qq = Diagonal(SA_F64[1,1,1,1.])
R = Diagonal(@SVector fill(1e-1, MC.control_dim(model)))
prob_max = MC.DoublePendulumMOI(model, opt, Qr, Qq, R, x0, Xref_max, rf=r_ee, p_ee=p_ee)

p_ee2 = p_ee - model.joint1.p2  # adjust end effector position to be relative to joint
prob_min = MC.DoublePendulumMOI(model, opt, Qr, Qq, R, x0, Xref_min, rf=r_ee, p_ee=p_ee2, minimalcoords=true)

# Create initial guess
Random.seed!(2)
U0 = [SA[0*sin(t*2pi) + 0.0*randn() + t*5 - 0sin(t*pi/(0.2))*(t<0.2)] for t in opt.thist]
# plot(U0)
Xsim,λsim = simulate(model, opt, U0, x0)
visualize!(vis, model, Xsim, opt)

z0_min = let prob = prob_min
    z0 = zeros(prob.n_nlp)
    for k = 1:prob.N
        z0[prob.xinds[k]] = θ0
        # z0[prob.xinds[k]] = θlin[k] 
        # z0[prob.xinds[k]] = θsim[k]
        if k < prob.N
            z0[prob.uinds[k]] = U0[k]
        end
    end
    z0
end

z0_max = let prob = prob_max
    z0 = zeros(prob.n_nlp)
    λ0 = [@SVector zeros(prob.p) for k = 1:prob.N-1]
    _,λsim = simulate(model, opt, U0, x0)
    for k = 1:prob.N
        z0[prob.xinds[k]] = x0
        # z0[prob.xinds[k]] = Xsim[k]
        # z0[prob.xinds[k]] = Xlin[k]
        if k < prob.N
            z0[prob.uinds[k]] = U0[k]
            # z0[prob.λinds[k]] = λ0[k] 
            z0[prob.λinds[k]] = λsim[k]
        end
    end
    z0
end

# Check initial costs
MOI.eval_objective(prob_min, z0_min)
MOI.eval_objective(prob_max, z0_max)

# Solve
outfile = "ipopt.out"
zsol_max, solver_max = MC.ipopt_solve(prob_max, z0_max, tol=1e-0, c_tol=1e-5, goal_tol=1e-4)
Xsol_max = [zsol_max[xi] for xi in prob_max.xinds]
Usol_max = [zsol_max[ui] for ui in prob_max.uinds]
λsol_max = [zsol_max[λi] for λi in prob_max.λinds]
if isdefined(Main, :vis)
    visualize!(vis, model, Xsol_max, opt)
end
data_max = MC.parseipoptoutput(outfile)
tsolve_max = MOI.get(solver_max, MOI.SolveTimeSec())
jacdensity_max = MC.getjacobiandensity(prob_max)
nnz_max = length(MOI.jacobian_structure(prob_max))

##
zsol_min, solver_min = MC.ipopt_solve(prob_min, z0_min, tol=1e-0, c_tol=1e-5, goal_tol=1e-4)
Xsol_min = [zsol_min[xi] for xi in prob_min.xinds]
Usol_min = [zsol_min[ui] for ui in prob_min.uinds]
λsol_min = [zsol_min[λi] for λi in prob_min.λinds]
Xmin_max = map(x->MC.min2max(model, x), Xsol_min)
if isdefined(Main, :vis)
    visualize!(vis, model, Xmin_max, opt)
end
data_min = MC.parseipoptoutput(outfile)
tsolve_min = MOI.get(solver_min, MOI.SolveTimeSec())
jacdensity_min = MC.getjacobiandensity(prob_min)
nnz_min = length(MOI.jacobian_structure(prob_min))

## Visualizer
if !isdefined(Main, :vis)
    vis = launchvis(model, x0; geom=[geom1, geom2])
end
visualize!(vis, model, Xsol_max, opt)
visualize!(vis, model, Xmin_max, opt)

# Compare solutions by converting min to max coordinates
zmin_max = let prob = prob_max
    z0 = zeros(prob.n_nlp)
    λ0 = [@SVector zeros(prob.p) for k = 1:prob.N-1]
    for k = 1:prob.N
        z0[prob.xinds[k]] = Xmin_max[k] 
        if k < prob.N
            z0[prob.uinds[k]] = Usol_min[k] 
            z0[prob.λinds[k]] = λ0[k] 
        end
    end
    z0
end
MOI.eval_objective(prob_max, zsol_max)
MOI.eval_objective(prob_min, zsol_min)
MOI.eval_objective(prob_max, zmin_max)

## Get EE Position
ee_max = map(x->MC.getendeffectorposition(prob_max, x), Xsol_max)
ee_min = map(x->MC.getendeffectorposition(prob_min, x), Xsol_min)
ee_minmax = map(x->MC.getendeffectorposition(prob_max, x), Xmin_max)
norm(ee_minmax-ee_min)

## Plots
using PGFPlotsX
using LaTeXStrings
using LaTeXTabulars
p_torques = @pgf TikzPicture(
    Axis({
        xlabel="time (s)",
        ylabel="control torque " * L"(N \cdot m)",
        "legend style"="{at={(0.03,0.93)},anchor=north west}",
    },
        PlotInc({ mark="none", "thick", "cyan" }, Table(x=opt.thist[1:end-1], y=Usol_min)),
        PlotInc({ mark="none", "thick", "orange" }, Table(x=opt.thist[1:end-1], y=Usol_max)),
        Legend(["Min Coords", "Max Coords"])
    )
)
p_cost = @pgf TikzPicture(
    Axis({
        xlabel="iterations",
        ylabel="objective value"
    },
        PlotInc({ mark="none", "thick", "cyan" }, Table(x=data_min["iter"], y=data_min["objective"])),
        PlotInc({ mark="none", "thick", "orange" }, Table(x=data_max["iter"], y=data_max["objective"])),
        Legend(["Min Coords", "Max Coords"])
    )
)
p_viol = @pgf TikzPicture(
    Axis({
        xlabel="iterations",
        ylabel="constraint violation",
        ymode="log",
        "legend style"="{at={(0.03,0.03)},anchor=south west}",
    },
        PlotInc({ mark="none", color="cyan", "thick", "solid" }, 
            Table(x=data_min["iter"], y=data_min["inf_pr"])
        ),
        PlotInc({ mark="none", color="cyan", "thick", "dashed" }, 
            Table(x=data_min["iter"], y=data_min["inf_du"])
        ),
        PlotInc({ mark="none", color="orange", "thick", "solid" }, 
            Table(x=data_max["iter"], y=data_max["inf_pr"])
        ),
        PlotInc({ mark="none", color="orange", "thick", "dashed" }, 
            Table(x=data_max["iter"], y=data_max["inf_du"])
        ),
        Legend(
            ["Min Coords - primal", "Min Coords - dual", 
            "Max Coords - primal", "Max Coords - dual"]
        )
    )
)
p_eemin = @pgf TikzPicture(
    Axis({
        xlabel="x position", ylabel="z position", 
    },
        PlotInc({ "scatter", "black", "scatter_src=explicit" }, 
            Table({x="x", y="y", meta="col"}, 
                x=[p[1] for p in ee_min], y=[p[3] for p in ee_min], col=1:opt.N
            )
        )
    )
)
p_eemax = @pgf TikzPicture(
    Axis({
        xlabel="x position", ylabel="z position", 
    },
        PlotInc({ "scatter", "black", "scatter_src=explicit" }, 
            Table({x="x", y="y", meta="col"}, 
                x=[p[1] for p in ee_max], y=[p[3] for p in ee_max], col=1:opt.N
            )
        )
    )
)
tikzdir = joinpath(dirname(pathof(MCTrajOpt)), "..", "tex", "figs")
pgfsave(joinpath(tikzdir,"acrobot_torques.tikz"), p_torques, include_preamble=false)
pgfsave(joinpath(tikzdir,"acrobot_cost.tikz"), p_cost, include_preamble=false)
pgfsave(joinpath(tikzdir,"acrobot_viol.tikz"), p_viol, include_preamble=false)
pgfsave(joinpath(tikzdir,"acrobot_eemin.tikz"), p_eemin, include_preamble=false)
pgfsave(joinpath(tikzdir,"acrobot_eemax.tikz"), p_eemax, include_preamble=false)

topercent(x) = string(round(x*100, digits=1)) * raw"\%"
latex_tabular(joinpath(tikzdir, "acrobot_data.tex"),
    Tabular("lcl"),
    [
        Rule(:top),
        ["Value", "Min Coords", "Max Coords"],
        Rule(:mid),
        ["Variables", prob_min.n_nlp, prob_max.n_nlp],
        ["Constraints", prob_min.m_nlp, prob_max.m_nlp],
        ["nnz(jac)", nnz_min, nnz_max],
        ["Jac density", topercent(jacdensity_min), topercent(jacdensity_max)],
        ["Iters", data_min["iter"][end], data_max["iter"][end]],
        ["Cost", round(data_min["objective"][end]), round(data_max["objective"][end])],
        ["Run time (s)", round(tsolve_min, digits=2), round(tsolve_max, digits=2)], 
        Rule(:bottom)
    ]
)


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


Xsim,λsim = simulate(model, opt, Usol, x0)
visualize!(vis, model, Xsim, opt)
norm(λsim - λsol)
plot(Usol)



#############################################
## Minimal Coordinates
#############################################

Xref2 = [SA[0,0,0,0] for k = 1:opt.N]
Q = Diagonal(SA[10,10,1e1,1e1])
visualize!(vis, model, x0)
p_ee2 = p_ee - model.joint1.p2  # adjust end effector position to be relative to joint
prob = MC.DoublePendulumMOI(model, opt, Qr, Qq, R, x0, Xref2, rf=r_ee, p_ee=p_ee2, minimalcoords=true)

# sim = SimParams(5.0, 0.01)
# xinit = SA[-pi/2 + 0.1,0,0,0]
# Usim = [3*cos(t) for t in sim.thist]
# Xsim = MC.simulate_mincoord(prob, sim, Usim, xinit)
# Xsolmax = map(Xsim) do x
#     MC.min2max(model, x)
# end
# if isdefined(Main, :vis)
#     visualize!(vis, model, Xsolmax, sim)
# end


# Create initial guess
z0 = zeros(prob.n_nlp)
U0 = [SA[randn()] for k = 1:prob.N-1]
for k = 1:prob.N
    z0[prob.xinds[k]] = [θ0; SA[0,0]]
    if k < prob.N
        z0[prob.uinds[k]] = U0[k]
    end
end


zsol, = MC.ipopt_solve(prob, z0, tol=1e-4, goal_tol=1e-6)
Xsol = [zsol[xi] for xi in prob.xinds]
Usol = [zsol[ui] for ui in prob.uinds]
λsol = [zsol[λi] for λi in prob.λinds]
Xsolmax = map(Xsol) do x
    MC.min2max(model, x)
end
if isdefined(Main, :vis)
    visualize!(vis, model, Xsolmax, opt)
end
prob.xinds[2]

## Check functions
ztest = MC.randtraj(prob)
grad_f = zeros(prob.n_nlp)
c = zeros(prob.m_nlp)
rc = MOI.jacobian_structure(prob)
row = [idx[1] for idx in rc]
col = [idx[2] for idx in rc]
jac = zeros(length(rc)) 
jac0 = zeros(prob.m_nlp, prob.n_nlp)

MOI.eval_objective(prob, ztest)
MOI.eval_objective_gradient(prob, grad_f, ztest)

MOI.eval_constraint(prob, c, ztest)
MOI.eval_constraint(prob, c, zsol)
MC.getendeffectorposition(prob, zsol[prob.xinds[end][1:2]])

MOI.eval_constraint_jacobian(prob, jac, ztest)
FiniteDiff.finite_difference_jacobian!(jac0, (c,x)->MOI.eval_constraint(prob, c, x), ztest)

@test all(isfinite,c)
@test all(isfinite,grad_f)
@test all(isfinite,jac)
@test all(isfinite,jac0)
@test sparse(row, col, jac, prob.m_nlp, prob.n_nlp) ≈ jac0 atol=1e-6
