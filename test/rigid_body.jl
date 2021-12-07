import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using MCTrajOpt
using ForwardDiff
using StaticArrays
using LinearAlgebra
using Test
using Random
using BenchmarkTools
const MC = MCTrajOpt

## Check the attitude Jacobian
q0 = normalize(@SVector randn(4))
g = zeros(3)
ForwardDiff.jacobian(g->MC.L(q0)*MC.cayleymap(g), g) ≈ MC.G(q0)

##
r0 = @SVector randn(3)
q0 = normalize(@SVector randn(4))
x0 = [r0; q0]
ν0 = @SVector randn(6)
ẋ0 = kinematics(x0,ν0)

# Make sure the kinematics work
@test inv_kinematics(x0, ẋ0) ≈ ν0
@test kinematics(x0, ν0) ≈ ẋ0

@test ForwardDiff.jacobian(x->inv_kinematics(x,ẋ0), x0) ≈ D1Kinv(x0,ẋ0)
@test ForwardDiff.jacobian(xdot->inv_kinematics(x0,xdot), ẋ0) ≈ D2Kinv(x0,ẋ0)

# Test the Lagrangian derivatives
body = RigidBody(1.0, Diagonal([0.1, 1, 1]))
params = SimParams(5.0, 0.05) 
@test D1L(body,x0,ν0) ≈ ForwardDiff.gradient(x->Lagrangian(body,x,ν0), x0)
@test D2L(body,x0,ν0) ≈ ForwardDiff.gradient(ν->Lagrangian(body,x0,ν), ν0)

# Test discrete Lagrangian derivatives
x1 = copy(x0)
x2 = [@SVector randn(3); normalize(@SVector randn(4))]
@test D1Ld(body,x1,x2,params.h) ≈ ForwardDiff.gradient(x->Ld(body,x,x2,params.h), x1)
@test D2Ld(body,x1,x2,params.h) ≈ ForwardDiff.gradient(x->Ld(body,x1,x,params.h), x2)

# Test DEL
F1 = @SVector randn(6)
F2 = @SVector randn(6)
x3 = [@SVector randn(3); normalize(@SVector randn(4))]
e = zeros(6)
⊕(x1,e) = MC.compose_states(x1, MC.err2fullstate(e))
@test MC.DEL(body, x1, x2, x3, F1, F2, params.h) ≈ ForwardDiff.gradient(
        e->MC.Ld(body, x1, x2 ⊕ e, params.h) + MC.Ld(body, x2 ⊕ e, x3, params.h), e 
    ) + params.h*(F1+F2)/2


# Test simulation
params = SimParams(1.0, 0.001) 
F = [SA[0,0,1, 0,0,1] for t in params.thist]  # force in the world frame, torque in body frame?
x0 = SA[0,0,0, 1,0,0,0.]
X = simulate(body, params, F, x0, newton_iters=50)
@test abs(X[end][3] - 0.5) < 1e-3
@test abs(norm(MC.logm(X[end][4:7])) - 0.5) < 1e-3


#############################################
# Trajectory Optimization
#############################################
using MathOptInterface
using Quaternions
const MOI = MathOptInterface 

##
body = RigidBody(1.0, Diagonal([0.1, 1, 1]))
sim = SimParams(5.0, 0.05) 
sim = SimParams(2.0, 0.1)

Qcost = Diagonal(SA_F64[10,10,10])
Rcost = Diagonal(SA_F64[.1,.1,.1, 1,1,1]) * 100

x0 = SA[0,0,0, 1,0,0,0.]
x1 = copy(x0)
xf = SA[1,2,3, sqrt(2)/2,sqrt(2)/2,0,0.]  # rotate 180 about x

# Create the NLP problem
prob = MCTrajOpt.ProblemMOI(body, sim, Qcost, Rcost, x0, x1, xf)
length(MOI.jacobian_structure(prob)) / (prob.n_nlp * prob.m_nlp) * 100

# Construct the initial guess
#   Linear interpolation / slerp from start to goal
r0 = x0[1:3]
rf = xf[1:3]
q0 = Quaternion(x0[4:7]...)
qf = Quaternion(xf[4:7]...)
X0 = map(1:prob.N) do k
    t = (k-1)/(prob.N-1)
    q = slerp(q0, qf, t)
    r = r0 + (rf - r0) * t
    SA[r[1], r[2], r[3], q.s, q.v1, q.v2, q.v3]
end
U0 = [@SVector zeros(6) for k = 1:prob.N-1]
z0 = zeros(prob.n_nlp)
packZ!(prob, z0, X0, U0)

# Create a simulated trajectory 
Usim = push!(deepcopy(U0), U0[end])
Xsim = simulate(body, sim, Usim, x0)
zsim = zero(z0)
packZ!(prob, zsim, Xsim, Usim)

# Solve
zsol, = MCTrajOpt.ipopt_solve(prob, z0, max_iter=1_000)
Xsol = [zsol[xi] for xi in prob.xinds]
Usol = [zsol[ui] for ui in prob.uinds]
Xsol[end]

anim = MeshCat.Animation(floor(Int,1/sim.h))
for k = 1:sim.N
    atframe(anim, k) do
        r = Xsol[k][1:3]
        q = Xsol[k][4:7]
        settransform!(vis["body"], compose(Translation(r), LinearMap(UnitQuaternion(q))))
    end
end
setanimation!(vis, anim)

## Test the functions
MOI.eval_objective(prob, z0) < MOI.eval_objective(prob, zsim)

grad_f = zeros(prob.n_nlp)
MOI.eval_objective_gradient(prob, grad_f, z0)

c = zeros(prob.m_nlp)
MOI.eval_constraint(prob, c, z0)
norm(c,Inf) > 0
MOI.eval_constraint(prob, c, zsim)
norm(c,Inf) < 1e-10 

# constraint jacobian
using SparseArrays
jac = spzeros(prob.m_nlp, prob.n_nlp)
ForwardDiff.jacobian!(jac, (c,x)->MOI.eval_constraint(prob, c, x), c, z0)

jac_rc = MOI.jacobian_structure(prob)
row = [rc[1] for rc in jac_rc]
col = [rc[2] for rc in jac_rc]

J = zeros(length(row))
MOI.eval_constraint_jacobian(prob, J, z0)

jac0 = sparse(row,col,J, prob.m_nlp, prob.n_nlp)
jac ≈ jac0


##
@btime MOI.eval_objective($prob, $z0)
@btime MOI.eval_objective_gradient($prob, $grad_f, $z0)
@btime MOI.eval_constraint($prob, $c, $z0)

@btime ForwardDiff.jacobian!($jac, (c,x)->MOI.eval_constraint($prob, c, x), $c, $z0)
@btime MOI.eval_constraint_jacobian($prob, $J, $z0)

function row_col!(row,col,r,c)
    for cc in c
        for rr in r
            push!(row,convert(Int,rr))
            push!(col,convert(Int,cc))
        end
    end
    return row, col
end

function sparsity_jacobian(n,m)

    row = []
    col = []

    r = 1:m
    c = 1:n

    row_col!(row,col,r,c)

    return collect(zip(row,col))
end
sparsity_jacobian(30, 9) == 
