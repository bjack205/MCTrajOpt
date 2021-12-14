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

##
body1 = RigidBody(1.0, Diagonal([0.1, 1.0, 1.0]))
body2 = RigidBody(1.0, Diagonal([0.1, 1.0, 1.0]))
joint = RevoluteJoint(SA[0.5,0,0], SA[-0.5,0,0], SA[0,0,1])
twobody = SpaceBar(body1, body2, joint)

# Simulate the system Forward
sim = SimParams(1.0, 0.05) 
function force(t)
    0.25 <= t < 0.5 && (return 10)
    0.5 <= t < 1.0 && (return -1)
    return 0
end
u = force.(sim.thist)
F = [SA[0,0,0, 0,0,-uk, 0,0,0, 0,0,uk] for uk in u]

x0 = SA[0,0,0, 1,0,0,0, 1,0,0, 1,0,0,0.]
X = simulate(twobody, sim, F, x0)
xf = X[end]


# Animation
vis = launchvis(twobody, x0)
visualize!(vis, twobody, X, sim)

# Test minimal to maximal coordinate kinematicx
r0 = SA[0,0,0] 
q0 = MCTrajOpt.expm2(SA[0,0,0]*deg2rad(45))
visualize!(vis, twobody, min2max(twobody, [r0; q0; deg2rad(30)]))

## TrajOpt
opt = SimParams(1.0, 0.1) 
Qr = Diagonal(SA_F64[1,1,1.])
Qq = Diagonal(SA_F64[1,1,1,1.])
R = Diagonal(SA_F64[1])

prob = MCTrajOpt.TwoBodyMOI(body1, body2, joint, opt, Qr, Qq, R, x0, xf)

# Create initial guess
z0 = zeros(prob.n_nlp)
X0 = [copy(x0) for k = 1:prob.N]
U0 = [SA[0.0] for k = 1:prob.N-1]
λ0 = [@SVector zeros(prob.p) for k = 1:prob.N-1]
for k = 1:prob.N
    z0[prob.xinds[k]] = X0[k]
    if k < prob.N
        z0[prob.uinds[k]] = U0[k]
        z0[prob.λinds[k]] = λ0[k] 
    end
end

grad_f = zeros(prob.n_nlp)
c = zeros(prob.m_nlp)
jac = zeros(prob.m_nlp, prob.n_nlp)

MOI.eval_objective(prob, z0)
MOI.eval_objective_gradient(prob, grad_f, z0)

# Joint Constraints
xtest = SVector{14}([randn(3); normalize(randn(4)); randn(3); normalize(randn(4))])
q = SVector{4}(randn(4))
r = @SVector randn(3)
b = @SVector randn(3)
λtest = @SVector rand(5)

MCTrajOpt.joint_constraints(prob, xtest)
joint_jac = ForwardDiff.jacobian(x->MCTrajOpt.joint_constraints(prob, x), xtest)
@test MCTrajOpt.∇joint_constraints(prob, xtest) ≈ joint_jac

@test MCTrajOpt.jtvp_joint_constraints(prob, xtest, λtest) ≈ joint_jac'λtest
# @time joint_hess = ForwardDiff.hessian(x->MCTrajOpt.joint_constraints(prob, x)'λtest, xtest)
joint_hess = FiniteDiff.finite_difference_hessian(x->MCTrajOpt.joint_constraints(prob, x)'λtest, xtest) 
@test MCTrajOpt.∇²joint_constraints(prob, xtest, λtest) ≈ joint_hess atol=1e-6

# DEL Constraints
using MCTrajOpt: G, L, Hmat, Tmat
function cay(g)
    M = 1/sqrt(1+g'g)
    SA[M, M*g[1], M*g[2], M*g[3]]
end
let h = prob.params.h, body = prob.model.b1, R = MCTrajOpt.R
    r = [@SVector randn(3) for i = 1:3]
    F = [@SVector randn(3) for i = 1:2]
    MCTrajOpt.DEL_trans(body, r[1], r[2], r[3], F[1], F[2], h)

    dr1, dr2, dr3, dF1, dF2 = MCTrajOpt.∇DEL_trans(body, r[1], r[2], r[3], F[1], F[2], h)
    @test dr1 ≈ ForwardDiff.jacobian(x->MCTrajOpt.DEL_trans(body, x, r[2], r[3], F[1], F[2], h), r[1])
    @test dr2 ≈ ForwardDiff.jacobian(x->MCTrajOpt.DEL_trans(body, r[1], x, r[3], F[1], F[2], h), r[2])
    @test dr3 ≈ ForwardDiff.jacobian(x->MCTrajOpt.DEL_trans(body, r[1], r[2], x, F[1], F[2], h), r[3])
    @test dF1 ≈ ForwardDiff.jacobian(x->MCTrajOpt.DEL_trans(body, r[1], r[2], r[3], x, F[2], h), F[1])
    @test dF2 ≈ ForwardDiff.jacobian(x->MCTrajOpt.DEL_trans(body, r[1], r[2], r[3], F[1], x, h), F[2])

    q = [normalize(@SVector randn(4)) for i = 1:4]
    T = [@SVector randn(3) for i = 1:2]
    g = @SVector zeros(3)
    J = body.J
    dq1, dq2, dq3, dT1, dT2 = MCTrajOpt.∇DEL_rot(body, q[1], q[2], q[3], T[1], T[2], h)

    # dq2 won't match ForwarDiff because the DEL equation is already differentiated wrt q2
    #   we'd need to differentiate the original Lagrangian to get the true answer
    # @test dq1 ≈ ForwardDiff.jacobian(x->MCTrajOpt.DEL_rot(body, L(q[1])*cay(x), q[2], q[3], T[1], T[2], h), g)
    # @test dq2 ≈ ForwardDiff.jacobian(x->MCTrajOpt.DEL_rot(body, q[1], L(q[2])*cay(x), q[3], T[1], T[2], h), g)
    # @test dq3 ≈ ForwardDiff.jacobian(x->MCTrajOpt.DEL_rot(body, q[1], q[2], L(q[3])*cay(x), T[1], T[2], h), g)
    # @test dT1 ≈ ForwardDiff.jacobian(x->MCTrajOpt.DEL_rot(body, q[1], q[2], q[3], x, T[2], h), T[1])
    # @test dT2 ≈ ForwardDiff.jacobian(x->MCTrajOpt.DEL_rot(body, q[1], q[2], q[3], T[1], x, h), T[2])
    @test dq1 ≈ ForwardDiff.jacobian(x->MCTrajOpt.DEL_rot(body, x, q[2], q[3], T[1], T[2], h), q[1])
    @test dq2 ≈ ForwardDiff.jacobian(x->MCTrajOpt.DEL_rot(body, q[1], x, q[3], T[1], T[2], h), q[2])
    @test dq3 ≈ ForwardDiff.jacobian(x->MCTrajOpt.DEL_rot(body, q[1], q[2], x, T[1], T[2], h), q[3])
    @test dT1 ≈ ForwardDiff.jacobian(x->MCTrajOpt.DEL_rot(body, q[1], q[2], q[3], x, T[2], h), T[1])
    @test dT2 ≈ ForwardDiff.jacobian(x->MCTrajOpt.DEL_rot(body, q[1], q[2], q[3], T[1], x, h), T[2])
    
end

# Constraints
MOI.eval_constraint(prob, c, z0)
jac .= 0
MOI.eval_constraint_jacobian(prob, jac, z0)
jac0 = zero(jac)
ForwardDiff.jacobian!(jac0, (c,x)->MOI.eval_constraint(prob, c, x), c, z0)
FiniteDiff.finite_difference_jacobian!(jac0, (c,x)->MOI.eval_constraint(prob, c, x), z0)
@test jac0 ≈ jac atol=1e-6

# Solve
zsol, = MCTrajOpt.ipopt_solve(prob, z0)

MOI.eval_constraint(prob, c, zsol)
norm(c, Inf)
findmax(c)
c
