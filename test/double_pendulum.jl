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

##
body1 = RigidBody(1.0, Diagonal([0.1, 1.0, 1.0]))
body2 = RigidBody(1.0, Diagonal([0.1, 1.0, 1.0]))
model = DoublePendulum(body1, body2)

# Test minimal to maximal coordinate
x0 = MC.min2max(model, [deg2rad(0), deg2rad(0)])
vis = launchvis(model, x0)
visualize!(vis, model, x0)

# Generate a random state
xtest = randstate(model)
vtest = @SVector randn(12)

# Test Kinetic energy
K0 = MC.kinetic_energy(model, xtest, vtest)
@test MC.kinetic_energy(model, xtest, vtest*2) ≈ K0*4

# Test potential_energy
@test MC.potential_energy(model, xtest) ≈ 0

# Test Lagrangian
@test MC.lagrangian(model, xtest, vtest) ≈ K0
@test MC.DxL(model, xtest, vtest) ≈ ForwardDiff.gradient(x->MC.lagrangian(model, x, vtest), xtest)
@test MC.DvL(model, xtest, vtest) ≈ ForwardDiff.gradient(v->MC.lagrangian(model, xtest, v), vtest)

# Test discrete Lagrangian
dx = SVector{12}([randn(3)*0.1; normalize(randn(3)*0.1); randn(3)*0.1; normalize(randn(3)*0.1)]) 
x1test = xtest
x2test = MC.compose_states(model, x1test, MC.err2fullstate(model, dx))
xmid,vmid = MC.midpoint(model, x1test, x2test, h)

d1x, d1v = MC.D1midpoint(model, x1test, x2test, h)
d2x, d2v = MC.D2midpoint(model, x1test, x2test, h)
@test ForwardDiff.jacobian(x->MC.midpoint(model, x, x2test, h)[1], x1test) ≈ d1x
@test ForwardDiff.jacobian(x->MC.midpoint(model, x, x2test, h)[2], x1test) ≈ d1v
@test ForwardDiff.jacobian(x->MC.midpoint(model, x1test, x, h)[1], x2test) ≈ d2x
@test ForwardDiff.jacobian(x->MC.midpoint(model, x1test, x, h)[2], x2test) ≈ d2v

# Discrete Legendre transforms
∇Ld1 = let 
    xmid, vmid = MC.midpoint(model, x1test, x2test, h) 
    dx1,dv1 = MC.D1midpoint(model, x1test, x2test, h)
    (dx1'MC.DxL(model, xmid, vmid)  + dv1'MC.DvL(model, xmid, vmid))*h
end
@test ForwardDiff.gradient(x->MC.discretelagrangian(model, x, x2test, h), x1test) ≈ ∇Ld1
@test MC.D1Ld(model, x1test, x2test, h) ≈ 
    ForwardDiff.gradient(e->MC.discretelagrangian(model, x1test ⊕ e, x2test, h), e)


∇Ld2 = let 
    xmid, vmid = MC.midpoint(model, x1test, x2test, h) 
    dx2,dv2 = MC.D2midpoint(model, x1test, x2test, h)
    (dx2'MC.DxL(model, xmid, vmid)  + dv2'MC.DvL(model, xmid, vmid))*h
end
@test ForwardDiff.gradient(x->MC.discretelagrangian(model, x1test, x, h), x2test) ≈ ∇Ld2
@test MC.errstate_jacobian(model,x2test)'∇Ld2 ≈ MC.D2Ld(model, x1test, x2test, h)
@test MC.D2Ld(model, x1test, x2test, h) ≈ 
    ForwardDiff.gradient(e->MC.discretelagrangian(model, x1test, x2test ⊕ e, h), e)


# Test the error state Jacobian
e = @SVector zeros(12)
⊕(x,e) = MC.compose_states(model, x, MC.err2fullstate(model, e))
@test ForwardDiff.jacobian(e->x1test ⊕ e, e) ≈ 
    MC.errstate_jacobian(model, x1test)

MC.D1Ld(model, x1test, x2test, h)

# Joint Constraints
c = MC.joint_constraints(model, x0)
@test norm(c, Inf) < 1e-12
xviol = SA[0.5,0,0, 1,0,0,0, 1.6,0,0, 1,0,0,0]
c = MC.joint_constraints(model, xviol)
@test norm(c, Inf) ≈ 0.1
xviol = SVector{14}([x0[1:10]; MC.expm(SA[1,0,0]*0.01)])
c = MC.joint_constraints(model, xviol)
@test norm(c, Inf) ≈ 0.005 atol=1e-6

@test MC.∇joint_constraints(model, xtest) ≈ 
    ForwardDiff.jacobian(x->MC.joint_constraints(model, x), xtest)

λtest = @SVector randn(10)
@test MC.jtvp_joint_constraints(model, xtest, λtest) ≈ MC.∇joint_constraints(model, xtest)'λtest

@test MC.∇²joint_constraints(model, xtest, λtest) ≈ 
    FiniteDiff.finite_difference_hessian(x->MC.joint_constraints(model, x)'λtest, xtest) atol=1e-6

e = zeros(12)
ehess = FiniteDiff.finite_difference_hessian(e->MC.joint_constraints(model, xtest ⊕ e)'λtest, e)
G = MC.errstate_jacobian(model, xtest)
@test ehess ≈ G'MC.∇²joint_constraints(model, xtest, λtest)*G + 
    MC.∇errstate_jacobian2(model, xtest, MC.jtvp_joint_constraints(model, xtest, λtest)) atol=1e-6

# DEL
F1 = SA[0,0,0, 0,0,-1, 0,0,0, 0,0,1]
F2 = copy(F1)
λ = @SVector zeros(10)
MC.DEL(model, x1test, x2test, x3test, F1, F2, λ, h)
MC.∇joint_constraints(model, x2test)

e = zeros(24)
hess = FiniteDiff.finite_difference_hessian(e->MC.discretelagrangian(model, x1test ⊕ e[1:12], x2test ⊕ e[13:24], h), e)
d11, d12 = MC.∇D1Ld(model, x1test, x2test, h)
norm(d11 - hess[1:12,1:12], Inf)
norm(d12 - hess[1:12,13:24], Inf)

d21, d22 = MC.∇D2Ld(model, x1test, x2test, h)
norm(d21 - hess[13:24,1:12], Inf)
norm(d22 - hess[13:24,13:24], Inf)

## Simulation
sim = SimParams(1.0, 0.01)
function wrench(t)
    T1 = 0.1 <= t < 0.2 ? 1.0 : 0.0
    SA[0, 0, 0, 0, 0, T1, 0, 0, 0, 0, 0, 0] 
end
F = wrench.(sim.thist) 
x0 = MC.min2max(model, [deg2rad(0.0), deg2rad(0.0)])
Xsim = MC.simulate(model, sim, F, x0)

visualize!(vis, model, Xsim, sim)

let params = sim
    X = [zero(x0) for k = 1:params.N]
    X[1] = x0
    X[2] = x0
    k = 2
    e1 = DEL(model, X[k-1], X[k], X[k+1], λ, F[k-1],F[k], h)
end