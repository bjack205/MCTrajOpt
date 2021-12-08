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

# Test midpoint discretization 
h = 0.1
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

# Test the error state Jacobian
import Rotations.⊕
⊕(x,e) = MC.compose_states(model, x, MC.err2fullstate(model, e))
e = @SVector zeros(12)
@test ForwardDiff.jacobian(e->x1test ⊕ e, e) ≈ 
    MC.errstate_jacobian(model, x1test)

# Discrete Legendre transforms
dlt1 = zeros(12)
∇Ld1 = let 
    xmid, vmid = MC.midpoint(model, x1test, x2test, h) 
    dx1,dv1 = MC.D1midpoint(model, x1test, x2test, h)
    (dx1'MC.DxL(model, xmid, vmid)  + dv1'MC.DvL(model, xmid, vmid))*h
end
@test ForwardDiff.gradient(x->MC.discretelagrangian(model, x, x2test, h), x1test) ≈ ∇Ld1
@test MC.D1Ld(model, x1test, x2test, h) ≈ 
    ForwardDiff.gradient(e->MC.discretelagrangian(model, x1test ⊕ e, x2test, h), e)
@test MC.D1Ld(model, x1test, x2test, h) ≈ MC.D1Ld!(model, dlt1, x1test, x2test, h)

dlt2 = zeros(12)
∇Ld2 = let 
    xmid, vmid = MC.midpoint(model, x1test, x2test, h) 
    dx2,dv2 = MC.D2midpoint(model, x1test, x2test, h)
    (dx2'MC.DxL(model, xmid, vmid)  + dv2'MC.DvL(model, xmid, vmid))*h
end
@test ForwardDiff.gradient(x->MC.discretelagrangian(model, x1test, x, h), x2test) ≈ ∇Ld2
@test MC.errstate_jacobian(model,x2test)'∇Ld2 ≈ MC.D2Ld(model, x1test, x2test, h)
@test MC.D2Ld(model, x1test, x2test, h) ≈ 
    ForwardDiff.gradient(e->MC.discretelagrangian(model, x1test, x2test ⊕ e, h), e)
@test MC.D2Ld(model, x1test, x2test, h) ≈ MC.D2Ld!(model, dlt2, x1test, x2test, h)


# Joint Constraints
c = MC.joint_constraints(model, x0)
@test norm(c, Inf) < 1e-12
xviol = x0 + SA[0.0,0,0, 0,0,0,0, 0.1,0,0, 0,0,0,0]
c = MC.joint_constraints(model, xviol)
@test norm(c, Inf) ≈ 0.1
xviol = SVector{14}([x0[1:10]; MC.expm(SA[1,0,0]*0.01)])
c = MC.joint_constraints(model, xviol)
@test norm(c, Inf) ≈ 0.005 atol=1e-6
c0 = zeros(10)
@test MC.joint_constraints!(model, c0, xviol) ≈ c

@test MC.∇joint_constraints(model, xtest) ≈ 
    ForwardDiff.jacobian(x->MC.joint_constraints(model, x), xtest)

# Jacobian-transpose vector product
λtest = @SVector randn(10)
@test MC.jtvp_joint_constraints(model, xtest, λtest) ≈ MC.∇joint_constraints(model, xtest)'λtest
jtvp = zeros(12)
MC.jtvp_joint_constraints!(model, jtvp, xtest, λtest)
@test jtvp ≈ MC.errstate_jacobian(model, xtest)'MC.∇joint_constraints(model, xtest)'λtest

# Jacobian of Jtvp
@test MC.∇²joint_constraints(model, xtest, λtest) ≈ 
    FiniteDiff.finite_difference_hessian(x->MC.joint_constraints(model, x)'λtest, xtest) atol=1e-6

e = zeros(12)
ehess = FiniteDiff.finite_difference_hessian(e->MC.joint_constraints(model, xtest ⊕ e)'λtest, e)
G = MC.errstate_jacobian(model, xtest)
@test ehess ≈ G'MC.∇²joint_constraints(model, xtest, λtest)*G + 
    MC.∇errstate_jacobian2(model, xtest, MC.jtvp_joint_constraints(model, xtest, λtest)) atol=1e-6

# diff against error quaternion
chess = zeros(12,12)
MC.∇²joint_constraints!(model, chess, xtest, λtest, errstate=Val(true))
@test chess ≈ ehess atol=1e-6

# diff against quaternion
chess = zeros(12,14)
MC.∇²joint_constraints!(model, chess, xtest, λtest)
@test chess ≈ 
    ForwardDiff.jacobian(x->MC.errstate_jacobian(model, x)'MC.jtvp_joint_constraints(model, x, λtest), xtest)

## Joint forces
ξ = zeros(12)
utest = SA[0.1,-0.3]
@test MC.getwrenches!(model, ξ, xtest, utest) ≈ MC.getwrenches(model, xtest, utest)

wjac = zeros(12, 16)
@test MC.∇getwrenches!(model, wjac, xtest, utest) ≈ 
    ForwardDiff.jacobian(x->MC.getwrenches(model, x[1:14], x[15:16]), [xtest; utest])

wjac2 = zeros(24, 32)
MC.∇getwrenches!(model, wjac2, xtest, utest, ix=17:30, iu=31:32, yi=13) 
wjac3 = ForwardDiff.jacobian(x->[zeros(12); MC.getwrenches(model, x[17:30], x[31:32])], [zeros(16); xtest; utest])
@test wjac2 ≈ wjac3

wjac2 = zeros(12, 32)
MC.∇getwrenches!(model, wjac2, xtest, utest, ix=17:30, iu=31:32, yi=1) 
wjac3 = ForwardDiff.jacobian(x->MC.getwrenches(model, x[17:30], x[31:32]), [zeros(16); xtest; utest])
@test wjac2 ≈ wjac3

wjac2 = zeros(24, 16)
MC.∇getwrenches!(model, wjac2, xtest, utest, yi=13) 
wjac3 = ForwardDiff.jacobian(x->[zeros(12); MC.getwrenches(model, x[1:14], x[15:16])], [xtest; utest])
@test wjac2 ≈ wjac3

## DEL
u1test = @SVector randn(2)
u2test = @SVector randn(2)
λ = @SVector zeros(10)
λtest = @SVector randn(10)
x3test = x2test ⊕ dx 
MC.DEL(model, x1test, x2test, x3test, λ, u1test, u2test, h)
del = zeros(12)
MC.DEL!(model, del, x1test, x2test, x3test, λtest, u1test, u2test, h)
@test del ≈ MC.DEL(model, x1test, x2test, x3test, λtest, u1test, u2test, h)

##
ix1 = 1:14
iu1 = 15:16
ix2 = ix1 .+ 16
iu2 = iu1 .+ 16
ix3 = ix2 .+ 16

jac_ip = zeros(12,16*3-2)
jac_op = zero(jac_ip)
jac_an = zero(jac_ip)
delfun(y,x) = MC.DEL!(model, y, x[ix1], x[ix2], x[ix3], λtest, x[iu1], x[iu2], h)
delfun(x) = MC.DEL(model, x[ix1], x[ix2], x[ix3], λtest, x[iu1], x[iu2], h)

z = Vector([x1test; u1test; x2test; u2test; x3test])
delfun(del, z)
@test delfun(z) ≈ del

jac_op = FiniteDiff.finite_difference_jacobian(delfun, z) 
FiniteDiff.finite_difference_jacobian!(jac_ip, delfun, z)
jac_an .= 0
MC.∇DEL!(model, jac_an, x1test, x2test, x3test, λtest, u1test, u2test, h)
@test jac_ip ≈ jac_op atol=1e-6
@test jac_ip ≈ jac_an atol=1e-6

delfun2(y,x) = MC.DEL!(model, y, x[ix1], x[ix2], x[ix3], λtest, x[iu1], x[iu2], h, yi=13)
y = zeros(2*length(del))
delfun2(y, z)
jac_op = zeros(2*length(del), length(z))
jac_ip = zero(jac_op)
FiniteDiff.finite_difference_jacobian!(jac_op, (y,x)->y[13:end] .= delfun(x), z)
FiniteDiff.finite_difference_jacobian!(jac_ip, (y,x)->delfun2(y,x), z)
@test jac_op ≈ jac_ip
@test norm(jac_op[1:12,:],Inf) < 1e-12

##
e = zeros(24)
hess = FiniteDiff.finite_difference_hessian(e->MC.discretelagrangian(model, x1test ⊕ e[1:12], x2test ⊕ e[13:24], h), e)
d11, d12 = MC.∇D1Ld(model, x1test, x2test, h)
@test norm(d11 - hess[1:12,1:12], Inf) < 1e-5
@test norm(d12 - hess[1:12,13:24], Inf) < 1e-5

d21, d22 = MC.∇D2Ld(model, x1test, x2test, h)
@test norm(d21 - hess[13:24,1:12], Inf) < 1e-5
@test norm(d22 - hess[13:24,13:24], Inf) < 1e-5


##

# ## Simulation
# sim = SimParams(1.0, 0.01)
# function wrench(t)
#     T1 = 0.1 <= t < 0.7 ? 1.0 : 0.0
#     SA[0,0,0, 0,0,T1, 0,0,0, 0,0,-T1] 
# end
# F = wrench.(sim.thist) 
# x0 = MC.min2max(model, [deg2rad(0.0), deg2rad(0.0)])
# Xsim = MC.simulate(model, sim, F, x0)

# visualize!(vis, model, Xsim, sim)

# ##
# body1 = RigidBody(1.0, Diagonal([0.1, 1.0, 1.0]))
# body2 = RigidBody(1.0, Diagonal([0.1, 1.0, 1.0]))
# model = DoublePendulum(body1, body2, gravity = false)
# sim = SimParams(5.0, 0.01)
# sim.N
# function control(t)
#     T1 = 0.1 <= t < 0.7 ? 2.0 : 0.0
#     T2 = cos(pi*t)*2
#     SA[T1,T2]
# end
# U = control.(sim.thist)
# # plot(hcat(Vector.(U)...)')

# x0 = MC.min2max(model, [0.0,0])
# # vis = launchvis(model, x0)
# visualize!(vis, model, x0)
# Xsim = MC.simulate(model, sim, U, x0)
# visualize!(vis, model, Xsim, sim)

