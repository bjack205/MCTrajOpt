using MCTrajOpt
using ForwardDiff
using StaticArrays
using LinearAlgebra
using Test
using Random

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
@test D1L_vel(body,x0,ν0) ≈ ForwardDiff.gradient(x->Lagrangian_vel(body,x,ν0), x0)
@test D2L_vel(body,x0,ν0) ≈ ForwardDiff.gradient(ν->Lagrangian_vel(body,x0,ν), ν0)

@test D1L_dot(body,x0,ẋ0) ≈ ForwardDiff.gradient(x->Lagrangian_dot(body,x,ẋ0),x0)
@test D2L_dot(body,x0,ẋ0) ≈ ForwardDiff.gradient(xdot->Lagrangian_dot(body,x0,xdot),ẋ0)

# Test discrete Lagrangian derivatives
x1 = copy(x0)
x2 = [@SVector randn(3); normalize(@SVector randn(4))]
@test D1Ld(body,x1,x2,params.h) ≈ ForwardDiff.gradient(x->Ld(body,x,x2,params.h), x1)
@test D2Ld(body,x1,x2,params.h) ≈ ForwardDiff.gradient(x->Ld(body,x1,x,params.h), x2)

# Test simulation

F = [SA[0,0,0.5*(0.1<t<0.5), 0,0,1.0* (0.1<t<.5)] for t in params.thist]  # force in the world frame, torque in body frame?
x0 = SA[0,0,0, sqrt(2)/2, sqrt(2)/2, 0,0]
X = [zero(x0) for k = 1:params.N]
X[1] = x0
X[2] = x0
X[3] = x0
e = DEL(body, X[1], X[2], X[3], F[1], F[2], params.h)
H = D3_DEL(body, X[1], X[2], X[3], F[1], F[2], params.h)

X = simulate(body, params, F, x0)

## Plots
using Plots
Xdot = diff(X) ./ params.h 
V = [inv_kinematics(X[k], Xdot[k]) for k = 1:length(Xdot)] 
Xhist = hcat(Vector.(X)...)'
Vhist = hcat(Vector.(V)...)'
plot(params.thist[1:end-1], Vhist[:,4:6])

## Visualization
using MeshCat, GeometryBasics, CoordinateTransformations, Rotations, Colors
vis = Visualizer()
dim = Vec(0.5, 0.7, 0.3)
geom  = Rect3(-dim/2, dim)
setobject!(vis["body"], geom, MeshPhongMaterial(color=colorant"green"))
open(vis)

anim = MeshCat.Animation(floor(Int,1/params.h))
for k = 1:params.N
    atframe(anim, k) do
        r = X[k][1:3]
        q = X[k][4:7]
        settransform!(vis["body"], compose(Translation(r), LinearMap(UnitQuaternion(q))))
    end
end
setanimation!(vis, anim)