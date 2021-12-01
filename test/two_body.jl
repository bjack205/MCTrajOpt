body1 = RigidBody(1.0, Diagonal([0.1, 1.0, 1.0]))
body2 = RigidBody(1.0, Diagonal([0.1, 1.0, 1.0]))
joint = RevoluteJoint(SA[0.5,0,0], SA[-0.5,0,0], SA[0,0,1])
twobody = TwoBody(body1, body2, joint)
params = SimParams(5.0, 0.05) 

# Generate state and velocity vectors
x0, x1 = map(1:2) do i
    r0 = @SVector randn(3)
    q0 = normalize(@SVector randn(4))
    r1 = @SVector randn(3)
    q1 = normalize(@SVector randn(4))
    [r0; q0; r1; q1]
end
ν0 = @SVector randn(12)
ν1 = @SVector randn(12)
ẋ0 = kinematics(twobody, x0, ν0)
ẋ1 = kinematics(twobody, x1, ν1)

# Make sure the kinematics work
@test inv_kinematics(twobody, x0, ẋ0) ≈ ν0
@test kinematics(twobody, x0, ν0) ≈ ẋ0

@test ForwardDiff.jacobian(x->inv_kinematics(twobody,x,ẋ0), x0) ≈ D1Kinv(twobody,x0,ẋ0)
@test ForwardDiff.jacobian(xdot->inv_kinematics(twobody,x0,xdot), ẋ0) ≈ D2Kinv(twobody,x0,ẋ0)

# Test the Lagrangian derivatives
@test D1L_vel(twobody,x0,ν0) ≈ ForwardDiff.gradient(x->Lagrangian_vel(twobody,x,ν0), x0)
@test D2L_vel(twobody,x0,ν0) ≈ ForwardDiff.gradient(ν->Lagrangian_vel(twobody,x0,ν), ν0)

@test D1L_dot(twobody,x0,ẋ0) ≈ ForwardDiff.gradient(x->Lagrangian_dot(twobody,x,ẋ0),x0)
@test D2L_dot(twobody,x0,ẋ0) ≈ ForwardDiff.gradient(xdot->Lagrangian_dot(twobody,x0,xdot),ẋ0)

# Test discrete Lagrangian derivatives
x2 = x0
@test D1Ld(twobody,x1,x2,params.h) ≈ ForwardDiff.gradient(x->Ld(twobody,x,x2,params.h), x1)
@test D2Ld(twobody,x1,x2,params.h) ≈ ForwardDiff.gradient(x->Ld(twobody,x1,x,params.h), x2)

# Constraints
jac = ForwardDiff.jacobian(q->MCTrajOpt.constraints(twobody, q), x0)
@test MCTrajOpt.∇constraints(twobody, x0) ≈ jac

# Discrete Euler-Lagrange
u = 0.5 * [0; 0; ones(19); -ones(40); zeros(10); ones(40); -ones(10)]
F = [SA[0,0,0, 0,0,-uk, 0,0,0, 0,0,uk] for uk in u]

x1 = SA[0,0,0, 1,0,0,0, 1,0,0, 1,0,0,0]
x2 = copy(x1)
x3 = copy(x1)
MCTrajOpt.constraints(twobody, x2) ≈ zeros(5)

# Simulation
x0 = SA[0,0,0, 1,0,0,0, 1,0,0, 1,0,0,0.]
X = simulate(twobody, params, F, x0)

##
k = 42
X[k-1] = [0.049522896024851666, 0.0, 0.0, 0.974924152934548, 0.0, 0.0, -0.2225374036535229, 0.9504771039751554, 0.0, 0.0, 0.974924152934548, 0.0, 0.0, 0.2225374036535229]
X[k] = [0.04954513340956883, 0.0, 0.0, 0.9749127481936183, 0.0, 0.0, -0.22258736129790394, 0.9504548665904385, 0.0, 0.0, 0.9749127481936183, 0.0, 0.0, 0.22258736129790394] 
X[k+1] = X[k]
λ = @SVector zeros(5)
e1 = DEL(twobody, X[k-1], X[k], X[k+1], λ, F[k-1], F[k], params.h)
e2 = MCTrajOpt.constraints(twobody, x3)
e = [e1;e2]
D = D3_DEL(twobody, x1, x2, x3, λ, F[2], F[3], params.h)
C2 = MCTrajOpt.∇constraints(twobody, x2) * MCTrajOpt.errstate_jacobian(twobody, x2)
C3 = MCTrajOpt.∇constraints(twobody, x3) * MCTrajOpt.errstate_jacobian(twobody, x3)
H = [D params.h*C2'; C3 zeros(5,5)]
(H\e)
params.h



## Visualization
using MeshCat, GeometryBasics, CoordinateTransformations, Rotations, Colors
vis = Visualizer()
geom = Cylinder(Point(-0.5,0,0), Point(0.5,0,0), 0.2)
setobject!(vis["body1"], geom, MeshPhongMaterial(color=colorant"green"))
setobject!(vis["body2"], geom, MeshPhongMaterial(color=colorant"green"))
open(vis)
let x = x0
    x1,x2 = MCTrajOpt.splitstate(twobody, x)
    ri,qi = SA[1,2,3], SA[4,5,6,7]
    r1,r2 = x1[ri], x2[ri]
    q1,q2 = x1[qi], x2[qi]
    settransform!(vis["body1"], compose(Translation(r1), LinearMap(UnitQuaternion(q1))))
    settransform!(vis["body2"], compose(Translation(r2), LinearMap(UnitQuaternion(q2))))
end

##
anim = MeshCat.Animation(floor(Int,1/params.h))
for k = 1:params.N
    atframe(anim, k) do
        let x = X[k] 
            x1,x2 = MCTrajOpt.splitstate(twobody, x)
            ri,qi = SA[1,2,3], SA[4,5,6,7]
            r1,r2 = x1[ri], x2[ri]
            q1,q2 = x1[qi], x2[qi]
            settransform!(vis["body1"], compose(Translation(r1), LinearMap(UnitQuaternion(q1))))
            settransform!(vis["body2"], compose(Translation(r2), LinearMap(UnitQuaternion(q2))))
        end
    end
end
setanimation!(vis, anim)


##