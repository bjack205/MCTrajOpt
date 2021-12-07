using StaticArrays
using LinearAlgebra
using ForwardDiff
using MCTrajOpt
using MCTrajOpt: L, R, Hmat, G, Tmat
using Rotations
using Plots

function DEL(x1, x2, x3, ξ1, ξ2, h)
    m = 1.0
    J = Diagonal(SA[1,1,1.])
    r1 = x1[SA[1,2,3]]
    r2 = x2[SA[1,2,3]]
    r3 = x3[SA[1,2,3]]
    q1 = x1[SA[4,5,6,7]]
    q2 = x2[SA[4,5,6,7]]
    q3 = x3[SA[4,5,6,7]]
    F1 = ξ1[SA[1,2,3]]
    F2 = ξ2[SA[1,2,3]]
    T1 = ξ1[SA[4,5,6]]
    T2 = ξ2[SA[4,5,6]]
    ξ = h*[(F1+F2)/2; (T1+T2)/2]


    [
        (1/h)*m*(r2-r1) - (1/h)*m*(r3-r2);
        (2.0/h)*G(q2)'*L(q1)*Hmat*J*Hmat'*L(q1)'*q2 + (2.0/h)*G(q2)'*Tmat*R(q3)'*Hmat*J*Hmat'*L(q2)'*q3 
    ] + ξ 
end

function simulate(x0, F, h)
    N = length(F)
    X = [zero(x0) for k = 1:N]
    X[1] = x0
    X[2] = x0
    for k = 2:N-1
        X[k+1] = copy(X[k])
        for i = 1:20
            e = DEL(X[k-1], X[k], X[k+1], F[k-1], F[k], h)
            if norm(e, Inf) < 1e-10
                # println("Converged in $i iterations")
                break
            end
            H = ForwardDiff.jacobian(x3->DEL(X[k-1], X[k], x3, F[k-1], F[k], h), X[k+1]) * MCTrajOpt.errstate_jacobian(X[k+1])
            Δ = -(H\e)
            dx = MCTrajOpt.err2fullstate(Δ)
            X[k+1] = MCTrajOpt.compose_states(X[k+1], dx)
            if i == 20
                @warn "Newton failed to converge in $i iterations"
            end
        end
    end
    return X
end

##
tf = 1.0
h = 0.001
N = round(Int,tf/dt + 1)
x0 = SA[0,0,0, 1,0,0,0.]
F = [SA[0,0,0, 0,0,1.] for k = 1:N]
F[1] = zero(F[1])
X = simulate(x0, F, h)
qf = UnitQuaternion(X[end][4:7]...)
theta = map(X) do x
    q = UnitQuaternion(x[4:7]...)
    AngleAxis(q).theta
end
plot(theta)
AngleAxis(qf).theta

##

let
    f(x,u) = SA[x[2], u[1]]
    times = range(0,1.0, length=101)
    x = SA[0,0]
    u = SA[1.0]
    for k = 1:length(times)-1
        # RK4
        h = times[k+1] - times[k]
        k1 = f(x, u) * h
        k2 = f(x + k1/2, u) * h
        k3 = f(x + k2/2, u) * h
        k4 = f(x + k3, u) * h
        x += (k1 + 2k2 + 2k3 + k4)/6
    end
    x
end

