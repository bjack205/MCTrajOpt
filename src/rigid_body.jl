struct RigidBody
    mass::Float64
    J::SMatrix{3,3,Float64,9}
end

mass_matrix(body::RigidBody) = SA[
    body.mass 0 0 0 0 0
    0 body.mass 0 0 0 0
    0 0 body.mass 0 0 0
    0 0 0 body.J[1,1] body.J[1,2] body.J[1,3]
    0 0 0 body.J[2,1] body.J[2,2] body.J[2,3]
    0 0 0 body.J[3,1] body.J[3,2] body.J[3,3]
]

state_dim(::RigidBody) = 7

mutable struct SimParams
    h::Float64   # time step (sec)
    tf::Float64  # total time (sec)
    N::Int       # number of time steps
    thist::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}}
    function SimParams(tf::Float64, h::Float64)
        thist = range(0, tf, step=h)
        N = length(thist)
        new(h, tf, N, thist)
    end
end

#############################################
# Kinematics 
#############################################

for method in (:compose_states, :kinematics, :D1K, :D2K, :inv_kinematics, :D1Kinv, :D2Kinv)
    @eval @inline $method(body::RigidBody, x, ν) = $method(x, ν)
end
@inline errstate_jacobian(body::RigidBody, x)  = errstate_jacobian(x)
@inline err2fullstate(model::RigidBody, x) = err2fullstate(x)

#############################################
# Lagrangian
#############################################
function Lagrangian_vel(body, x, ν)
    M = mass_matrix(body)
    T = 1/2 * ν'M*ν
    U = 0.0
    return T - U
end

function Lagrangian_dot(body, x, xdot)
    ν = inv_kinematics(body, x, xdot)
    Lagrangian_vel(body, x, ν)
end

D1L_vel(body, x, ν) = @SVector zeros(state_dim(body))
D2L_vel(body, x, ν) = mass_matrix(body) * ν 

D1L_dot(body, x, xdot) = D1L_vel(body, x, inv_kinematics(body, x, xdot)) + 
    D1Kinv(body, x, xdot)'D2L_vel(body, x, inv_kinematics(body, x, xdot))
D2L_dot(body, x, xdot) = D2Kinv(body, x, xdot)'D2L_vel(body, x, inv_kinematics(body, x, xdot))

#############################################
# Discrete Lagrangian
#############################################
function Ld(body, x1, x2, h)
    h * Lagrangian_dot(body, (x1+x2)/2, (x2-x1)/h)
end

function D1Ld(body, x1, x2, h)
    xmid = (x1 + x2)/2
    ẋmid = (x2 - x1)/h
    h/2 * D1L_dot(body,xmid,ẋmid) - D2L_dot(body,xmid,ẋmid)
end

function D2Ld(body, x1, x2, h)
    xmid = (x1 + x2)/2
    ẋmid = (x2 - x1)/h
    h/2 * D1L_dot(body,xmid,ẋmid) + D2L_dot(body,xmid,ẋmid)
end

#############################################
# Discrete Euler-Lagrange
#############################################

function DEL(body, x1, x2, x3, F1, F2, h)
    # scaling = Diagonal(SA[1,1,1,0.5,0.5,0.5])
    # scaling * errstate_jacobian(body, x2)'*(D2Ld(body,x1,x2,h) + D1Ld(body,x2,x3,h)) + h * (F1+F2)/2
    mass = body.mass
    J = body.J
    ri = SA[1,2,3]
    qi = SA[4,5,6,7]
    r1,q1 = x1[ri], x1[qi]
    r2,q2 = x2[ri], x2[qi]
    r3,q3 = x3[ri], x3[qi]
    G2 = L(q2)*Hmat
    DELr = mass*(r2-r1)/h - mass*(r3-r2)*h
    DELq = (2/h) * G2'L(q1)*Hmat*J * Hmat'L(q1)'q2 + (2/h) * G2'Tmat*R(q3)'Hmat*J*Hmat'L(q2)'q3
    [DELr; DELq] + h*(F1+F2)/2
end

function D3_DEL(body, x1,x2,x3, F1,F2, h)
    ForwardDiff.jacobian(x->DEL(body,x1,x2,x,F1,F2,h), x3) * errstate_jacobian(body, x3)
end


#############################################
# Simulation
#############################################

function simulate(body::RigidBody, params::SimParams, F, x0; newton_iters=20, tol=1e-12)
    X = [zero(x0) for k = 1:params.N]
    X[1] = x0
    X[2] = x0
    for k = 2:params.N-1
        h = params.h

        # Initial guess
        X[k+1] = X[k]

        for i = 1:newton_iters
            e = DEL(body, X[k-1], X[k], X[k+1], F[k-1],F[k], h)
            if norm(e, Inf) < tol
                break
            end
            H = D3_DEL(body, X[k-1], X[k], X[k+1], F[k-1],F[k], h)
            Δ = -(H\e)
            Δr = Δ[SA[1,2,3]]
            ϕ = Δ[SA[4,5,6]]  # delta rotation
            Δq = SA[sqrt(1-ϕ'ϕ), ϕ[1], ϕ[2], ϕ[3]]
            Δx = [Δr; Δq]
            X[k+1] = compose_states(X[k+1], Δx)

            if i == newton_iters
                @warn "Newton failed to converge within $i iterations at timestep $k"
            end
        end

    end
    return X
end