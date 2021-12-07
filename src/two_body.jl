using MCTrajOpt
using ForwardDiff
using StaticArrays
using LinearAlgebra
using Test
using Random

struct RevoluteJoint
    p1::SVector{3,Float64}
    p2::SVector{3,Float64}
    axis::SVector{3,Float64}      # in body1 frame
    orth::SMatrix{2,4,Float64,8}  # in body1 frame
    function RevoluteJoint(p1, p2, axis)
        axis = normalize(axis)
        if axis != SA[0,0,1]
            e0 = SA[0 -1 0; 1 0 0; 0 0 1] * axis  # rotate 90 degrees about z
        else
            e0 = SA[1 0 0; 0 0 -1; 0 1 0] * axis  # rotate 90 degrees about x
        end
        axis = SA_F64[axis[1], axis[2], axis[3]]
        e1 = axis × e0
        e2 = axis × e1
        orth = SA[
            0 e1[1] e1[2] e1[3]
            0 e2[1] e2[2] e2[3]
        ]
        p1 = SA_F64[p1[1], p1[2], p1[3]]
        p2 = SA_F64[p2[1], p2[2], p2[3]]
        new(p1, p2, axis, orth)
    end
end


function joint_constraint(joint::RevoluteJoint, r_1, q_1, r_2, q_2)
    [
        r_1 + Amat(q_1)*joint.p1 - (r_2 + Amat(q_2)*joint.p2);  # joint location 
        joint.orth * L(q_1)'q_2                                 # joint axis
    ]
end

function ∇joint_constraint(joint::RevoluteJoint, r_1, q_1, r_2, q_2)
    Z23 = @SMatrix zeros(2,3)
    dr_1 = [I3; Z23]
    dq_1 = [∇rot(q_1, joint.p1); joint.orth*R(q_2)'Tmat]
    dr_2 = [-I3; Z23]
    dq_2 = [-∇rot(q_2, joint.p2); joint.orth*L(q_1)']
    return dr_1, dq_1, dr_2, dq_2
end

function jtvp_joint_constraint(joint::RevoluteJoint, r_1, q_1, r_2, q_2, λ)
    λt = λ[SA[1,2,3]]
    λa = λ[SA[4,5]]
    dr_1 = λt
    dq_1 = ∇rot(q_1, joint.p1)'λt + Tmat*R(q_2)'joint.orth'λa
    dr_2 = -λt
    dq_2 = -∇rot(q_2, joint.p2)'λt + L(q_1)*joint.orth'λa 
    return dr_1, dq_1, dr_2, dq_2
end

function ∇²joint_constraint(joint::RevoluteJoint, r_1, q_1, r_2, q_2, λ)
    λt = λ[SA[1,2,3]]
    λa = λ[SA[4,5]]
    dq_11 = ∇²rot(q_1, joint.p1, λt)
    dq_12 = Tmat*L(joint.orth'λa)*Tmat
    dq_21 = R(joint.orth'λa)
    dq_22 = -∇²rot(q_2, joint.p2, λt)
    return dq_11, dq_12, dq_21, dq_22
end

function ∇²joint_constraint!(joint::RevoluteJoint, jac, r_1, q_1, r_2, q_2, λ)
    dq_11, dq_12, dq_21, dq_22 = ∇²joint_constraint(joint, r_1, q_1, r_2, q_2, λ)
    iq_1 = 4:7
    iq_2 = iq_1 .+ 7
    jac[iq_1, iq_1] .+= dq_11
    jac[iq_1, iq_2] .+= dq_12
    jac[iq_2, iq_1] .+= dq_21
    jac[iq_2, iq_2] .+= dq_22
    jac
end

function joint_kinematics(joint::RevoluteJoint, r_1, q_1, θ)
    q12 = expm(Amat(q_1)*joint.axis * θ)
    q_2 = L(q12)*q_1
    r_2 = r_1 + Amat(q_1)*joint.p1 - Amat(q_2)*joint.p2
    return r_2, q_2
end

function wrench(joint::RevoluteJoint, x1, x2, u::Real)
    q_1 = x1[SA[4,5,6,7]]
    q_2 = x2[SA[4,5,6,7]]

    q21 = L(q_2)'q_1
    τ1 = -joint.axis * u  # torque on body 1
    τ2 = Amat(q21)*joint.axis * u
    ξ1 = SA[0,0,0, τ1[1], τ1[2], τ1[3]]
    ξ2 = SA[0,0,0, τ2[1], τ2[2], τ2[3]]
    return ξ1, ξ2
end

abstract type TwoBody end
struct SpaceBar <: TwoBody 
    b1::RigidBody
    b2::RigidBody
    joint::RevoluteJoint
end
state_dim(::TwoBody) = 14

function randstate(model::TwoBody)
    r_1 = @SVector randn(3)
    r_2 = @SVector randn(3)
    q_1 = normalize(@SVector randn(4))
    q_2 = normalize(@SVector randn(4))
    return [r_1; q_1; r_2; q_2]
end

function MCTrajOpt.mass_matrix(body::TwoBody)
    M1 = mass_matrix(body.b1)
    M2 = mass_matrix(body.b2)
    blockdiag(M1,M2)
end

function splitstate(model::TwoBody, x)
    inds = SVector{7}(1:7)
    x[inds], x[inds .+ 7]
end

getrind(body::TwoBody, j) = SA[1,2,3] .+ (j-1)*7
getqind(body::TwoBody, j) = SA[4,5,6,7] .+ (j-1)*7

gettran(body::TwoBody, x) = x[SA[1,2,3]], x[SA[8,9,10]]
gettran(body::TwoBody, x, j) = j == 0 ? basetran(body) : x[SA[1,2,3] .+ (j-1)*7]
getquat(body::TwoBody, x) = x[SA[4,5,6,7]], x[SA[11,12,13,14]]
getquat(body::TwoBody, x, j) = j == 0 ? basequat(body) : x[SA[4,5,6,7] .+ (j-1)*7]
getlinvel(body::TwoBody, v, j) = v[SA[1,2,3] .+ (j-1)*6]
getangvel(body::TwoBody, v, j) = v[SA[4,5,6] .+ (j-1)*6]



function splitvel(model::TwoBody, ν)
    inds = SVector{6}(1:6)
    ν[inds], ν[inds .+ 6]
end

function blockdiag(A::StaticMatrix{m1,n1}, B::StaticMatrix{m2,n2}) where {m1,n1,m2,n2}
    A_ = [A @SMatrix zeros(m1, n2)]
    B_ = [(@SMatrix zeros(m2, n1)) B]
    [A_; B_]
end

#############################################
# Kinematics
#############################################
function compose_states(model::TwoBody, x1, x2)
    x11,x12 = splitstate(model, x1)
    x21,x22 = splitstate(model, x2)
    x31 = compose_states(x11, x21)
    x32 = compose_states(x12, x22)
    [x31; x32]
end

function err2fullstate(model::TwoBody, x)
    e1,e2 = splitvel(model, x)
    x1,x2 = err2fullstate(e1), err2fullstate(e2)
    [x1; x2]
end

function errstate_jacobian(model::TwoBody, x)
    x1,x2 = splitstate(model, x)
    G1 = errstate_jacobian(x1)
    G2 = errstate_jacobian(x2)
    blockdiag(G1, G2)
end

function ∇errstate_jacobian(model::TwoBody, x, b)
    x1,x2 = splitstate(model, x)
    b1,b2 = splitstate(model, b)
    G1 = ∇errstate_jacobian(x1, b1)
    G2 = ∇errstate_jacobian(x2, b2)
    blockdiag(G1, G2)
end

function ∇errstate_jacobian2(model::TwoBody, x, b)
    x1,x2 = splitstate(model, x)
    b1,b2 = splitstate(model, b)
    G1 = ∇errstate_jacobian2(x1, b1)
    G2 = ∇errstate_jacobian2(x2, b2)
    blockdiag(G1, G2)
end

function kinematics(model::TwoBody,x,ν)
    x1,x2 = splitstate(model, x)
    ν1,ν2 = splitvel(model, ν)
    x1dot = kinematics(x1,ν1)
    x2dot = kinematics(x2,ν2)
    [x1dot; x2dot]
end

function D1K(model::TwoBody, x,ν)
    x1,x2 = splitstate(model, x)
    ν1,ν2 = splitvel(model, ν)
    D1 = D1K(x1, ν1)
    D2 = D1K(x2, ν2)
    blockdiag(D1,D2)
end

function D2K(model::TwoBody, x,ν)
    x1,x2 = splitstate(model, x)
    ν1,ν2 = splitvel(model, ν)
    D1 = D2K(x1, ν1)
    D2 = D2K(x2, ν2)
    blockdiag(D1,D2)
end

function inv_kinematics(model::TwoBody, x, xdot)
    x1,x2 = splitstate(model, x)
    ẋ1,ẋ2 = splitstate(model, xdot)
    ν1 = inv_kinematics(x1, ẋ1)
    ν2 = inv_kinematics(x2, ẋ2)
    [ν1; ν2]
end

function D1Kinv(model::TwoBody, x, xdot)
    x1,x2 = splitstate(model, x)
    ẋ1,ẋ2 = splitstate(model, xdot)
    D1 = D1Kinv(x1,ẋ1)
    D2 = D1Kinv(x2,ẋ2)
    blockdiag(D1,D2)
end

function D2Kinv(model::TwoBody, x, xdot)
    x1,x2 = splitstate(model, x)
    ẋ1,ẋ2 = splitstate(model, xdot)
    D1 = D2Kinv(x1,ẋ1)
    D2 = D2Kinv(x2,ẋ2)
    blockdiag(D1,D2)
end


function min2max(model::TwoBody, q)
    r_1 = q[SA[1,2,3]]    # base position
    q_1 = q[SA[4,5,6,7]]  # pase orientation
    θ21 = q[8]            # joint angle
    ϕ21 = model.joint.axis  # joint axis in body 1 frame 
    p1,p2 = model.joint.p1, model.joint.p2

    q12 = expm(ϕ21*θ21)   # rotation of body 1 in body 2 frame 
    q_2 = L(q12)*q_1
    r_2 = r_1 + Amat(q_1)*p1 - Amat(q_2)*p2
    return [r_1; q_1; r_2; q_2]
end

#############################################
# Discrete Euler-Lagrange
#############################################

function constraints(model::TwoBody, x)
    p1,p2 = model.joint.p1, model.joint.p2
    x1,x2 = splitstate(model, x)
    ri,qi = SA[1,2,3], SA[4,5,6,7]
    r1,r2 = x1[ri], x2[ri]
    q1,q2 = x1[qi], x2[qi]
    c_pos = r1 + Hmat'R(q1)'*L(q1)*Hmat*p1 - r2 - Hmat'*R(q2)'*L(q2)*Hmat*p2
    c_ori = model.joint.orth*L(q1)'*q2
    [c_pos; c_ori]
end

function ∇constraints(model::TwoBody, x)
    p1,p2 = model.joint.p1, model.joint.p2
    x1,x2 = splitstate(model, x)
    ri,qi = SA[1,2,3], SA[4,5,6,7]
    r1,r2 = x1[ri], x2[ri]
    q1,q2 = x1[qi], x2[qi]
    orth = model.joint.orth
    I3 = SA[1 0 0; 0 1 0; 0 0 1] 
    z23 = @SMatrix zeros(2,3)
    ∇rot1 = Hmat'*(R(q1)'R(Hmat*p1) + L(q1)*L(Hmat*p1)*Tmat)
    ∇rot2 = Hmat'*(R(q2)'R(Hmat*p2) + L(q2)*L(Hmat*p2)*Tmat)
    ∇pos = [I3 ∇rot1 -I3 -∇rot2]
    ∇ori = [z23 orth*R(q2)*Tmat z23 orth*L(q1)']
    [∇pos; ∇ori]
end

function DEL(body, x1, x2, x3, λ, F1, F2, h)
    # scaling = Diagonal(SA[1,1,1,1/2,1/2,1/2,1,1,1,1/2,1/2,1/2])
    errstate_jacobian(body, x2)'*(D2Ld(body,x1,x2,h) + D1Ld(body,x2,x3,h)) + 
        errstate_jacobian(body, x2)'h*∇constraints(body, x2)'λ + h * (F1+F2)/2 
end

function D3_DEL(body, x1,x2,x3,λ, F1,F2, h)
    ForwardDiff.jacobian(x->DEL(body,x1,x2,x,λ, F1,F2,h), x3) * errstate_jacobian(body, x3)
end

#############################################
# Simulation
#############################################

function simulate(model::TwoBody, params::SimParams, F, x0; newton_iters=20, tol=1e-12)
    X = [zero(x0) for k = 1:params.N]
    X[1] = x0
    X[2] = x0
    xi = SVector{12}(1:12)
    yi = SVector{5}(13:17)
    for k = 2:params.N-1
        h = params.h

        # Initial guess
        X[k+1] = X[k]
        λ = @SVector zeros(5)

        for i = 1:newton_iters
            e1 = DEL(model, X[k-1], X[k], X[k+1], λ, F[k-1],F[k], h)
            e2 = constraints(model, X[k+1])
            e = [e1; e2]
            if norm(e, Inf) < tol
                break
            end
            D = D3_DEL(model, X[k-1], X[k], X[k+1], λ, F[k-1],F[k], h)
            C2 = ∇constraints(model, X[k]) * errstate_jacobian(model, X[k])
            C3 = ∇constraints(model, X[k+1]) * errstate_jacobian(model, X[k+1])
            H = [D h*C2']
            H = [H; [C3 @SMatrix zeros(5,5)]]
            Δ = -(H\e)
            Δx = err2fullstate(model, Δ[xi]) 
            X[k+1] = compose_states(model, X[k+1], Δx)
            λ += Δ[yi]

            if i == newton_iters
                @warn "Newton failed to converge within $i iterations at timestep $k"
            end
        end

    end
    return X
end