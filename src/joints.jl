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

numconstraints(::RevoluteJoint) = 5

function joint_constraint(joint::RevoluteJoint, r_1, q_1, r_2, q_2)
    [
        r_1 + Amat(q_1)*joint.p1 - (r_2 + Amat(q_2)*joint.p2);  # joint location 
        joint.orth * L(q_1)'q_2                                 # joint axis
    ]
end

function ∇joint_constraint(joint::RevoluteJoint, r_1, q_1, r_2, q_2)
    Z23 = @SMatrix zeros(2,3)
    dr_1 = [I3; Z23]
    dq_1 = [∇rot(q_1, joint.p1); joint.orth*R(q_2)*Tmat]
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
    q12 = expm2(Amat(q_1)*joint.axis * θ)
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

force1(joint::RevoluteJoint, r_1, q_1, r_2, q_2, u::Real) = SA_F64[0,0,0]
force2(joint::RevoluteJoint, r_1, q_1, r_2, q_2, u::Real) = SA_F64[0,0,0]

torque1(joint::RevoluteJoint, r_1, q_1, r_2, q_2, u::Real) = -joint.axis * u 
torque2(joint::RevoluteJoint, r_1, q_1, r_2, q_2, u::Real) = Amat(L(q_2)'q_1)*joint.axis * u 

∇force11(joint::RevoluteJoint, r_1, q_1, r_2, q_2, u::Real) = ((@SMatrix zeros(3,3)), @SMatrix zeros(3,4))
∇force12(joint::RevoluteJoint, r_1, q_1, r_2, q_2, u::Real) = ((@SMatrix zeros(3,3)), @SMatrix zeros(3,4))
∇force1u(joint::RevoluteJoint, r_1, q_1, r_2, q_2, u::Real) = @SMatrix zeros(3,1) 

∇force21(joint::RevoluteJoint, r_1, q_1, r_2, q_2, u::Real) = ((@SMatrix zeros(3,3)), @SMatrix zeros(3,4))
∇force22(joint::RevoluteJoint, r_1, q_1, r_2, q_2, u::Real) = ((@SMatrix zeros(3,3)), @SMatrix zeros(3,4))
∇force2u(joint::RevoluteJoint, r_1, q_1, r_2, q_2, u::Real) = @SMatrix zeros(3,1) 

∇torque11(joint::RevoluteJoint, r_1, q_1, r_2, q_2, u::Real) = ((@SMatrix zeros(3,3)), @SMatrix zeros(3,4))
∇torque12(joint::RevoluteJoint, r_1, q_1, r_2, q_2, u::Real) = ((@SMatrix zeros(3,3)), @SMatrix zeros(3,4))
∇torque1u(joint::RevoluteJoint, r_1, q_1, r_2, q_2, u::Real) = -joint.axis 

function ∇torque21(joint::RevoluteJoint, r_1, q_1, r_2, q_2, u::Real)
    dr = @SMatrix zeros(3,3)
    q21 = L(q_2)'q_1
    dq = ∇rot(q21, joint.axis * u) * L(q_2)'
    return dr, dq
end
function ∇torque22(joint::RevoluteJoint, r_1, q_1, r_2, q_2, u::Real)
    dr = @SMatrix zeros(3,3)
    q21 = L(q_2)'q_1
    dq = ∇rot(q21, joint.axis * u) * R(q_1) * Tmat
    return dr, dq
end
∇torque2u(joint::RevoluteJoint, r_1, q_1, r_2, q_2, u::Real) = Amat(L(q_2)'q_1) * joint.axis

wrench1(joint::RevoluteJoint, r_1, q_1, r_2, q_2, u::Real) = 
    force1(joint, r_1, q_2, r_2, q_2, u), torque1(joint, r_1, q_1, r_2, q_2, u)
wrench2(joint::RevoluteJoint, r_1, q_1, r_2, q_2, u::Real) = 
    force2(joint, r_1, q_2, r_2, q_2, u), torque2(joint, r_1, q_1, r_2, q_2, u)
