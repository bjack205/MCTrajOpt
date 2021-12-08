struct DoublePendulum <: TwoBody
    b1::RigidBody
    b2::RigidBody
    joint0::RevoluteJoint
    joint1::RevoluteJoint
    gravity::Float64
end

function DoublePendulum(b1::RigidBody, b2::RigidBody; gravity::Bool=false)
    joint0 = RevoluteJoint(SA[0.0,0,0], SA[0,0,-0.5], SA[0,1,0])
    joint1 = RevoluteJoint(SA[0,0,0.5], SA[0,0,-0.5], SA[0,1,0])
    g = gravity ? 9.81 : 0.0
    DoublePendulum(b1, b2, joint0, joint1, g)
end

basetran(model::DoublePendulum) = SA[0,0,0.]
basequat(model::DoublePendulum) = SA[1,0,0,0.]

function min2max(model::DoublePendulum, q)
    θ1 = q[1]
    θ2 = q[2]
    r_0 = model.joint0.p1
    q_1 = expm(model.joint0.axis*θ1)
    r_1 = r_0 - Amat(q_1)*model.joint0.p2
    q12 = expm(model.joint1.axis*θ2)
    q_2 = L(q12)*q_1
    r_2 = r_1 + Amat(q_1)*model.joint1.p1 - Amat(q_2)*model.joint1.p2
    return [r_1; q_1; r_2; q_2]
end

function kinetic_energy(model::DoublePendulum, x, v)
    T = 0.0
    for j = 1:2
        body = j == 1 ? model.b1 : model.b2
        ν = getlinvel(model, v, j)
        ω = getangvel(model, v, j)
        T += 0.5*(body.mass*ν'ν + ω'body.J*ω)
    end
    return T
    # 0.5 * v'mass_matrix(model)*v
end

function potential_energy(model::DoublePendulum, x)
    U = 0.0
    g = model.gravity
    for j = 1:2
        r = gettran(model, x, j)
        body = j == 1 ? model.b1 : model.b2
        U += g*body.mass*r[3]
    end
    return U
    # r_1, r_2 = gettran(model, x)
    # m1 = model.b1.mass
    # m2 = model.b2.mass
    # g = model.gravity
    # return g*(m1*r_1[3] + m2*r_2[3])
end

function ∇potential_energy(model::DoublePendulum, x)
    m1 = model.b1.mass
    m2 = model.b2.mass
    g = model.gravity
    SA[
        0, 0, m1*g, 0, 0, 0, 0, 0, 0, m2*g, 0, 0, 0, 0
    ]
end

lagrangian(model, x, v) = kinetic_energy(model, x, v) - potential_energy(model, x)
DxL(model, x, v) = -∇potential_energy(model, x)
DvL(model, x, v) = mass_matrix(model)*v

function discretelagrangian(model::DoublePendulum, x1, x2, h)
    x, v = midpoint(model, x1, x2, h)
    h*lagrangian(model, x, v)
end

function midpoint(model::DoublePendulum, x1, x2, h)
    r1_1, r1_2 = gettran(model, x1)
    r2_1, r2_2 = gettran(model, x2)
    r_1, v_1 = midpoint_lin(model.b1, r1_1, r2_1, h)
    r_2, v_2 = midpoint_lin(model.b2, r1_2, r2_2, h)
    # r_1 = (r1_1 + r2_1)/2
    # r_2 = (r1_2 + r2_2)/2
    # v_1 = (r2_1 - r1_1)/h
    # v_2 = (r2_2 - r1_2)/h

    q1_1, q1_2 = getquat(model, x1)
    q2_1, q2_2 = getquat(model, x2)
    q_1, ω_1 = midpoint_rot(model.b1, q1_1, q2_1, h)
    q_2, ω_2 = midpoint_rot(model.b2, q1_2, q2_2, h)
    # q_1 = q1_1
    # q_2 = q1_2
    # ω_1 = 2*Hmat'L(q1_1)'q2_1/h
    # ω_2 = 2*Hmat'L(q1_2)'q2_2/h
    
    x = [r_1; q_1; r_2; q_2]
    v = [v_1; ω_1; v_2; ω_2]
    return x, v
end

function D1midpoint(model, x1, x2, h)
    I3 = SA[1 0 0; 0 1 0; 0 0 1] 
    I4 = SA[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1] 
    q1_1, q1_2 = getquat(model, x1)
    q2_1, q2_2 = getquat(model, x2)

    d1x_1 = blockdiag(I3/2, I4)
    d1x_2 = blockdiag(I3/2, I4)
    d1x = blockdiag(d1x_1, d1x_2)

    d1v_1 = blockdiag(-I3/h, 2*Hmat'R(q2_1)*Tmat/h)
    d1v_2 = blockdiag(-I3/h, 2*Hmat'R(q2_2)*Tmat/h)
    d1v = blockdiag(d1v_1, d1v_2)
    return d1x, d1v
end

function D2midpoint(model, x1, x2, h)
    I3 = SA[1 0 0; 0 1 0; 0 0 1] 
    Z4 = @SMatrix zeros(4,4) 
    q1_1, q1_2 = getquat(model, x1)
    q2_1, q2_2 = getquat(model, x2)

    d1x_1 = blockdiag(I3/2, Z4)
    d1x_2 = blockdiag(I3/2, Z4)
    d1x = blockdiag(d1x_1, d1x_2)

    d1v_1 = blockdiag(I3/h, 2*Hmat'L(q1_1)'/h)
    d1v_2 = blockdiag(I3/h, 2*Hmat'L(q1_2)'/h)
    d1v = blockdiag(d1v_1, d1v_2)
    return d1x, d1v
end

function D1Ld!(model::DoublePendulum, y, x1, x2, h; yi=1)
    ir = (1:3) .+ (yi-1)
    iq = (4:6) .+ (yi-1)
    g = model.gravity
    for j = 1:2
        body = j == 1 ? model.b1 : model.b2
        m, J = body.mass, body.J
        r1, r2 = gettran(model, x1, j), gettran(model, x2, j)
        q1, q2 = getquat(model, x1, j), getquat(model, x2, j)
        
        y[ir] .+= -m/h * (r2 - r1) - h*m*g*SA[0,0,1]/2
        y[iq] .+= 4/h * G(q1)'Tmat*R(q2)'Hmat * J * Hmat'L(q1)'q2
        ir = ir .+ 6
        iq = iq .+ 6
    end
    return y
end

function D1Ld(model::DoublePendulum, x1, x2, h)
    m_1 = model.b1.mass
    m_2 = model.b2.mass
    J_1 = model.b1.J
    J_2 = model.b2.J
    r1_1, r1_2 = gettran(model, x1)
    r2_1, r2_2 = gettran(model, x2)
    q1_1, q1_2 = getquat(model, x1)
    q2_1, q2_2 = getquat(model, x2)
    g = model.gravity
    [
        -m_1/h * (r2_1-r1_1) - h*m_1*g*SA[0,0,1]/2
        4/h * G(q1_1)'Tmat*R(q2_1)'Hmat * J_1 * Hmat'L(q1_1)'q2_1
        -m_2/h * (r2_2-r1_2) - h*m_2*g*SA[0,0,1]/2
        4/h * G(q1_2)'Tmat*R(q2_2)'Hmat * J_2 * Hmat'L(q1_2)'q2_2
    ]
end

function ∇D1Ld(model::DoublePendulum, x1, x2, h)
    m_1 = model.b1.mass
    m_2 = model.b2.mass
    J_1 = model.b1.J
    J_2 = model.b2.J
    q1_1, q1_2 = getquat(model, x1)
    q2_1, q2_2 = getquat(model, x2)

    I3 = SA[1 0 0; 0 1 0; 0 0 1]
    Z34 = @SMatrix zeros(3,3)
    Z33 = @SMatrix zeros(3,3)
    Z37 = @SMatrix zeros(3,6)
    s = 4 
    dq1_1 = s/h * G(q1_1)'Tmat*R(q2_1)'Hmat * J_1 * Hmat'R(q2_1)*Tmat * G(q1_1) + 
        s/h * ∇G2(q1_1, Tmat*R(q2_1)'Hmat * J_1 * Hmat'L(q1_1)'q2_1)
    dq2_1 = s/h * G(q1_1)'Tmat*R(q2_1)'Hmat * J_1 * Hmat'L(q1_1)'G(q2_1) + 
        s/h * G(q1_1)'Tmat*L(Hmat * J_1 * Hmat'L(q1_1)'q2_1)*Tmat*G(q2_1)

    dq1_2 = s/h * G(q1_2)'Tmat*R(q2_2)'Hmat * J_2 * Hmat'R(q2_2)*Tmat*G(q1_2) + 
        s/h * ∇G2(q1_2, Tmat*R(q2_2)'Hmat * J_2 * Hmat'L(q1_2)'q2_2)
    dq2_2 = s/h * G(q1_2)'Tmat*R(q2_2)'Hmat * J_2 * Hmat'L(q1_2)'G(q2_2) + 
        s/h * G(q1_2)'Tmat*L(Hmat * J_2 * Hmat'L(q1_2)'q2_2)*Tmat*G(q2_2)

    d1 = [
        [m_1/h*I3 Z34    Z37            ];
        [Z33       dq1_1  Z37            ];
        [Z37              m_2/h*I3   Z34];
        [Z37               Z33      dq1_2]
    ]
    d2 = [
        [-m_1/h*I3 Z34    Z37            ];
        [Z33       dq2_1  Z37            ];
        [Z37             -m_2/h*I3   Z34];
        [Z37               Z33      dq2_2]
    ]
    return d1, d2
end

function D2Ld!(model::DoublePendulum, y, x1, x2, h; yi=1)
    ir = (1:3) .+ (yi-1)
    iq = (4:6) .+ (yi-1)
    g = model.gravity
    for j = 1:2
        body = j == 1 ? model.b1 : model.b2
        m, J = body.mass, body.J
        r1, r2 = gettran(model, x1, j), gettran(model, x2, j)
        q1, q2 = getquat(model, x1, j), getquat(model, x2, j)
        
        y[ir] .+= m/h * (r2 - r1) - h*m*g*SA[0,0,1]/2
        y[iq] .+= 4/h * G(q2)'L(q1)*Hmat * J * Hmat'L(q1)'q2
        ir = ir .+ 6
        iq = iq .+ 6
    end
    return y
end

function D2Ld(model::DoublePendulum, x1, x2, h)
    m_1 = model.b1.mass
    m_2 = model.b2.mass
    J_1 = model.b1.J
    J_2 = model.b2.J
    r1_1, r1_2 = gettran(model, x1)
    r2_1, r2_2 = gettran(model, x2)
    q1_1, q1_2 = getquat(model, x1)
    q2_1, q2_2 = getquat(model, x2)
    g = model.gravity
    [
        m_1/h * (r2_1-r1_1) - h*m_1*g*SA[0,0,1]/2
        4/h * G(q2_1)'L(q1_1)*Hmat * J_1 * Hmat'L(q1_1)'q2_1
        m_2/h * (r2_2-r1_2) - h*m_2*g*SA[0,0,1]/2
        4/h * G(q2_2)'L(q1_2)*Hmat * J_2 * Hmat'L(q1_2)'q2_2
    ]
end

function ∇D2Ld(model::DoublePendulum, x1, x2, h)
    m_1 = model.b1.mass
    m_2 = model.b2.mass
    J_1 = model.b1.J
    J_2 = model.b2.J
    q1_1, q1_2 = getquat(model, x1)
    q2_1, q2_2 = getquat(model, x2)

    I3 = SA[1 0 0; 0 1 0; 0 0 1]
    Z34 = @SMatrix zeros(3,3)
    Z33 = @SMatrix zeros(3,3)
    Z37 = @SMatrix zeros(3,6)
    s = 4 

    dq1_1 = s/h * G(q2_1)'R(Hmat * J_1 * Hmat'L(q1_1)'q2_1)*G(q1_1) + 
        s/h * G(q2_1)'L(q1_1)*Hmat * J_1 * Hmat'R(q2_1)*Tmat*G(q1_1)
    dq2_1 = s/h * G(q2_1)'L(q1_1)*Hmat * J_1 * Hmat'L(q1_1)'G(q2_1) + 
        s/h * ∇G2(q2_1, L(q1_1)*Hmat * J_1 * Hmat'L(q1_1)'q2_1)

    dq1_2 = s/h * G(q2_2)'R(Hmat * J_2 * Hmat'L(q1_2)'q2_2)*G(q1_2) + 
        s/h * G(q2_2)'L(q1_2)*Hmat * J_2 * Hmat'R(q2_2)*Tmat*G(q1_2)
    dq2_2 = s/h * G(q2_2)'L(q1_2)*Hmat * J_2 * Hmat'L(q1_2)'G(q2_2) + 
        s/h * ∇G2(q2_2, L(q1_2)*Hmat * J_2 * Hmat'L(q1_2)'q2_2)

    d1 = [
        [-m_1/h*I3 Z34    Z37            ];
        [Z33       dq1_1  Z37            ];
        [Z37              -m_2/h*I3   Z34];
        [Z37               Z33      dq1_2]
    ]
    d2 = [
        [m_1/h*I3 Z34    Z37            ];
        [Z33       dq2_1  Z37            ];
        [Z37             m_2/h*I3   Z34];
        [Z37               Z33      dq2_2]
    ]
    return d1, d2
end

function joint_constraints!(model::DoublePendulum, c, x; yi=1)
    p = 5
    for j = 1:2
        joint = j == 1 ? model.joint0 : model.joint1
        ci = (1:p) .+ (j-1)*p .+ (yi-1)
        r_1, r_2 = gettran(model, x, j-1), gettran(model, x, j)
        q_1, q_2 = getquat(model, x, j-1), getquat(model, x, j)
        c[ci] .= joint_constraint(joint, r_1, q_1, r_2, q_2)
    end
    return c
end

function joint_constraints(model::DoublePendulum, x)
    r_1, r_2 = gettran(model, x)
    q_1, q_2 = getquat(model, x)
    q_0 = SA[1,0,0,0]
    joint0 = model.joint0
    joint1 = model.joint1
    [
        joint0.p1 - (r_1 + Amat(q_1)*joint0.p2);
        joint0.orth * L(q_0)'q_1;
        r_1 + Amat(q_1)*joint1.p1 - (r_2 + Amat(q_2)*joint1.p2);  # joint location 
        joint1.orth * L(q_1)'q_2                                  # joint axis
    ]
end

function ∇joint_constraints(model::DoublePendulum, x)
    r_1, r_2 = gettran(model, x)
    q_1, q_2 = getquat(model, x)
    joint0 = model.joint0
    joint1 = model.joint1
    q_0 = SA[1,0,0,0]

    I3 = @SMatrix [1 0 0; 0 1 0; 0 0 1]
    Z23 = @SMatrix zeros(2,3)
    jactran_1 = [-I3 -∇rot(q_1, joint0.p2) @SMatrix zeros(3,7)]
    jacaxis_1 = [Z23 joint0.orth*L(q_0)' @SMatrix zeros(2,7)]
    jactran_2 = [I3 ∇rot(q_1, joint1.p1) -I3 -∇rot(q_2, joint1.p2)]
    jacaxis_2 = [Z23 joint1.orth*R(q_2)*Tmat Z23 joint1.orth*L(q_1)']
    return [jactran_1; jacaxis_1; jactran_2; jacaxis_2]
end

function ∇joint_constraints!(model::DoublePendulum, jac, x;
    ix = 1:14, yi=1
)   
    p = 5
    for j = 1:2
        joint = j == 1 ? model.joint0 : model.joint1
        ci = (1:p) .+ (j-1)*p .+ (yi-1)
        r_1, r_2 = gettran(model, x, j-1), gettran(model, x, j)
        q_1, q_2 = getquat(model, x, j-1), getquat(model, x, j)
        ir_1 = (1:3) .+ (j-2)*7 .+ (ix[1] - 1)  # rind[k,j-1]
        iq_1 = (4:7) .+ (j-2)*7 .+ (ix[1] - 1)  
        ir_2 = (1:3) .+ (j-1)*7 .+ (ix[1] - 1)  # rind[k,j] 
        iq_2 = (4:7) .+ (j-1)*7 .+ (ix[1] - 1)

        dr_1, dq_1, dr_2, dq_2 = ∇joint_constraint(joint, r_1, q_1, r_2, q_2)
        # println("ci = $(ci), xinds = $(ir_1[1]):$(iq_2[end])")
        if (ir_1[1] - (ix[1]-1)) > 0
            # println("processed prev state")
            @view(jac[ci, ir_1]) .+= dr_1
            @view(jac[ci, iq_1]) .+= dq_1
        end
        @view(jac[ci, ir_2]) .+= dr_2
        @view(jac[ci, iq_2]) .+= dq_2
    end
    return jac 
end

function jtvp_joint_constraints!(model::DoublePendulum, y, x, λ; yi=1)
    ir_1 = (1:3) .- 6 .+ (yi-1)
    iq_1 = (4:6) .- 6 .+ (yi-1)
    p = 5
    iλt = SA[1,2,3]
    iλa = SA[4,5]
    for j = 1:2
        ir_2 = ir_1 .+ 6
        iq_2 = iq_1 .+ 6
        ci = (1:p) .+ (j-1)*p
        joint = j == 1 ? model.joint0 : model.joint1
        r_1, r_2 = gettran(model, x, j-1), gettran(model, x, j)
        q_1, q_2 = getquat(model, x, j-1), getquat(model, x, j)
        λj = view(λ, ci)

        dr_1, dq_1, dr_2, dq_2 = jtvp_joint_constraint(joint, r_1, q_1, r_2, q_2, λj)
        if (ir_1[1] - (yi-1)) > 0 
            y[ir_1] .+= dr_1
            y[iq_1] .+= G(q_1)'dq_1
        end
        y[ir_2] .+= dr_2
        y[iq_2] .+= G(q_2)'dq_2

        ir_1 = ir_1 .+ 6
        iq_1 = iq_1 .+ 6
    end
end

function jtvp_joint_constraints(model::DoublePendulum, x, λ)
    r_1, r_2 = gettran(model, x)
    q_1, q_2 = getquat(model, x)
    λt_1, λa_1 = λ[SA[1,2,3]], λ[SA[4,5]]
    λt_2, λa_2 = λ[SA[6,7,8]], λ[SA[9,10]]
    joint0 = model.joint0
    joint1 = model.joint1
    q_0 = SA[1,0,0,0]

    dr_1 = -λt_1 + λt_2 
    dq_1 = -∇rot(q_1, joint0.p2)'λt_1 + L(q_0)'joint0.orth'λa_1 + 
            ∇rot(q_1, joint1.p1)'λt_2 + Tmat*R(q_2)'joint1.orth'λa_2
    dr_2 = -λt_2
    dq_2 = -∇rot(q_2, joint1.p2)'λt_2 + L(q_1)*joint1.orth'λa_2
    return [dr_1; dq_1; dr_2; dq_2]
end

function ∇²joint_constraints!(model::DoublePendulum, hess, x, λ; errstate=Val(false))
    ir_1 = (1:3) .- 6
    iϕ_1 = (4:6) .- 6
    p = 5
    for j = 1:2
        ir_2 = ir_1 .+ 6
        iϕ_2 = iϕ_1 .+ 6
        if errstate == Val(true)
            iq_1 = iϕ_1
            iq_2 = iϕ_2
        else
            iq_1 = (4:7) .+ (j-2)*7
            iq_2 = (4:7) .+ (j-1)*7
        end

        ci = (1:p) .+ (j-1)*p
        joint = j == 1 ? model.joint0 : model.joint1
        r_1, r_2 = gettran(model, x, j-1), gettran(model, x, j)
        q_1, q_2 = getquat(model, x, j-1), getquat(model, x, j)
        λj = view(λ, ci)

        dq_11, dq_12, dq_21, dq_22 = ∇²joint_constraint(joint, r_1, q_1, r_2, q_2, λj)
        dr_1, dq_1, dr_2, dq_2 = jtvp_joint_constraint(joint, r_1, q_1, r_2, q_2, λj)
        if ir_1[1] > 0 
            hess[iϕ_1, iq_1] .+= ∇²err(dq_11, dq_1, q_1, q_1, errstate) 
            hess[iϕ_1, iq_2] .+= ∇²err(dq_12, dq_1, q_1, q_2, errstate) 
            hess[iϕ_2, iq_1] .+= ∇²err(dq_21, dq_2, q_2, q_1, errstate) 
        end
        hess[iϕ_2, iq_2] .+= ∇²err(dq_22, dq_2, q_2, q_2, errstate) 

        ir_1 = ir_1 .+ 6
        iϕ_1 = iϕ_1 .+ 6
    end
    hess
end

function ∇²joint_constraints(model::DoublePendulum, x, λ)
    r_1, r_2 = gettran(model, x)
    q_1, q_2 = getquat(model, x)
    λt_1, λa_1 = λ[SA[1,2,3]], λ[SA[4,5]]
    λt_2, λa_2 = λ[SA[6,7,8]], λ[SA[9,10]]
    joint0 = model.joint0
    joint1 = model.joint1
    q_0 = SA[1,0,0,0]

    Z43 = @SMatrix zeros(4,3)
    dr_1 = @SMatrix zeros(3,14)
    dq_1 = [Z43 -∇²rot(q_1, joint0.p2, λt_1) + ∇²rot(q_1, joint1.p1, λt_2) Z43 Tmat*L(joint1.orth'λa_2)*Tmat]
    dr_2 = @SMatrix zeros(3,14)
    dq_2 = [Z43 R(joint1.orth'λa_2) Z43 -∇²rot(q_2, joint1.p2, λt_2)]
    [dr_1; dq_1; dr_2; dq_2]
end

function DEL(model::DoublePendulum, x1, x2, x3, λ, F1, F2, h)
    D2Ld(model, x1, x2, h) + D1Ld(model, x2, x3, h) + h*(F1 + F2)/2 + 
        h*errstate_jacobian(model, x2)'∇joint_constraints(model, x2)'λ
end

function DEL!(model::DoublePendulum, y, x1, x2, x3, λ, F1, F2, h; yi=1)
    yview = view(y, (1:12) .+ (yi-1))
    yview .= 0
    jtvp_joint_constraints!(model, y, x2, λ, yi=yi)
    yview .*= h
    yview .+= h*(F1 + F2)/2
    D2Ld!(model, y, x1, x2, h, yi=yi)
    D1Ld!(model, y, x2, x3, h, yi=yi)
    return

    nL = 2
    yview = view(y, (1:6*nL) .+ (yi-1))
    @. yview = h*(F1 + F2)/2

    ir = (1:3) .+ (yi-1)
    iq = (4:6) .+ (yi-1)
    p = 5
    g = model.gravity
    for j = 1:nL
        body = j == 1 ? model.b1 : model.b2
        m, J = body.mass, body.J
        r1, r2, r3, = gettran(model, x1, j), gettran(model, x2, j), gettran(model, x3, j)
        q1, q2, q3, = getquat(model, x1, j), getquat(model, x2, j), getquat(model, x3, j)
        
        # D1Ld
        y[ir] .+= -m/h * (r3 - r2) - h*m*g*SA[0,0,1]/2
        y[iq] .+= 4/h * G(q2)'Tmat*R(q3)'Hmat * J * Hmat'L(q2)'q3

        # D2Ld
        y[ir] .+= m/h * (r2 - r1) - h*m*g*SA[0,0,1]/2
        y[iq] .+= 4/h * G(q2)'L(q1)*Hmat * J * Hmat'L(q1)'q2

        # @goto loopctr
        # Joint constraints
        joint = j == 1 ? model.joint0 : model.joint1
        ir_1 = ir .- 6  # assumes DEL constraint is consecutive
        iq_1 = iq .- 6
        ir_2 = ir
        iq_2 = iq
        ci = (1:p) .+ (j-1)*p 

        r_1, r_2 = gettran(model, x2, j-1), gettran(model, x2, j)
        q_1, q_2 = getquat(model, x2, j-1), getquat(model, x2, j)

        λj = view(λ, ci)
        dr_1, dq_1, dr_2, dq_2 = jtvp_joint_constraint(joint, r_1, q_1, r_2, q_2, λj)
        if (ir_1[1] - yi) > 0 
            y[ir_1] .+= h*dr_1
            y[iq_1] .+= h*G(q_1)'dq_1
        end
        y[ir_2] .+= h*dr_2
        y[iq_2] .+= h*G(q_2)'dq_2

        @label loopctr
        ir = ir .+ 6
        iq = iq .+ 6
    end
end

function ∇DEL!(model::DoublePendulum, jac, x1, x2, x3, λ, F1, F2, h; 
    ix1 = 1:14, ix2 = ix1 .+ 16, ix3 = ix2 .+ 16, yi=1
)
    # @. y = h*(F1 + F2)/2
    # D2Ld!(model, y, x1, x2, h)
    # D1Ld!(model, y, x2, x3, h)
    # jtvp_joint_constraints!(model, y, x2, λ)
    # Jcon = errstate_jacobian(model, x2)'∇²joint_constraints(model, x2, λ) + 
    #     ∇errstate_jacobian(model, x2, jtvp_joint_constraints(model, x2, λ))
    # jac_x2 = view(jac, :, ix2)
    # ∇²joint_constraints!(model, jac_x2, x2, λ, errstate=Val(false))
    # jac_x2 .*= h

    # Joint Constraints
    hess = jac
    x = x2
    errstate = Val(false)
    ir_1 = (1:3) .- 6 .+ (yi-1)
    iϕ_1 = (4:6) .- 6 .+ (yi-1)
    p = 5
    for j = 1:2
        ir_2 = ir_1 .+ 6
        iϕ_2 = iϕ_1 .+ 6
        if errstate == Val(true)
            iq_1 = iϕ_1
            iq_2 = iϕ_2
        else
            iq_1 = (4:7) .+ (j-2)*7 .+ (ix2[1] - 1)
            iq_2 = (4:7) .+ (j-1)*7 .+ (ix2[1] - 1)
        end

        ci = (1:p) .+ (j-1)*p
        joint = j == 1 ? model.joint0 : model.joint1
        r_1, r_2 = gettran(model, x, j-1), gettran(model, x, j)
        q_1, q_2 = getquat(model, x, j-1), getquat(model, x, j)
        λj = view(λ, ci)

        dq_11, dq_12, dq_21, dq_22 = ∇²joint_constraint(joint, r_1, q_1, r_2, q_2, λj)
        dr_1, dq_1, dr_2, dq_2 = jtvp_joint_constraint(joint, r_1, q_1, r_2, q_2, λj)
        if (ir_1[1] - (yi-1)) > 0 
            @view(hess[iϕ_1, iq_1]) .+= ∇²err(dq_11, dq_1, q_1, q_1, errstate) * h
            @view(hess[iϕ_1, iq_2]) .+= ∇²err(dq_12, dq_1, q_1, q_2, errstate) * h
            @view(hess[iϕ_2, iq_1]) .+= ∇²err(dq_21, dq_2, q_2, q_1, errstate) * h
        end
        @view(hess[iϕ_2, iq_2]) .+= ∇²err(dq_22, dq_2, q_2, q_2, errstate) * h

        ir_1 = ir_1 .+ 6
        iϕ_1 = iϕ_1 .+ 6
    end
    # jac_x2 = view(jac, :, ix2)
    # jac_x2 .*= h


    # Discrete Legendre Transforms
    ir = (1:3) .+ (yi-1)
    iq = (4:6) .+ (yi-1)
    p = 5
    g = model.gravity
    for j = 1:2
        body = j == 1 ? model.b1 : model.b2
        m, J = body.mass, body.J
        r1, r2, r3,   = gettran(model, x1, j), gettran(model, x2, j), gettran(model, x3, j)
        q1, q2, q3,   = getquat(model, x1, j), getquat(model, x2, j), getquat(model, x3, j)
        ir1, iq1 = ix1[getrind(model, j)], ix1[getqind(model, j)]
        ir2, iq2 = ix2[getrind(model, j)], ix2[getqind(model, j)]
        ir3, iq3 = ix3[getrind(model, j)], ix3[getqind(model, j)]
        
        # D1Ld
        @view(jac[ir, ir2]) .+= +m/h * I3
        @view(jac[ir, ir3]) .+= -m/h * I3
        @view(jac[iq, iq2]) .+= 4/h * G(q2)'Tmat*R(q3)'Hmat * J * Hmat'R(q3)*Tmat + 
            4/h * ∇G(q2, Tmat*R(q3)'Hmat * J * Hmat'L(q2)'q3)
        @view(jac[iq, iq3]) .+= 4/h * G(q2)'Tmat*R(q3)'Hmat * J * Hmat'L(q2)' + 
            4/h * G(q2)'Tmat*L(Hmat * J * Hmat'L(q2)'q3) * Tmat

        # D2Ld
        @view(jac[ir, ir1]) .+= -m/h * I3
        @view(jac[ir, ir2]) .+= +m/h * I3
        @view(jac[iq, iq1]) .+= 4/h * G(q2)'R(Hmat * J * Hmat'L(q1)'q2) + 
            4/h * G(q2)'L(q1)*Hmat * J * Hmat'R(q2)*Tmat
        @view(jac[iq, iq2]) .+= 4/h * G(q2)'L(q1)*Hmat * J * Hmat'L(q1)' + 
            4/h * ∇G(q2, L(q1)*Hmat * J * Hmat'L(q1)'q2)

        ir = ir .+ 6
        iq = iq .+ 6
    end

    return
end

function ∇DEL(model::DoublePendulum, x1, x2, x3, λ, F1, F2,h)
    dx1_2, dx2_2 = ∇D2Ld(model, x1, x2, h)
    dx2_1, dx3_1 = ∇D1Ld(model, x2, x3, h)
    
    G2 = errstate_jacobian(model, x2)
    dx2_c = G2'∇²joint_constraints(model, x2, λ2)*G2 + 
        h*∇errstate_jacobian2(model, x2, jtvp_joint_constraints(model, x2, λ))

    dF1 = h*I/2
    dF2 = h*I/2
    return dx1_2, dx2_2 + dx2_1 + dx2_c, dx3_1, dF1, dF2
end

function ∇DEL3(model::DoublePendulum, x1, x2, x3, λ, F1, F2, h)
    dx2_1, dx3_1 = ∇D1Ld(model, x2, x3, h)
    return dx3_1
end

function getwrenches(model::DoublePendulum, x, u)
    x_0 = [model.joint0.p1; SA[1,0,0,0]]
    x_1, x_2 = splitstate(model, x)
    ξ0_0, ξ0_1 = wrench(model.joint0, x_0, x_1, u[1])
    ξ1_1, ξ1_2 = wrench(model.joint1, x_1, x_2, u[2])
    return [ξ0_1 + ξ1_1; ξ1_2]
end

function getwrenches!(model::DoublePendulum, ξ, x, u; yi=1)
    for j = 1:2
        joint = j == 1 ? model.joint0 : model.joint1
        r_1, r_2 = gettran(model, x, j-1), gettran(model, x, j)
        q_1, q_2 = getquat(model, x, j-1), getquat(model, x, j)
        iF_1 = (1:3) .+ (j-2)*6  .+ (yi-1)
        iT_1 = (4:6) .+ (j-2)*6  .+ (yi-1)
        iF_2 = (1:3) .+ (j-1)*6  .+ (yi-1)
        iT_2 = (4:6) .+ (j-1)*6  .+ (yi-1)

        F_1, T_1 = wrench1(joint, r_1, q_1, r_2, q_2, u[j]) 
        F_2, T_2 = wrench2(joint, r_1, q_1, r_2, q_2, u[j]) 
        if (iF_1[1] - (yi-1)) > 0
            ξ[iF_1] .+= F_1
            ξ[iT_1] .+= T_1
        end
        ξ[iF_2] .+= F_2
        ξ[iT_2] .+= T_2
    end
    return ξ
end

function ∇getwrenches!(model::DoublePendulum, jac, x, u; ix=1:14, iu=15:16, yi=1, s=1.0)
    for j = 1:2
        joint = j == 1 ? model.joint0 : model.joint1
        iu_j = iu[j]:iu[j]
        r_1, r_2 = gettran(model, x, j-1), gettran(model, x, j)
        q_1, q_2 = getquat(model, x, j-1), getquat(model, x, j)
        iF_1 = (1:3) .+ (j-2)*6 .+ (yi-1)
        iT_1 = (4:6) .+ (j-2)*6 .+ (yi-1)
        iF_2 = (1:3) .+ (j-1)*6 .+ (yi-1)
        iT_2 = (4:6) .+ (j-1)*6 .+ (yi-1)

        ir_1 = (1:3) .+ (j-2)*7 .+ (ix[1] - 1)
        iq_1 = (4:7) .+ (j-2)*7 .+ (ix[1] - 1)
        ir_2 = (1:3) .+ (j-1)*7 .+ (ix[1] - 1)
        iq_2 = (4:7) .+ (j-1)*7 .+ (ix[1] - 1)

        if (iF_1[1] - (yi-1)) > 0
            println("Past 1st row")
            F_11 = ∇force11(joint, r_1, q_1, r_2, q_2, u[j])
            @view(jac[iF_1, ir_1]) .+= F_11[1] * s
            @view(jac[iF_1, iq_1]) .+= F_11[2] * s

            F_12 = ∇force12(joint, r_1, q_1, r_2, q_2, u[j])
            @view(jac[iF_1, ir_2]) .+= F_12[1] * s
            @view(jac[iF_1, iq_2]) .+= F_12[2] * s

            F_1u = ∇force1u(joint, r_1, q_1, r_2, q_2, u[j])
            @view(jac[iF_1, iu_j]) .+= F_1u * s

            T_11 = ∇torque11(joint, r_1, q_1, r_2, q_2, u[j])
            @view(jac[iT_1, ir_1]) .+= T_11[1] * s
            @view(jac[iT_1, iq_1]) .+= T_11[2] * s

            T_12 = ∇torque12(joint, r_1, q_1, r_2, q_2, u[j])
            @view(jac[iT_1, ir_2]) .+= T_12[1] * s
            @view(jac[iT_1, iq_2]) .+= T_12[2] * s

            T_1u = ∇torque1u(joint, r_1, q_1, r_2, q_2, u[j])
            @view(jac[iT_1, iu_j]) .+= T_1u * s
        end
        if (ir_1[1] - (ix[1] - 1)) > 0
            println("Past 1st column")
            F_21 = ∇force21(joint, r_1, q_1, r_2, q_2, u[j])
            @view(jac[iF_2, ir_1]) .+= F_21[1] * s
            @view(jac[iF_2, iq_1]) .+= F_21[2] * s

            T_21 = ∇torque21(joint, r_1, q_1, r_2, q_2, u[j])
            @view(jac[iT_2, ir_1]) .+= T_21[1] * s
            @view(jac[iT_2, iq_1]) .+= T_21[2] * s
        end
        F_22 = ∇force22(joint, r_1, q_1, r_2, q_2, u[j])
        @view(jac[iF_2, ir_2]) .+= F_22[1] * s
        @view(jac[iF_2, iq_2]) .+= F_22[2] * s

        F_2u = ∇force2u(joint, r_1, q_1, r_2, q_2, u[j])
        @view(jac[iF_2, iu_j]) .+= F_2u * s

        T_22 = ∇torque22(joint, r_1, q_1, r_2, q_2, u[j])
        @view(jac[iT_2, ir_2]) .+= T_22[1] * s
        @view(jac[iT_2, iq_2]) .+= T_22[2] * s

        T_2u = ∇torque2u(joint, r_1, q_1, r_2, q_2, u[j])
        @view(jac[iT_2, iu_j]) .+= T_2u * s
    end
    return jac
end

function simulate(model::DoublePendulum, params::SimParams, U, x0; newton_iters=20, tol=1e-12)
    X = [zero(x0) for k = 1:params.N]
    X[1] = x0
    X[2] = x0
    xi = SVector{12}(1:12)
    yi = SVector{10}(13:22)

    for k = 2:params.N-1
        h = params.h

        # Initial guess
        X[k+1] = X[k]
        λ = @SVector zeros(10)

        for i = 1:newton_iters
            F1 = getwrenches(model, X[k-1], U[k-1])
            F2 = getwrenches(model, X[k], U[k])

            e1 = DEL(model, X[k-1], X[k], X[k+1], λ, F1,F2, h)
            e2 = joint_constraints(model, X[k+1])
            e = [e1; e2]
            if norm(e, Inf) < tol
                # println("Converged in $i iters at time $(params.thist[k])")
                break
            end
            # D = ∇DEL3(model, X[k-1], X[k], X[k+1], λ, F[k-1],F[k], h)
            D = ForwardDiff.jacobian(x3->DEL(model, X[k-1], X[k], x3, λ, F1, F2, h), X[k+1]) * errstate_jacobian(model, X[k+1])
            C2 = ForwardDiff.jacobian(x->joint_constraints(model, x), X[k]) * errstate_jacobian(model, X[k])
            C3 = ForwardDiff.jacobian(x->joint_constraints(model, x), X[k]) * errstate_jacobian(model, X[k+1])
            # C2 = ∇joint_constraints(model, X[k]) * errstate_jacobian(model, X[k])
            # C3 = ∇joint_constraints(model, X[k+1]) * errstate_jacobian(model, X[k+1])
            H = [D h*C2']
            H = [H; [C3 @SMatrix zeros(10,10)]]
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