struct RobotArm
    geom::Vector{CylindricalBody}
    links::Vector{RigidBody} 
    joints::Vector{RevoluteJoint}
    gravity::Float64
    numlinks::Int
    slink::Int  # spacing between links
    stime::Int  # spacing between time steps
    function RobotArm(geom::Vector{CylindricalBody}, joints::Vector{RevoluteJoint}; gravity::Bool=false)
        @assert length(geom) == length(joints)
        links = RigidBody.(geom)
        numlinks = length(links)
        g = gravity ? 9.81 : 0.0
        slink = 1
        stime = numlinks
        new(geom, links, joints, g, numlinks, slink, stime)
    end
end

numconstraints(arm::RobotArm) = sum(numconstraints.(arm.joints))

# State extraction
basetran(model::RobotArm) = SA[0,0,0.]
basequat(model::RobotArm) = SA[1,0,0,0.]

getrind(body::RobotArm, j, k=1) = SA[1,2,3] .+ (j-1)*7*body.slink .+ (k-1)*7*body.stime
getqind(body::RobotArm, j, k=1) = SA[4,5,6,7] .+ (j-1)*7*body.slink .+ (k-1)*7*body.stime

gettran(body::RobotArm, x, j, k=1) = j == 0 ? basetran(body) : x[SVector{3}(getrind(body, j, k))]
getquat(body::RobotArm, x, j, k=1) = j == 0 ? basequat(body) : x[SVector{4}(getqind(body, j, k))]

getlinvel(body::RobotArm, v, j, k=1) = v[SA[1,2,3] .+ (j-1)*6*body.slink .+ (k-1)*6*body.stime]
getangvel(body::RobotArm, v, j, k=1) = v[SA[4,5,6] .+ (j-1)*6*body.slink .+ (k-1)*6*body.stime]

function compose_states!(model::RobotArm, x3, x1, x2)
    for j = 1:model.numlinks
        ri = getrind(model, j)
        qi = getqind(model, j)
        r1 = gettran(model, x1, j)
        r2 = gettran(model, x2, j)
        q1 = getquat(model, x1, j)
        q2 = getquat(model, x2, j)

        x3[ri] = r1 + r2
        x3[qi] = L(q1)*q2
    end
    return x3
end

function err2fullstate!(model::RobotArm, x, e)
    for j = 1:model.numlinks
        ri = getrind(model, j)
        qi = getqind(model, j)
        dr = getlinvel(model, e, j)
        dϕ = getangvel(model, e, j)
        x[ri] = dr 
        x[qi] = cayleymap(dϕ)
    end
    return x
end

function min2max(model::RobotArm, θ)
    r_0 = basetran(model)
    q_0 = basequat(model)
    r_prev = r_0
    q_prev = q_0
    r = [copy(r_0) for j = 1:model.numlinks+1]  # includes base link
    q = [copy(q_0) for j = 1:model.numlinks+1]  # includes base link
    x = zeros(model.numlinks * 7)
    for j = 1:model.numlinks
        r, q = joint_kinematics(model.joints[j], r_prev, q_prev, θ[j])
        x[getrind(model, j)] = r
        x[getqind(model, j)] = q
        r_prev, q_prev = r, q
    end
    # return r, q
    return x
end

function kinetic_energy(model::RobotArm, x, v)
    T = 0.0
    for j = 1:model.numlinks
        body = model.links[j]
        ν = getlinvel(model, v, j)
        ω = getangvel(model, v, j)
        T += 0.5*(body.mass*ν'ν + ω'body.J*ω)
    end
    return T
end

function potential_energy(model::RobotArm, x)
    U = 0.0
    g = model.gravity
    for j = 1:model.numlinks
        body = model.links[j]
        r = gettran(model, x, j)
        U += g*body.mass*r[3]
    end
    return U
end

function ∇potential_energy!(model::RobotArm, y, x; k=1)
    g = model.gravity
    for j = 1:model.numlinks
        body = model.links[j]
        ri = getrind(model, j, k)
        r = gettran(model, x, j)
        U += g*body.mass*r[3]
    end
end

function D1Ld!(model::RobotArm, y, x1, x2, h; yi=1)
    ir = (1:3) .+ (yi-1)
    iq = (4:6) .+ (yi-1)
    g = model.gravity
    for j = 1:model.numlinks
        body = model.links[j]
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

function D2Ld!(model::RobotArm, y, x1, x2, h; yi=1)
    ir = (1:3) .+ (yi-1)
    iq = (4:6) .+ (yi-1)
    g = model.gravity
    for j = 1:model.numlinks
        body = model.links[j]
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

function joint_constraints!(model::RobotArm, c, x)
    p = 5
    for j = 1:model.numlinks
        joint = model.joints[j]
        ci = (1:p) .+ (j-1)*p
        r_1, r_2 = gettran(model, x, j-1), gettran(model, x, j)
        q_1, q_2 = getquat(model, x, j-1), getquat(model, x, j)
        c[ci] .= joint_constraint(joint, r_1, q_1, r_2, q_2)
    end
    return c
end

function ∇joint_constraints!(model::RobotArm, jac, x; errstate=Val(false), 
    transpose=Val(false), xi=1, yi=1, s = one(eltype(jac))
)   
    if transpose == Val(true)
        xi,yi=yi,xi
    end

    p = 5
    for j = 1:model.numlinks
        joint = model.joints[j]
        ci = (1:p) .+ (j-1)*p .+ (xi-1)
        r_1, r_2 = gettran(model, x, j-1), gettran(model, x, j)
        q_1, q_2 = getquat(model, x, j-1), getquat(model, x, j)
        w = errstate == Val(true) ? 6 : 7
        ir_1 = (1:3) .+ (j-2)*w .+ (yi - 1)  # rind[k,j-1]
        iq_1 = (4:w) .+ (j-2)*w .+ (yi - 1)  
        ir_2 = (1:3) .+ (j-1)*w .+ (yi - 1)  # rind[k,j] 
        iq_2 = (4:w) .+ (j-1)*w .+ (yi - 1)

        idx(i1,i2) = idx(i1,i2,transpose)
        idx(i1,i2,::Val{true}) = (i2,i1)
        idx(i1,i2,::Val{false}) = (i1,i2)
        dotran(A) = dotran(A, transpose) 
        dotran(A, ::Val{true}) = A'
        dotran(A, ::Val{false}) = A

        dr_1, dq_1, dr_2, dq_2 = ∇joint_constraint(joint, r_1, q_1, r_2, q_2)
        if (ir_1[1] - (yi-1)) > 0
            @view(jac[idx(ci, ir_1)...]) .+= dotran(dr_1) * s
            @view(jac[idx(ci, iq_1)...]) .+= dotran(dq_1 * G(q_1, errstate)) * s
        end
        @view(jac[idx(ci, ir_2)...]) .+= dotran(dr_2) * s
        @view(jac[idx(ci, iq_2)...]) .+= dotran(dq_2 * G(q_2, errstate)) * s
    end
    return jac 
end

function jtvp_joint_constraints!(model::RobotArm, y, x, λ; yi=1)
    ir_1 = (1:3) .- 6 .+ (yi-1)
    iq_1 = (4:6) .- 6 .+ (yi-1)
    p = 5
    for j = 1:model.numlinks
        joint = model.joints[j]
        ir_2 = ir_1 .+ 6
        iq_2 = iq_1 .+ 6
        ci = (1:p) .+ (j-1)*p
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
    return y
end

function getwrenches!(model::RobotArm, ξ, x, u; yi=1)
    for j = 1:model.numlinks
        joint = model.joints[j]
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

function ∇getwrenches!(model::RobotArm, jac, x, u; ix=1:14, iu=15:16, yi=1, s=1.0)
    for j = 1:model.numlinks
        joint = model.joints[j]
        hasforce = !(joint isa RevoluteJoint)
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

            if hasforce 
                F_11 = ∇force11(joint, r_1, q_1, r_2, q_2, u[j])
                @view(jac[iF_1, ir_1]) .+= F_11[1] * s
                @view(jac[iF_1, iq_1]) .+= F_11[2] * s

                F_12 = ∇force12(joint, r_1, q_1, r_2, q_2, u[j])
                @view(jac[iF_1, ir_2]) .+= F_12[1] * s
                @view(jac[iF_1, iq_2]) .+= F_12[2] * s

                F_1u = ∇force1u(joint, r_1, q_1, r_2, q_2, u[j])
                @view(jac[iF_1, iu_j]) .+= F_1u * s
            end

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
            if hasforce
                F_21 = ∇force21(joint, r_1, q_1, r_2, q_2, u[j])
                @view(jac[iF_2, ir_1]) .+= F_21[1] * s
                @view(jac[iF_2, iq_1]) .+= F_21[2] * s
            end

            T_21 = ∇torque21(joint, r_1, q_1, r_2, q_2, u[j])
            @view(jac[iT_2, ir_1]) .+= T_21[1] * s
            @view(jac[iT_2, iq_1]) .+= T_21[2] * s
        end

        if hasforce
            F_22 = ∇force22(joint, r_1, q_1, r_2, q_2, u[j])
            @view(jac[iF_2, ir_2]) .+= F_22[1] * s
            @view(jac[iF_2, iq_2]) .+= F_22[2] * s

            F_2u = ∇force2u(joint, r_1, q_1, r_2, q_2, u[j])
            @view(jac[iF_2, iu_j]) .+= F_2u * s
        end

        T_22 = ∇torque22(joint, r_1, q_1, r_2, q_2, u[j])
        @view(jac[iT_2, ir_2]) .+= T_22[1] * s
        @view(jac[iT_2, iq_2]) .+= T_22[2] * s

        T_2u = ∇torque2u(joint, r_1, q_1, r_2, q_2, u[j])
        @view(jac[iT_2, iu_j]) .+= T_2u * s
    end
    return jac
end

function DEL!(model::RobotArm, y, x1, x2, x3, λ, u1, u2, h; yi=1)
    @assert length(u1) == length(u2) == model.numlinks 
    edim = 6 * model.numlinks
    yview = view(y, (1:edim) .+ (yi-1))
    yview .= 0

    getwrenches!(model, yview, x1, u1)
    getwrenches!(model, yview, x2, u2)
    yview ./= 2

    jtvp_joint_constraints!(model, y, x2, λ, yi=yi)
    yview .*= h
    D2Ld!(model, y, x1, x2, h, yi=yi)
    D1Ld!(model, y, x2, x3, h, yi=yi)
    return y
end

function ∇DEL!(model::RobotArm, jac, x1, x2, x3, λ, u1, u2, h; 
    ix1 = 1:14, ix2 = ix1 .+ 16, ix3 = ix2 .+ 16, iu1=15:16, iu2=iu1 .+ 16, yi=1, λi=iu2[end]+1
)
    ∇getwrenches!(model, jac, x1, u1, ix=ix1, iu=iu1, yi=yi, s=h/2)
    ∇getwrenches!(model, jac, x2, u2, ix=ix2, iu=iu2, yi=yi, s=h/2)

    # Joint Constraints
    hess = jac
    x = x2
    errstate = Val(false)
    ir_1 = (1:3) .- 6 .+ (yi-1)
    iϕ_1 = (4:6) .- 6 .+ (yi-1)
    p = 5
    for j = 1:model.numlinks
        joint = model.joints[j]
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

    # Discrete Legendre Transforms
    ir = (1:3) .+ (yi-1)
    iq = (4:6) .+ (yi-1)
    p = 5
    g = model.gravity
    for j = 1:model.numlinks
        body = model.links[j]
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

    # Derivative wrt multiplier
    ∇joint_constraints!(model, jac, x2, 
        xi=yi, yi=λi, s=h, errstate=Val(true), transpose=Val(true))

    return jac
end

function ∇DEL3!(model::RobotArm, jac, x1, x2, x3, λ, F1, F2, h; yi=1)
    # Discrete Legendre Transforms
    ir = (1:3) .+ (yi-1)
    iq = (4:6) .+ (yi-1)
    M = model.numlinks  # number of links
    g = model.gravity
    for j = 1:model.numlinks
        body = model.links[j]
        m, J = body.mass, body.J
        q1, q2, q3,   = getquat(model, x1, j), getquat(model, x2, j), getquat(model, x3, j)
        ir3 = getrind(model, j)
        iq3 = getqind(model, j)
        
        # D1Ld
        @view(jac[ir, ir]) .+= -m/h * I3
        @view(jac[iq, iq]) .+= 4/h * G(q2)'Tmat*R(q3)'Hmat * J * Hmat'L(q2)'G(q3) + 
            4/h * G(q2)'Tmat*L(Hmat * J * Hmat'L(q2)'q3) * Tmat*G(q3)

        ir = ir .+ 6
        iq = iq .+ 6
    end

    # Derivative wrt multiplier
    ∇joint_constraints!(model, jac, x2, 
        xi=yi, yi=6*M+1, s=h, errstate=Val(true), transpose=Val(true))

    # Joint constraints at next time step
    ∇joint_constraints!(model, jac, x3,
        xi=yi+6*M, yi=1, errstate=Val(true))
    return jac
end

function simulate(model::RobotArm, params::SimParams, U, x0; newton_iters=20, tol=1e-12)
    M = model.numlinks 
    p = numconstraints(model)
    edim = 6M + p
    Δx = zero(x0)

    X = [zero(x0) for k = 1:params.N]
    λ = [zeros(p) for i = 1:params.N]

    xi = SVector{6M}(1:6M)
    yi = SVector{p}(6M .+ (1:p))
    e = zeros(edim)
    econ = view(e, 6M .+ (1:p))
    H = zeros(edim, edim)

    # Set Initial conditions
    X[1] = x0
    X[2] = x0

    for k = 2:params.N-1
        h = params.h

        # Initial guess
        X[k+1] .= X[k]

        for i = 1:newton_iters
            u1 = U[k-1]
            u2 = U[k]

            DEL!(model, e, X[k-1], X[k], X[k+1], λ[k], u1,u2, h)
            joint_constraints!(model, econ, X[k+1])
            H .= 0
            ∇DEL3!(model, H, X[k-1], X[k], X[k+1], λ[k], u1, u2, h)
            Δ = -(H\e)
            err2fullstate!(model, Δx, Δ[xi])
            compose_states!(model, X[k+1], X[k+1], Δx)

            if norm(e, Inf) < tol
                break
            end

            if i == newton_iters
                @warn "Newton failed to converge within $i iterations at timestep $k"
            end
        end

    end
    return X, λ

end