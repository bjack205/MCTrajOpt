struct TwoBodyMOI <: MOI.AbstractNLPEvaluator
    model::TwoBody
    params::SimParams
    Qr::Diagonal{Float64, SVector{3, Float64}}
    Qq::Diagonal{Float64, SVector{4, Float64}}
    R::Diagonal{Float64, SVector{1, Float64}}
    r0::Vector{SVector{3,Float64}}  # initial position
    q0::Vector{SVector{4,Float64}}  # initial orientation
    rf::Vector{SVector{3,Float64}}  # goal position
    qf::Vector{SVector{4,Float64}}  # goal orientation

    n_nlp::Int
    m_nlp::Int
    N::Int
    L::Int  # number of links
    rinds::Matrix{SVector{3,Int}}   # N × L matrix of position indices
    qinds::Matrix{SVector{4,Int}}
    xinds::Vector{SVector{14,Int}}
    uinds::Vector{SVector{1,Int}}
    λinds::Vector{SVector{5,Int}}
end

function TwoBodyMOI(b1::RigidBody, b2::RigidBody, joint::RevoluteJoint, params::SimParams,
    Qr, Qq, R, x0, xf
)
    model = TwoBody(b1, b2, joint)

    N = params.N
    L = 2
    n = 7 * L
    m = 1  # controls
    p = 5  # constraint forces

    ri = SA[1,2,3]
    qi = SA[4,5,6,7]
    rinds = [ri .+ ((k-1)*(14+m+p) + (i-1)*7) for k = 1:N, i = 1:2]
    qinds = [qi .+ ((k-1)*(14+m+p) + (i-1)*7) for k = 1:N, i = 1:2]
    xinds = [[ri; qi] for (ri,qi) in zip(rinds,qinds)]
    xinds = [[x1; x2] for (x1,x2) in eachrow(xinds)]
    uinds = SVector{m}.([(k-1)*(n+m+p) + n .+ (1:m) for k = 1:N-1])
    λinds = SVector{p}.([(k-1)*(n+m+p) + n + m .+ (1:p) for k = 1:N-1])

    r0 = [x0[ri] for ri in rinds[1,:]]
    q0 = [x0[qi] for qi in qinds[1,:]]
    rf = [xf[ri] for ri in rinds[1,:]]
    qf = [xf[qi] for qi in qinds[1,:]]

    n_nlp = sum(length.(xinds)) + sum(length.(uinds)) + sum(length.(λinds))
    p_del = (N-2)*6*L
    p_joints = (N-1)*p
    p_quatnorm = N
    m_nlp = p_del + p_joints + p_quatnorm 
    TwoBodyMOI(model, params, Qr, Qq, R, r0, q0, rf, qf, 
        n_nlp, m_nlp, N, L, rinds, qinds, xinds, uinds, λinds
    )
end

function MOI.eval_objective(prob::TwoBodyMOI, x)
    J = 0.0
    rinds, qinds = prob.rinds, prob.qinds
    uinds = prob.uinds
    for k = 1:prob.N
        r1,r2 = x[rinds[k,1]], x[rinds[k,2]]
        q1,q2 = x[qinds[k,1]], x[qinds[k,2]]
        dr1,dr2 = r1 - prob.rf[1], r2 - prob.rf[2]
        dq1,dq2 = q1 - prob.qf[1], q2 - prob.qf[2]
        s = k < prob.N ? 1.0 : 10.0
        J += 0.5*(dr1'prob.Qr*dr1 + dq1'prob.Qq*dq1) * s
        J += 0.5*(dr2'prob.Qr*dr2 + dq2'prob.Qq*dq2) * s
        if k < prob.N
            u = x[uinds[k]]
            J += 0.5*(u'prob.R*u)
        end
    end
    return J
end

function MOI.eval_objective_gradient(prob::TwoBodyMOI, grad_f, x)
    obj(x) = MOI.eval_objective(prob, x)
    ForwardDiff.gradient!(grad_f, obj, x)  # TODO: use a cache
    return
end

function MOI.eval_constraint(prob::TwoBodyMOI, c, x)
    h = prob.params.h
    Br_1 = @SMatrix [0; 0; 0]
    Bq_1 = @SMatrix [0; 0; -1]
    Br_2 = @SMatrix [0; 0; 0]
    Bq_2 = @SMatrix [0; 0; +1]
    rinds, qinds = prob.rinds, prob.qinds
    uinds, λinds = prob.uinds, prob.λinds
    ri1 = 1:3
    qi1 = 4:6 
    ri2 = ri1 .+ 6
    qi2 = qi1 .+ 6

    # DEL constraint
    for (i,k) in enumerate(2:prob.N-1)
        r1_1, r1_2 = x[rinds[k-1,1]], x[rinds[k-1,2]]
        q1_1, q1_2 = x[qinds[k-1,1]], x[qinds[k-1,2]]
        u1 = x[uinds[k-1]]
        r2_1, r2_2 = x[rinds[k,1]], x[rinds[k,2]]
        q2_1, q2_2 = x[qinds[k,1]], x[qinds[k,2]]
        u2 = x[uinds[k,1]]
        r3_1, r3_2 = x[rinds[k+1,1]], x[rinds[k+1,2]]
        q3_1, q3_2 = x[qinds[k+1,1]], x[qinds[k+1,2]]

        # Get forces and torques on each body
        F1_1,T1_1 = Br_1*u1, Bq_1*u1
        F1_2,T1_2 = Br_2*u1, Bq_2*u1
        F2_1,T2_1 = Br_1*u2, Bq_1*u2
        F2_2,T2_2 = Br_2*u2, Bq_2*u2

        c[ri1] = DEL_trans(prob.model.b1, r1_1, r2_1, r3_1, F1_1, F2_1, h)
        c[qi1] = DEL_rot(prob.model.b1,   q1_1, q2_1, q3_1, T1_1, T2_1, h)
        c[ri2] = DEL_trans(prob.model.b2, r1_2, r2_2, r3_2, F1_2, F2_2, h)
        c[qi2] = DEL_rot(prob.model.b2,   q1_2, q2_2, q3_2, T1_2, T2_2, h)

        λ2 = x[λinds[i]]
        x2 = x[prob.xinds[k]]
        # λ2 = @SVector ones(5)
        c[ri1[1]:qi2[end]] += errstate_jacobian(prob.model, x2)'∇joint_constraints(prob, x2)'λ2

        ri1 = ri1 .+ 12
        qi1 = qi1 .+ 12
        ri2 = ri2 .+ 12
        qi2 = qi2 .+ 12
    end

    # Joint constraints
    ji = 6*prob.L*(prob.N-2) .+ (1:5)
    for (i,k) in enumerate(2:prob.N)
        x2 = x[prob.xinds[k]]
        c[ji] .= joint_constraints(prob, x2)
        ji = ji .+ 5
        # i > 2 && break
    end

    # Quaternion norm constraints
    ni = 6*prob.L*(prob.N-2) + 5*(prob.N-1)
    for (i,k) in enumerate(1:prob.N)
        qk = x[qinds[k]]
        c[ni+i] = 1 - qk'qk
    end

    return
end

function MOI.eval_constraint_jacobian(prob::TwoBodyMOI, jac, x)
    J0 = reshape(jac, prob.m_nlp, prob.n_nlp)
    # c = zeros(eltype(x), prob.m_nlp)
    # ForwardDiff.jacobian!(J0, (c,z)->MOI.eval_constraint(prob, c, z), c, x)

    h = prob.params.h
    Br_1 = @SMatrix [0; 0; 0]
    Bq_1 = @SMatrix [0; 0; -1]
    Br_2 = @SMatrix [0; 0; 0]
    Bq_2 = @SMatrix [0; 0; +1]

    rinds, qinds = prob.rinds, prob.qinds
    uinds, λinds = prob.uinds, prob.λinds
    ri1 = 1:3
    qi1 = 4:6 
    ri2 = ri1 .+ 6
    qi2 = qi1 .+ 6
    for (i,k) in enumerate(2:prob.N-1)
        r1_1, r1_2 = x[rinds[k-1,1]], x[rinds[k-1,2]]
        q1_1, q1_2 = x[qinds[k-1,1]], x[qinds[k-1,2]]
        u1 = x[uinds[k-1]]
        r2_1, r2_2 = x[rinds[k,1]], x[rinds[k,2]]
        q2_1, q2_2 = x[qinds[k,1]], x[qinds[k,2]]
        u2 = x[uinds[k,1]]
        r3_1, r3_2 = x[rinds[k+1,1]], x[rinds[k+1,2]]
        q3_1, q3_2 = x[qinds[k+1,1]], x[qinds[k+1,2]]
        
        # Get forces and torques on each body
        F1_1,T1_1 = Br_1*u1, Bq_1*u1
        F1_2,T1_2 = Br_2*u1, Bq_2*u1
        F2_1,T2_1 = Br_1*u2, Bq_1*u2
        F2_2,T2_2 = Br_2*u2, Bq_2*u2

        dr_1 = ∇DEL_trans(prob.model.b1, r1_1, r2_1, r3_1, F1_1, F2_1, h)
        dq_1 = ∇DEL_rot(prob.model.b1,   q1_1, q2_1, q3_1, T1_1, T2_1, h)
        dr_2 = ∇DEL_trans(prob.model.b2, r1_2, r2_2, r3_2, F1_2, F2_2, h)
        dq_2 = ∇DEL_rot(prob.model.b2,   q1_2, q2_2, q3_2, T1_2, T2_2, h)
        J0[ri1, rinds[k-1,1]] = dr_1[1]
        J0[ri1, rinds[k+0,1]] = dr_1[2]
        J0[ri1, rinds[k+1,1]] = dr_1[3]
        J0[ri1, uinds[k-1]] += dr_1[4] * Br_1
        J0[ri1, uinds[k+0]] += dr_1[5] * Br_1

        J0[qi1, qinds[k-1,1]] = dq_1[1]
        J0[qi1, qinds[k+0,1]] = dq_1[2]
        J0[qi1, qinds[k+1,1]] = dq_1[3]
        J0[qi1, uinds[k-1]] += dq_1[4] * Bq_1
        J0[qi1, uinds[k+0]] += dq_1[5] * Bq_1

        J0[ri2, rinds[k-1,2]] = dr_2[1]
        J0[ri2, rinds[k+0,2]] = dr_2[2]
        J0[ri2, rinds[k+1,2]] = dr_2[3]
        J0[ri2, uinds[k-1]] += dr_2[4] * Br_2
        J0[ri2, uinds[k+0]] += dr_2[5] * Br_2

        J0[qi2, qinds[k-1,2]] = dq_2[1]
        J0[qi2, qinds[k+0,2]] = dq_2[2]
        J0[qi2, qinds[k+1,2]] = dq_2[3]
        J0[qi2, uinds[k-1]] += dq_2[4] * Bq_2
        J0[qi2, uinds[k+0]] += dq_2[5] * Bq_2

        λ2 = x[λinds[i]]
        x2 = x[prob.xinds[k]]
        # λ2 = @SVector ones(5)
        J0[ri1[1]:qi2[end], prob.xinds[k]] += 
            errstate_jacobian(prob.model, x2)'∇²joint_constraints(prob, x2, λ2) +
            ∇errstate_jacobian(prob.model, x2, ∇joint_constraints(prob, x2)'λ2)
        J0[ri1[1]:qi2[end], prob.λinds[i]] += errstate_jacobian(prob.model, x2)'∇joint_constraints(prob, x2)'

        ri1 = ri1 .+ 12
        qi1 = qi1 .+ 12
        ri2 = ri2 .+ 12
        qi2 = qi2 .+ 12
    end

    # Joint constraints
    ji = 6*prob.L*(prob.N-2) .+ (1:5)
    for (i,k) in enumerate(2:prob.N)
        x2 = x[prob.xinds[k]]
        J0[ji, prob.xinds[k]] .= ∇joint_constraints(prob, x2) 
        ji = ji .+ 5
    end

    # Quaternion norm constraints
    ni = 6*prob.L*(prob.N-2) + 5*(prob.N-1)
    for (i,k) in enumerate(1:prob.N)
        qk = x[qinds[k]]
        J0[ni+i, qinds[k]] = -2qk
    end
end


function joint_constraints(prob::TwoBodyMOI, x)
    r_1 = x[SA[1,2,3]]
    q_1 = x[SA[4,5,6,7]]
    r_2 = x[SA[8,9,10]]
    q_2 = x[SA[11,12,13,14]]
    joint = prob.model.joint
    [
        r_1 + Hmat'R(q_1)'L(q_1)*Hmat*joint.p1 - (r_2 + Hmat'R(q_2)'L(q_2)*Hmat*joint.p2);  # joint location 
        joint.orth * L(q_1)'q_2                                                            # joint axis
    ]
end

function ∇joint_constraints(prob::TwoBodyMOI, x)
    # con(x) = joint_constraints(prob, x)
    # ForwardDiff.jacobian(con, x) * errstate_jacobian(prob.model, x)
    r_1 = x[SA[1,2,3]]
    q_1 = x[SA[4,5,6,7]]
    r_2 = x[SA[8,9,10]]
    q_2 = x[SA[11,12,13,14]]
    joint = prob.model.joint

    I3 = @SMatrix [1 0 0; 0 1 0; 0 0 1]
    Z23 = @SMatrix zeros(2,3)
    jac1 = [I3 ∇rot(q_1, joint.p1) -I3 -∇rot(q_2, joint.p2)] 
    jac2 = [Z23 joint.orth*R(q_2)*Tmat Z23 joint.orth*L(q_1)']
    return [ jac1; jac2 ]
end

function jtvp_joint_constraints(prob::TwoBodyMOI, x, λ)
    r_1 = x[SA[1,2,3]]
    q_1 = x[SA[4,5,6,7]]
    r_2 = x[SA[8,9,10]]
    q_2 = x[SA[11,12,13,14]]
    λ_1 = λ[SA[1,2,3]]
    λ_2 = λ[SA[4,5]]
    joint = prob.model.joint
    
    dr_1 = λ_1
    dq_1 = ∇rot(q_1, joint.p1)'λ_1 + Tmat*R(q_2)'joint.orth'λ_2
    dr_2 = -λ_1
    dq_2 = -∇rot(q_2, joint.p2)'λ_1 + L(q_1)*joint.orth'λ_2
    return [ dr_1; dq_1; dr_2; dq_2 ]
end

function ∇²joint_constraints(prob::TwoBodyMOI, x, λ)
    r_1 = x[SA[1,2,3]]
    q_1 = x[SA[4,5,6,7]]
    r_2 = x[SA[8,9,10]]
    q_2 = x[SA[11,12,13,14]]
    λ_1 = λ[SA[1,2,3]]
    λ_2 = λ[SA[4,5]]
    joint = prob.model.joint

    Z43 = @SMatrix zeros(4,3)
    ∇dr_1 = @SMatrix zeros(3, 14) 
    ∇dq_1 = [Z43 ∇²rot(q_1, joint.p1, λ_1) Z43 Tmat*L(joint.orth'λ_2)*Tmat]
    ∇dr_2 = @SMatrix zeros(3, 14) 
    ∇dq_2 = [Z43 R(joint.orth'λ_2) Z43 -∇²rot(q_2, joint.p2, λ_1)]
    return [ ∇dr_1; ∇dq_1; ∇dr_2; ∇dq_2]
end

MOI.features_available(prob::TwoBodyMOI) = [:Grad, :Jac]

MOI.initialize(prob::TwoBodyMOI, features) = nothing

function MOI.jacobian_structure(prob::TwoBodyMOI) 
    return vec(Tuple.(CartesianIndices((prob.m_nlp,prob.n_nlp))))
end

function ipopt_solve(prob::TwoBodyMOI, x0; tol=1e-6, c_tol=1e-6, max_iter=1_000)

    n_nlp = prob.n_nlp 
    m_nlp = prob.m_nlp 
    # x_l, x_u = prob.x_lb, prob.x_ub
    # c_l, c_u = prob.c_lb, prob.c_ub
    
    x_l = fill(-Inf,n_nlp)
    x_u = fill(+Inf,n_nlp)
    x_l[1:3] = prob.r0[1]
    x_u[1:3] = prob.r0[1]
    x_l[4:7] = prob.q0[1]
    x_u[4:7] = prob.q0[1]
    x_l[8:10] = prob.r0[2]
    x_u[8:10] = prob.r0[2]
    x_l[11:14] = prob.q0[2]
    x_u[11:14] = prob.q0[2]
    c_l = zeros(m_nlp)
    c_u = zeros(m_nlp)

    nlp_bounds = MOI.NLPBoundsPair.(c_l, c_u)
    block_data = MOI.NLPBlockData(nlp_bounds, prob, true)

    solver = Ipopt.Optimizer()
    solver.options["max_iter"] = max_iter
    solver.options["tol"] = tol
    solver.options["constr_viol_tol"] = c_tol

    x = MOI.add_variables(solver, n_nlp)
    for i = 1:n_nlp
        MOI.add_constraint(solver, x[i], MOI.LessThan(x_u[i]))
        MOI.add_constraint(solver, x[i], MOI.GreaterThan(x_l[i]))
        MOI.set(solver, MOI.VariablePrimalStart(), x[i], x0[i])
    end

    MOI.set(solver, MOI.NLPBlock(), block_data)
    MOI.set(solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(solver)

    res = MOI.get(solver, MOI.VariablePrimal(), x)
    return res, solver
end