using MathOptInterface
const MOI = MathOptInterface

struct DoublePendulumMOI{p} <: MOI.AbstractNLPEvaluator
    model::DoublePendulum
    params::SimParams
    Qr::Diagonal{Float64, SVector{3, Float64}}
    Qq::Diagonal{Float64, SVector{4, Float64}}
    R::Diagonal{Float64, SVector{2, Float64}}
    r0::Vector{SVector{3,Float64}}  # initial position
    q0::Vector{SVector{4,Float64}}  # initial orientation
    rgoal::SVector{3,Float64}       # End-effector goal
    p_ee::SVector{3,Float64}        # Location of end-effector in body 2 frame
    goalcon::Bool                   # Use end-effector goal constraint
    Xref::Vector{SVector{14,Float64}}

    n_nlp::Int
    m_nlp::Int
    N::Int
    L::Int  # number of links
    p::Int  # number of constraints per joint
    ncons::Dict{Symbol,Int}
    rinds::Matrix{SVector{3,Int}}   # N × L matrix of position indices
    qinds::Matrix{SVector{4,Int}}
    xinds::Vector{SVector{14,Int}}
    uinds::Vector{SVector{2,Int}}
    λinds::Vector{SVector{p,Int}}
    blocks::BlockViews
end

function DoublePendulumMOI(model::DoublePendulum, params::SimParams,
    Qr, Qq, R, x0, Xref; rf = nothing, p_ee = SA[0,0,0.5]
)
    N = params.N
    L = 2
    n = 7 * L
    m = 2  # controls
    p = 10  # constraint forces

    ri = SA[1,2,3]
    qi = SA[4,5,6,7]
    Nz = n + m + p
    rinds = [ri .+ ((k-1)*Nz + (i-1)*7) for k = 1:N, i = 1:L]
    qinds = [qi .+ ((k-1)*Nz + (i-1)*7) for k = 1:N, i = 1:L]
    xinds = [[ri; qi] for (ri,qi) in zip(rinds,qinds)]
    xinds = [[x1; x2] for (x1,x2) in eachrow(xinds)]
    uinds = SVector{m}.([(k-1)*Nz + n .+ (1:m) for k = 1:N-1])
    λinds = SVector{p}.([(k-1)*Nz + n + m .+ (1:p) for k = 1:N-1])

    r0 = [x0[ri] for ri in rinds[1,:]]
    q0 = [x0[qi] for qi in qinds[1,:]]

    goalcon = !isnothing(rf)
    if !goalcon
        rf = SA[0,0,0.] 
    end

    n_nlp = sum(length.(xinds)) + sum(length.(uinds)) + sum(length.(λinds))
    p_del = (N-2)*6*L
    p_joints = (N-1)*p
    p_quatnorm = (N-1)*L
    p_goal = 3 * goalcon
    m_nlp = p_del + p_joints + p_quatnorm + p_goal
    ncons = Dict(:DEL=>p_del, :joints=>p_joints, :quatnorm=>p_quatnorm)

    blocks = BlockViews(m_nlp, n_nlp)

    prob = DoublePendulumMOI{p}(model, params, Qr, Qq, R, r0, q0, rf, p_ee, goalcon, Xref,
        n_nlp, m_nlp, N, L, p, ncons, rinds, qinds, xinds, uinds, λinds, blocks
    )
    initialize_sparsity!(prob)
    prob
end

function initialize_sparsity!(prob::DoublePendulumMOI)
    rinds, qinds = prob.rinds, prob.qinds
    uinds, λinds = prob.uinds, prob.λinds
    xinds = prob.xinds
    blocks = prob.blocks

    off = 0

    jac = NonzerosVector(zeros(0), blocks)
    z = ones(prob.n_nlp)
    blocks.initializing = true
    MOI.eval_constraint_jacobian(prob, jac, z)
    blocks.initializing = false 
    return

    # Discrete Euler-Lagrange (dynamics) constraints
    ci = 1:6*prob.L
    for (i,k) in enumerate(2:prob.N-1)
        setblock!(blocks, ci, xinds[k-1])
        setblock!(blocks, ci, xinds[k])
        setblock!(blocks, ci, xinds[k+1])
        setblock!(blocks, ci, uinds[k-1])
        setblock!(blocks, ci, uinds[k])
        setblock!(blocks, ci, λinds[i])
        ci = ci .+ 6*prob.L
        off += 6*prob.L
    end

    # Joint constraints
    ci = off .+ (1:prob.p)
    for (i,k) in enumerate(2:prob.N)  # assume the initial configuration is feasible
        setblock!(blocks, ci, xinds[k])
        ci = ci .+ prob.p
        off += prob.p
    end

    # Quaternion norm constraints
    ci = off .+ 1
    for (i,k) in enumerate(2:prob.N)
        L = prob.L
        for j = 1:L
            setblock!(blocks, ci:ci, qinds[k,j])
            ci += 1
        end
        off += L
    end

    if prob.goalcon
        ci = off .+ (1:3)
        setblock!(blocks, ci, rinds[prob.N,2])
        setblock!(blocks, ci, qinds[prob.N,2])
    end
    return
end

function MOI.eval_objective(prob::DoublePendulumMOI, z)
    model = prob.model
    J = 0.0
    rinds, qinds = prob.rinds, prob.qinds
    uinds = prob.uinds
    for k = 1:prob.N
        for j = 1:prob.L
            r = z[rinds[k,j]]
            q = z[qinds[k,j]]
            xref = prob.Xref[k]
            rref = gettran(model, xref, j)
            qref = getquat(model, xref, j)
            dr = r - rref
            dq = q - qref
            J += 0.5 * (dr'prob.Qr*dr + dq'prob.Qq*dq)
        end
        # r_1, r_2 = Z[rinds[k,1]], Z[rinds[k,2]]
        # q_1, q_2 = Z[qinds[k,1]], Z[qinds[k,2]]
        # xref = prob.Xref[k]
        # rref_1, rref_2 = gettran(prob.model, xref)
        # qref_1, qref_2 = getquat(prob.model, xref)

        # dr_1, dr_2 = r_1 - rref_1, r_2 - rref_2 
        # dq_1, dq_2 = q_1 - qref_1, q_2 - qref_2 
        # J += 0.5 * (dr_1'prob.Qr*dr_1 + dq_1'prob.Qq*dq_1)
        # J += 0.5 * (dr_2'prob.Qr*dr_2 + dq_2'prob.Qq*dq_2)
        if k < prob.N
            u = z[uinds[k]]
            J += 0.5 * u'prob.R*u
        end
    end
    return J
end

function MOI.eval_objective_gradient(prob::DoublePendulumMOI, grad_f, x)
    obj(x) = MOI.eval_objective(prob, x)
    ForwardDiff.gradient!(grad_f, obj, x)  # TODO: use a cache
    return
end

function MOI.eval_constraint(prob::DoublePendulumMOI, c, z)
    h = prob.params.h
    rinds, qinds = prob.rinds, prob.qinds
    uinds, λinds = prob.uinds, prob.λinds
    xinds = prob.xinds
    ncons = prob.ncons

    off = 0

    # Discrete Euler-Lagrange (dynamics) constraints
    ci = 1:6*prob.L
    for (i,k) in enumerate(2:prob.N-1)
        x1 = z[xinds[k-1]] 
        u1 = z[uinds[k-1]]
        x2 = z[xinds[k]] 
        u2 = z[uinds[k]]
        x3 = z[xinds[k+1]] 

        # Get the wrenches on each body as function of the control inputs
        F1 = getwrenches(prob.model, x1, u1)
        F2 = getwrenches(prob.model, x2, u2)

        # Compute the Discrete Euler-Lagrange constraint
        λ = z[λinds[i]]
        c[ci] = DEL(prob.model, x1, x2, x3, λ, u1, u2, h)
        # DEL!(prob.model, c, x1, x2, x3, λ, F1, F2, h, yi=ci[1])
        
        ci = ci .+ 6*prob.L
        off += 6*prob.L
    end

    # Joint constraints
    ci = off .+ (1:prob.p)
    for (i,k) in enumerate(2:prob.N)  # assume the initial configuration is feasible
        x = z[xinds[k]]
        c[ci] = joint_constraints(prob.model, x)
        ci = ci .+ prob.p
        off += prob.p
    end

    # Quaternion norm constraints
    ci = off .+ 1 
    L = prob.L
    for (i,k) in enumerate(2:prob.N)
        for j = 1:L
            q = z[qinds[k,j]]
            c[ci] = q'q - 1
            ci += 1
        end
        off += L
    end

    if prob.goalcon
        ci = off .+ (1:3)
        rf = z[rinds[prob.N,2]]
        qf = z[qinds[prob.N,2]]
        c[ci] = rf + Amat(qf)*prob.p_ee - prob.rgoal
    end
    return
end

function MOI.eval_constraint_jacobian(prob::DoublePendulumMOI, jac, z)
    # J0 = reshape(jac, prob.m_nlp, prob.n_nlp)
    # c = zeros(eltype(x), prob.m_nlp)
    # ForwardDiff.jacobian!(J0, (c,z)->MOI.eval_constraint(prob, c, z), c, x)
    jac .= 0
    J0 = NonzerosVector(jac, prob.blocks)

    h = prob.params.h
    rinds, qinds = prob.rinds, prob.qinds
    uinds, λinds = prob.uinds, prob.λinds
    xinds = prob.xinds
    ncons = prob.ncons

    off = 0

    # Discrete Euler-Lagrange (dynamics) constraints
    ci = 1:6*prob.L
    for (i,k) in enumerate(2:prob.N-1)
        x1 = z[xinds[k-1]] 
        u1 = z[uinds[k-1]]
        x2 = z[xinds[k]] 
        u2 = z[uinds[k]]
        x3 = z[xinds[k+1]] 

        con(x1,x2,x3,u1,u2,λ) = begin
            # F1 = getwrenches(prob.model, x1, u1)
            # F2 = getwrenches(prob.model, x2, u2)
            DEL(prob.model, x1, x2, x3, λ, u1, u2, h) 
        end 

        # Compute the Discrete Euler-Lagrange constraint
        λ2 = z[λinds[i]]
        J0[ci, xinds[k-1]] = ForwardDiff.jacobian(x->con(x, x2, x3, u1, u2, λ2), x1)
        J0[ci, xinds[k]]   = ForwardDiff.jacobian(x->con(x1, x, x3, u1, u2, λ2), x2)
        J0[ci, xinds[k+1]] = ForwardDiff.jacobian(x->con(x1, x2, x, u1, u2, λ2), x3)
        J0[ci, uinds[k-1]] = ForwardDiff.jacobian(u->con(x1, x2, x3, u, u2, λ2), u1)
        J0[ci, uinds[k]]   = ForwardDiff.jacobian(u->con(x1, x2, x3, u1, u, λ2), u2)
        J0[ci, λinds[i]]   = ForwardDiff.jacobian(λ->con(x1, x2, x3, u1, u2, λ), λ2)
        
        ci = ci .+ 6*prob.L
        off += 6*prob.L
    end

    # Joint constraints
    ci = off .+ (1:prob.p)
    for (i,k) in enumerate(2:prob.N)  # assume the initial configuration is feasible
        x = z[xinds[k]]
        J0[ci, xinds[k]] = ∇joint_constraints(prob.model, x)
        ci = ci .+ prob.p
        off += prob.p
    end

    # Quaternion norm constraints
    ci = off .+ 1
    L = prob.L
    for (i,k) in enumerate(2:prob.N)
        for j = 1:L
            q = z[qinds[k,j]]
            J0[ci:ci, qinds[k,j]] = 2q'
            ci += 1
        end
        off += L
    end

    if prob.goalcon
        ci = off .+ (1:3)
        rf = z[rinds[prob.N,2]]
        qf = z[qinds[prob.N,2]]
        J0[ci, rinds[prob.N,2]] = SA[1 0 0; 0 1 0; 0 0 1] 
        J0[ci, qinds[prob.N,2]] = ∇rot(qf, prob.p_ee) 
    end
    return 
end

MOI.features_available(prob::DoublePendulumMOI) = [:Grad, :Jac]

MOI.initialize(prob::DoublePendulumMOI, features) = nothing

function MOI.jacobian_structure(prob::DoublePendulumMOI) 
    return getrc(prob.blocks)
    # return vec(Tuple.(CartesianIndices((prob.m_nlp,prob.n_nlp))))
end

function ipopt_solve(prob::DoublePendulumMOI, x0; tol=1e-6, c_tol=1e-6, max_iter=1_000, goal_tol=1e-2)

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

    if prob.goalcon
        c_l[end-2:end] .= -goal_tol
        c_u[end-2:end] .= +goal_tol
    end

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
