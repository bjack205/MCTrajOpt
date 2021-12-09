
struct ArmMOI{Nu,Nc} <: MOI.AbstractNLPEvaluator
    model::RobotArm
    params::SimParams
    Qr::Diagonal{Float64, SVector{3, Float64}}
    Qq::Diagonal{Float64, SVector{4, Float64}}
    R::Diagonal{Float64, SVector{Nu, Float64}}
    r0::Vector{SVector{3,Float64}}  # initial position
    q0::Vector{SVector{4,Float64}}  # initial orientation
    rgoal::SVector{3,Float64}       # End-effector goal
    p_ee::SVector{3,Float64}        # Location of end-effector in body 2 frame
    goalcon::Bool                   # Use end-effector goal constraint
    Xref::Vector{Vector{Float64}}

    n_nlp::Int
    m_nlp::Int
    N::Int
    L::Int  # number of links
    p::Int  # number of constraints per joint
    ncons::Dict{Symbol,Int}
    rinds::Matrix{SVector{3,Int}}   # N × L matrix of position indices
    qinds::Matrix{SVector{4,Int}}
    xinds::Vector{UnitRange{Int}}
    uinds::Vector{SVector{Nu,Int}}
    λinds::Vector{SVector{Nc,Int}}
    blocks::BlockViews
    function ArmMOI(model::RobotArm, params::SimParams, Qr, Qq, R, x0, Xref; 
        rf = nothing, p_ee = SA[0,0,0.5]
    )
        N = params.N
        L = model.numlinks 
        n = 7 * L                               # states
        m = L                                   # controls
        p = sum(numconstraints.(model.joints))  # constraint forces

        # Generate indices
        ri = SA[1,2,3]
        qi = SA[4,5,6,7]
        Nz = n + m + p
        rinds = [ri .+ ((k-1)*Nz + (i-1)*7) for k = 1:N, i = 1:L]
        qinds = [qi .+ ((k-1)*Nz + (i-1)*7) for k = 1:N, i = 1:L]
        xinds = [(k-1)*Nz .+ (1:n) for k = 1:N]
        uinds = SVector{m}.([(k-1)*Nz + n .+ (1:m) for k = 1:N-1])
        λinds = SVector{p}.([(k-1)*Nz + n + m .+ (1:p) for k = 1:N-1])

        # Parse initial state
        r0 = [x0[ri] for ri in rinds[1,:]]
        q0 = [x0[qi] for qi in qinds[1,:]]

        # Goal constraint
        goalcon = !isnothing(rf)
        if !goalcon
            rf = SA[0,0,0.] 
        end

        # NLP info
        n_nlp = sum(length.(rinds)) + sum(length.(qinds)) + sum(length.(uinds)) + sum(length.(λinds))
        p_del = (N-2)*6*L
        p_joints = (N-1)*p
        p_quatnorm = (N-1)*L
        p_goal = 3 * goalcon
        m_nlp = p_del + p_joints + p_quatnorm + p_goal
        ncons = Dict(:DEL=>p_del, :joints=>p_joints, :quatnorm=>p_quatnorm)

        # Sparsity blocks
        blocks = BlockViews(m_nlp, n_nlp)

        # Create type and initialize the sparsity pattern
        prob = new{m,p}(model, params, Qr, Qq, R, r0, q0, rf, p_ee, goalcon, Xref,
            n_nlp, m_nlp, N, L, p, ncons, rinds, qinds, xinds, uinds, λinds, blocks
        )
        initialize_sparsity!(prob)
        prob

    end
end

function initialize_sparsity!(prob::ArmMOI)
    blocks = prob.blocks

    # Run through the constraint Jacobian to record 
    # the sparsity structure
    jac = NonzerosVector(zeros(0), blocks)
    z = ones(prob.n_nlp)
    blocks.initializing = true
    MOI.eval_constraint_jacobian(prob, jac, z)
    blocks.initializing = false 
    return
end

function MOI.eval_objective(prob::ArmMOI, z)
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
            # dq = q - qref
            # J += 0.5 * (dr'prob.Qr*dr + dq'prob.Qq*dq)
            dq = q'qref
            J += 0.5 * (dr'prob.Qr*dr + min(1+dq, 1-dq) * 1)
        end
        if k < prob.N
            u = z[uinds[k]]
            J += 0.5 * u'prob.R*u
        end
    end
    return J
end

function MOI.eval_objective_gradient(prob::ArmMOI, grad_f, x)
    obj(x) = MOI.eval_objective(prob, x)
    ForwardDiff.gradient!(grad_f, obj, x)  # TODO: use a cache
    return
end

function MOI.eval_constraint(prob::ArmMOI, c, z)
    h = prob.params.h
    rinds, qinds = prob.rinds, prob.qinds
    uinds, λinds = prob.uinds, prob.λinds
    xinds = prob.xinds

    off = 0

    # Discrete Euler-Lagrange (dynamics) constraints
    ci = 1:6*prob.L
    for (i,k) in enumerate(2:prob.N-1)
        x1 = view(z, xinds[k-1])
        u1 = z[uinds[k-1]]
        x2 = view(z, xinds[k])
        u2 = z[uinds[k]]
        x3 = view(z, xinds[k+1])

        # Compute the Discrete Euler-Lagrange constraint
        λ = z[λinds[i]]
        DEL!(prob.model, c, x1, x2, x3, λ, u1, u2, h, yi=ci[1])
        
        ci = ci .+ 6*prob.L
        off += 6*prob.L
    end

    # Joint constraints
    ci = off .+ (1:prob.p)
    for (i,k) in enumerate(2:prob.N)  # assume the initial configuration is feasible
        x = view(z, xinds[k])
        # c[ci] = joint_constraints(prob.model, x)
        joint_constraints!(prob.model, view(c, ci), x)
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
        rf = z[rinds[prob.N,end]]
        qf = z[qinds[prob.N,end]]
        c[ci] = rf + Amat(qf)*prob.p_ee - prob.rgoal
    end
    return
end

function MOI.eval_constraint_jacobian(prob::ArmMOI, jac, z)
    jac .= 0
    J0 = NonzerosVector(jac, prob.blocks)

    h = prob.params.h
    rinds, qinds = prob.rinds, prob.qinds
    uinds, λinds = prob.uinds, prob.λinds
    xinds = prob.xinds

    off = 0

    # Discrete Euler-Lagrange (dynamics) constraints
    ci = 1:6*prob.L
    for (i,k) in enumerate(2:prob.N-1)
        x1 = view(z, xinds[k-1])
        u1 = z[uinds[k-1]]
        x2 = view(z, xinds[k])
        u2 = z[uinds[k]]
        x3 = view(z, xinds[k+1])

        # Compute the Discrete Euler-Lagrange constraint
        λ2 = z[λinds[i]]
        ∇DEL!(prob.model, J0, x1, x2, x3, λ2, u1, u2, h, 
            ix1=xinds[k-1], ix2=xinds[k], ix3=xinds[k+1], iu1=uinds[k-1], iu2=uinds[k], yi=ci[1], λi=λinds[i][1])
        
        ci = ci .+ 6*prob.L
        off += 6*prob.L
    end

    # Joint constraints
    ci = off .+ (1:prob.p)
    for (i,k) in enumerate(2:prob.N)  # assume the initial configuration is feasible
        x = view(z, xinds[k])
        ∇joint_constraints!(prob.model, J0, x, xi=ci[1], yi=xinds[k][1])
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
        rf = z[rinds[prob.N,end]]
        qf = z[qinds[prob.N,end]]
        J0[ci, rinds[prob.N,end]] = I3 
        J0[ci, qinds[prob.N,end]] = ∇rot(qf, prob.p_ee) 
    end
    return 
end

MOI.features_available(prob::ArmMOI) = [:Grad, :Jac]

MOI.initialize(prob::ArmMOI, features) = nothing

function MOI.jacobian_structure(prob::ArmMOI) 
    return getrc(prob.blocks)
end

function ipopt_solve(prob::ArmMOI, x0; tol=1e-6, c_tol=1e-6, max_iter=1_000, goal_tol=1e-2)

    n_nlp = prob.n_nlp 
    m_nlp = prob.m_nlp 
    
    x_l = fill(-Inf,n_nlp)
    x_u = fill(+Inf,n_nlp)
    for j = 1:prob.L
        x_l[prob.rinds[1,j]] = prob.r0[j]
        x_u[prob.rinds[1,j]] = prob.r0[j]
        x_l[prob.qinds[1,j]] = prob.q0[j]
        x_u[prob.qinds[1,j]] = prob.q0[j]
    end
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

function randtraj(prob::ArmMOI{Nu,Nc}) where {Nu,Nc}
    z0 = zeros(prob.n_nlp)
    for k = 1:prob.N
        for j = 1:prob.L
            z0[prob.rinds[k,j]] = @SVector randn(3)
            z0[prob.qinds[k,j]] = normalize(@SVector randn(4))
        end
        if k < prob.N
            z0[prob.uinds[k]] = @SVector randn(Nu) 
            z0[prob.λinds[k]] = @SVector randn(Nc) 
        end
    end
    return z0
end

function buildtraj(prob::ArmMOI, X, U, λ)
    z0 = zeros(prob.n_nlp)
    for k = 1:prob.N
        for j = 1:prob.L
            z0[prob.rinds[k,j]] = gettran(prob.model, X[k], j)
            z0[prob.qinds[k,j]] = getquat(prob.model, X[k], j)
        end
        if k < prob.N
            z0[prob.uinds[k]] = U[k] 
            z0[prob.λinds[k]] = λ[k] 
        end
    end
    return z0
end