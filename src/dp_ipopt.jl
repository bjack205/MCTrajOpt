using MathOptInterface
const MOI = MathOptInterface

struct DoublePendulumMOI{Nx,Nu,p} <: MOI.AbstractNLPEvaluator
    model::DoublePendulum
    state::RBD.StateCache{Float64, RBD.TypeSortedCollections.TypeSortedCollection{Tuple{Vector{RigidBodyDynamics.Joint{Float64, RigidBodyDynamics.Revolute{Float64}}}}, 1}}
    results::RBD.DynamicsResultCache{Float64}
    params::SimParams
    Qr::Diagonal{Float64, SVector{3, Float64}}
    Qq::Diagonal{Float64, SVector{4, Float64}}
    R::Diagonal{Float64, SVector{Nu, Float64}}
    r0::Vector{SVector{3,Float64}}  # initial position
    q0::Vector{SVector{4,Float64}}  # initial orientation
    rgoal::SVector{3,Float64}       # End-effector goal
    p_ee::SVector{3,Float64}        # Location of end-effector in body 2 frame
    goalcon::Bool                   # Use end-effector goal constraint
    Xref::Vector{SVector{Nx,Float64}}

    n_nlp::Int
    m_nlp::Int
    N::Int
    L::Int  # number of links
    p::Int  # number of constraints per joint
    ncons::Dict{Symbol,Int}
    rinds::Matrix{SVector{3,Int}}   # N × L matrix of position indices
    qinds::Matrix{SVector{4,Int}}
    xinds::Vector{SVector{Nx,Int}}
    uinds::Vector{SVector{Nu,Int}}
    λinds::Vector{SVector{p,Int}}
    blocks::BlockViews
end
ismincoord(::DoublePendulumMOI{Nx}) where Nx = Nx == 4

function DoublePendulumMOI(model::DoublePendulum, params::SimParams,
    Qr, Qq, R, x0, Xref; rf = nothing, p_ee = SA[0,0,0.5], minimalcoords::Bool=false
)
    # Build RBD model
    mech = builddoublependulum([model.b1, model.b2], [model.joint0, model.joint1], gravity=model.gravity)
    state = RBD.StateCache(mech)
    cache = RBD.DynamicsResultCache(mech)

    N = params.N
    L = 2
    n = minimalcoords ? 2 * L :  7 * L
    m = control_dim(model)      # controls
    p = minimalcoords ? 0 : 5L  # constraint forces

    ri = SA[1,2,3]
    qi = SA[4,5,6,7]
    Nz = n + m + p
    rinds = [ri .+ ((k-1)*Nz + (i-1)*7) for k = 1:N, i = 1:L]
    qinds = [qi .+ ((k-1)*Nz + (i-1)*7) for k = 1:N, i = 1:L]
    # xinds = [[ri; qi] for (ri,qi) in zip(rinds,qinds)]
    # xinds = [[x1; x2] for (x1,x2) in eachrow(xinds)]
    xinds = SVector{n}.([(k-1)*Nz .+ (1:n) for k = 1:N])
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
    if minimalcoords
        m_nlp = (N-1)*2*L + p_goal
    else
        m_nlp = p_del + p_joints + p_quatnorm + p_goal
    end
    ncons = Dict(:DEL=>p_del, :joints=>p_joints, :quatnorm=>p_quatnorm)

    blocks = BlockViews(m_nlp, n_nlp)

    prob = DoublePendulumMOI{n,m,p}(model, state, cache, params, Qr, Qq, R, r0, q0, rf, 
        p_ee, goalcon, Xref,
        n_nlp, m_nlp, N, L, p, ncons, rinds, qinds, xinds, uinds, λinds, blocks
    )
    initialize_sparsity!(prob)
    prob
end

function getjacobiandensity(prob::DoublePendulumMOI)
    nnz = length(MOI.jacobian_structure(prob))
    return nnz / (prob.n_nlp * prob.m_nlp)
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
end

function MOI.eval_objective(prob::DoublePendulumMOI, z)
    model = prob.model
    J = 0.0
    rinds, qinds = prob.rinds, prob.qinds
    xinds, uinds = prob.xinds, prob.uinds
    for k = 1:prob.N
        if ismincoord(prob)
            θk = z[xinds[k]]
            θref = prob.Xref[k]
            # xk = min2max(model, θk) 
            # xref = min2max(model, θref) 
            p_ee = getendeffectorposition(prob, θk[1:2])
            e = p_ee - prob.rgoal
            J += 0.5 * e'prob.Qr*e
            # dθ = θk - θref
            # J += 0.5 * dθ'prob.Qq*dθ
        else
            xk = z[xinds[k]]
            xref = prob.Xref[k]
        for j = 2:prob.L
            r = gettran(model, xk, j)
            q = getquat(model, xk, j)
            e = r + Amat(q)*prob.p_ee - prob.rgoal
            J += 0.5 * e'prob.Qr*e
            # rref = gettran(model, xref, j)
            # qref = getquat(model, xref, j)
            # dr = r - rref
            # dq = q - qref
            # c[ci] = rf + Amat(qf)*prob.p_ee - prob.rgoal
            # J += 0.5 * (dr'prob.Qr*dr + dq'prob.Qq*dq)
        end
        end
        if k < prob.N
            u = z[uinds[k]]
            J += 0.5 * u'prob.R*u
        end
    end
    return J
end

function MOI.eval_objective_gradient(prob::DoublePendulumMOI, grad_f, x)
    obj(x) = MOI.eval_objective(prob, x)
    # ForwardDiff.gradient!(grad_f, obj, x)  # TODO: use a cache
    FiniteDiff.finite_difference_gradient!(grad_f, obj, x)
    return
end

function getendeffectorposition(prob::DoublePendulumMOI{Nx}, x) where Nx
    if ismincoord(prob)
        if length(x) == 4
            theta = x[SA[1,2]]
        else
            theta = x
        end
        state = prob.state[eltype(theta)]
        RBD.set_configuration!(state, theta)
        elbow = RBD.findjoint(state.mechanism, "elbow") 
        p_ee = RBD.transform(state, RBD.Point3D(RBD.frame_after(elbow), prob.p_ee), RBD.root_frame(state.mechanism))
    else
        rf, qf = x[prob.rinds[1,end]], x[prob.qinds[1,end]]
        return rf + Amat(qf)*prob.p_ee
    end
    return p_ee.v
end

function implicitmidpoint!(prob::DoublePendulumMOI, c, θ1, u1, θ2, h)
    θmid = (θ1 + θ2) / 2
    T = eltype(θ2)
    results = prob.results[T]
    state = prob.state[T]
    τ = SA[zero(T); u1[1]]
    # c_ = zeros(T, length(θ2))

    RBD.dynamics!(c, results, state, θmid, τ)
    c .= θ1 + h*c - θ2
end

function dircol_constraints(prob::DoublePendulumMOI, c, z)
    model = prob.model
    h = prob.params.h
    xinds, uinds = prob.xinds, prob.uinds

    off = 0
    ci = 1:4
    for (i,k) in enumerate(1:prob.N-1)
        θ1 = z[xinds[k]]
        u1 = z[uinds[k]]
        θ2 = z[xinds[k+1]]
        c[ci] .= implicitmidpoint(model, θ1, u1, θ2, h)
        # cview = view(c, ci)
        # implicitmidpoint!(prob, cview, θ1, u1, θ2)
        ci = ci .+ 4
        off += 4
    end

    if prob.goalcon
        ci = off .+ (1:3)
        θf = z[xinds[end]][SA[1,2]]  # get angles
        p_ee = getendeffectorposition(prob, θf)
        c[ci] = p_ee - prob.rgoal 
    end

end

function ∇dircol_constraints(prob::DoublePendulumMOI, jac, z)
    h = prob.params.h
    model = prob.model
    jac .= 0
    J0 = NonzerosVector(jac, prob.blocks)
    xinds, uinds = prob.xinds, prob.uinds

    off = 0
    ci = 1:4
    for (i,k) in enumerate(1:prob.N-1)
        θ1 = z[xinds[k]]
        u1 = z[uinds[k]]
        θ2 = z[xinds[k+1]]

        # cols = xinds[k][1] : xinds[k+1][end]
        # f!(y,z) = implicitmidpoint!(prob, y, z[xinds[1]], z[uinds[1]], z[xinds[2]])
        # Jview = view(J0, ci, cols) 
        # ForwardDiff.jacobian!(Jview, f!, ctmp, [θ1; u1; θ2])
        f(z) = implicitmidpoint(model, z[xinds[1]], z[uinds[1]], z[xinds[2]], h)
        ∇f = ForwardDiff.jacobian(f, [θ1; u1; θ2])
        J0[ci, xinds[k]] .= ∇f[:,xinds[1]] 
        J0[ci, uinds[k]] .= ∇f[:,uinds[1]] 
        J0[ci, xinds[k+1]] .= ∇f[:,xinds[2]] 
        ci = ci .+ 4
        off += 4
    end

    if prob.goalcon
        ci = off .+ (1:3)
        θf = z[xinds[end]][SA[1,2]]  # get angles
        J0[ci, xinds[end][1:2]] = ForwardDiff.jacobian(x->getendeffectorposition(prob, x), θf)
    end
end

function MOI.eval_constraint(prob::DoublePendulumMOI, c, z)
    if ismincoord(prob)
        return dircol_constraints(prob, c, z)
    end

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

        # Compute the Discrete Euler-Lagrange constraint
        λ = z[λinds[i]]
        DEL!(prob.model, c, x1, x2, x3, λ, u1, u2, h, yi=ci[1])
        
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
    if ismincoord(prob)
        return ∇dircol_constraints(prob, jac, z)
    end

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
        x = z[xinds[k]]
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
end

function ipopt_solve(prob::DoublePendulumMOI, x0; tol=1e-6, c_tol=1e-6, max_iter=1_000, goal_tol=1e-2)

    n_nlp = prob.n_nlp 
    m_nlp = prob.m_nlp 
    
    x_l = fill(-Inf,n_nlp)
    x_u = fill(+Inf,n_nlp)
    if ismincoord(prob)
        x_l[1:4] = [-pi,0,0,0]
        x_u[1:4] = [-pi,0,0,0]
        x_l[prob.xinds[end]] = zeros(4)
        x_u[prob.xinds[end]] = zeros(4)
    else
        for k = 1:2
            for j = 1:prob.L
                x_l[prob.rinds[k,j]] = prob.r0[j]
                x_u[prob.rinds[k,j]] = prob.r0[j]
                x_l[prob.qinds[k,j]] = prob.q0[j]
                x_u[prob.qinds[k,j]] = prob.q0[j]
            end
        end
    end
    # x_l[1:3] = prob.r0[1]
    # x_u[1:3] = prob.r0[1]
    # x_l[4:7] = prob.q0[1]
    # x_u[4:7] = prob.q0[1]
    # x_l[8:10] = prob.r0[2]
    # x_u[8:10] = prob.r0[2]
    # x_l[11:14] = prob.q0[2]
    # x_u[11:14] = prob.q0[2]
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

function randtraj(prob::DoublePendulumMOI)
    z0 = zeros(prob.n_nlp)
    for k = 1:prob.N
        for j = 1:prob.L
            if ismincoord(prob)
                z0[prob.xinds[k]] = @SVector randn(4)
            else
                z0[prob.rinds[k,j]] = @SVector randn(3)
                z0[prob.qinds[k,j]] = normalize(@SVector randn(4))
            end
        end
        if k < prob.N
            z0[prob.uinds[k]] = @SVector randn(control_dim(prob.model)) 
            if !ismincoord(prob)
                z0[prob.λinds[k]] = @SVector randn(10)
            z0[prob.λinds[k]] = @SVector randn(10) 
                z0[prob.λinds[k]] = @SVector randn(10)
            end
        end
    end
    return z0
end

function simulate_mincoord(prob::DoublePendulumMOI, params::SimParams, U, x0; newton_iters=20, tol=1e-12)
    h = params.h
    X = [copy(x0) for k = 1:params.N]
    X[1] = x0

    r = zeros(4)
    H = zeros(4,4)

    for k = 1:params.N-1
        X[k+1] = copy(X[k])

        res!(y,x) = implicitmidpoint!(prob, y, X[k], U[k], x, h)
        # println("Time step $k")
        for i = 1:newton_iters
            res!(r, X[k+1])
            # println("  res = $(norm(r,Inf))")
            if norm(r,Inf) < tol
                break
            end
            ForwardDiff.jacobian!(H, res!, r, X[k+1])
            # H = ForwardDiff.jacobian(res, X[k+1])
            X[k+1] += -(H\r)

            if i == newton_iters
                @warn "Hit max Newton iterations at time step $k"
            end
        end
    end
    return X
end