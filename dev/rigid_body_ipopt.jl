using MathOptInterface
using Ipopt
const MOI = MathOptInterface 

struct MatrixBlock
    m::Int  # number of rows
    n::Int  # number of columns
    off::Int  # location in vector
    i1::UnitRange{Int}  # rows in sparse matrix
    i2::UnitRange{Int}  # columns in sparse matrix
end
function MatrixBlock(m::Int, n::Int, off::Int, i1::AbstractVector, i2::AbstractVector)
    MatrixBlock(m, n, off, i1[begin]:i1[end], i2[begin]:i2[end])
end

Base.length(block::MatrixBlock) = prod(size(block))
Base.size(block::MatrixBlock) = (block.m, block.n)

function copyblock!(dest::AbstractVector, src::AbstractMatrix, block::MatrixBlock)
    copyto!(dest, block.off, src, 1, length(block))
end

struct ProblemMOI <: MOI.AbstractNLPEvaluator
    # Problem description
    body::RigidBody
    params::SimParams
    Q::Diagonal{Float64, SVector{3, Float64}}
    R::Diagonal{Float64, SVector{6, Float64}}
    r0::SVector{3,Float64}  # initial position
    q0::SVector{4,Float64}  # initial orientation
    r1::SVector{3,Float64}  # next initial position
    q1::SVector{4,Float64}  # next initial orientation
    rf::SVector{3,Float64}  # goal position
    qf::SVector{4,Float64}  # goal orientation

    # Internal data
    n_nlp::Int  # number of primals
    m_nlp::Int  # number of duals (constraints)
    N::Int      # horizon length
    blocks::Vector{Dict{Symbol,MatrixBlock}}
    rinds::Vector{SVector{3,Int}}
    qinds::Vector{SVector{4,Int}}
    xinds::Vector{SVector{7,Int}}
    uinds::Vector{SVector{6,Int}}
    cinds::Vector{SVector{6,Int}}
    x_lb::Vector{Float64}  # primal lower bounds
    x_ub::Vector{Float64}  # primal upper bounds
    c_lb::Vector{Float64}  # constraint lower bounds
    c_ub::Vector{Float64}  # constraint upper bounds
end

function ProblemMOI(body::RigidBody, params::SimParams, Q, R, x0, x1, xf)
    N = params.N
    ri = SA[1,2,3]
    qi = SA[4,5,6,7]
    r0,q0 = x0[ri], x0[qi]
    r1,q1 = x1[ri], x1[qi]
    rf,qf = xf[ri], xf[qi]
    n = 7
    m = 6
    p = 6
    xinds = SVector{n}.([(k-1)*(n+m) .+ (1:n) for k = 1:N])
    uinds = SVector{m}.([(k-1)*(n+m) + n .+ (1:m) for k = 1:N-1])
    rinds = [xi[ri] for xi in xinds]
    qinds = [xi[qi] for xi in xinds]
    cinds = SVector{p}.(eachcol(reshape(1:p*(N-2), p, N-2)))

    # Set bounds
    n_nlp = sum(length.(xinds)) + sum(length.(uinds))
    m_nlp = sum(length.(cinds))
    x_lb = fill(-Inf, n_nlp)
    x_ub = fill(+Inf, n_nlp)
    c_lb = fill(0.0, m_nlp)
    c_ub = fill(0.0, m_nlp)

    # Set initial conditions
    x_lb[1:2n] = [x0; x1]
    x_ub[1:2n] = [x0; x1]

    off = 1
    blocks = map(enumerate(2:N-1)) do (i,k) 
        x1 = MatrixBlock(p, n, off, cinds[i], xinds[k-1]); off += length(x1)
        u1 = MatrixBlock(p, m, off, cinds[i], uinds[k-1]); off += length(u1)
        x2 = MatrixBlock(p, n, off, cinds[i], xinds[k]);   off += length(x2)
        u2 = MatrixBlock(p, m, off, cinds[i], uinds[k]);   off += length(u2)
        x3 = MatrixBlock(p, n, off, cinds[i], xinds[k+1]); off += length(x3)
        Dict(:x1=>x1, :u1=>u1, :x2=>x2, :u2=>u2, :x3=>x3)
    end

    ProblemMOI(body, params, Q, R, r0, q0, r1, q1, rf, qf, n_nlp, m_nlp, N,
        blocks, rinds, qinds, xinds, uinds, cinds, x_lb, x_ub, c_lb, c_ub
    )
end

function packZ!(prob::ProblemMOI, z, X, U)
    for k = 1:prob.N
        z[prob.xinds[k]] .= X[k]
        if k < prob.N 
            z[prob.uinds[k]] .= U[k]
        end
    end
    return z
end

function MOI.eval_objective(prob::ProblemMOI, x)
    J = 0.0
    # rf = prob.rf
    # qf = prob.qf
    # Q,R = prob.Q, prob.R
    # r1 = x[1:3] - rf
    # q1 = x[4:7] - qf
    # u1 = x[8:13]
    # r2 = x[14:16] - rf
    # q2 = x[17:20] - qf
    # u2 = x[21:26]
    # r3 = x[27:29] - rf
    # q3 = x[30:33] - qf
    # u3 = x[34:39]
    # r4 = x[40:42] - rf
    # q4 = x[43:46] - qf
    # u4 = x[47:52]
    # r5 = x[53:55] - rf
    # q5 = x[56:59] - qf

    # J += 0.5*(r1'Q*r1 + q1'q1 + u1'R*u1)
    # J += 0.5*(r2'Q*r2 + q2'q2 + u2'R*u2)
    # J += 0.5*(r3'Q*r3 + q3'q3 + u3'R*u3)
    # J += 0.5*(r4'Q*r4 + q4'q4 + u4'R*u4)
    # J += 0.5*(r5'Q*r5 + q5'q5) * 100
    # return J

    Q,R = prob.Q, prob.R
    rinds, qinds, uinds = prob.rinds, prob.qinds, prob.uinds
    for k = 1:prob.N
        r = x[rinds[k]]
        q = x[qinds[k]]
        dr = r - prob.rf
        dq = q - prob.qf
        if k < prob.N
            u = x[uinds[k]]
            J += 0.5 * u'R*u
            s = 1.0
        else
            s = 10.0
        end
        J += 0.5 * (dr'Q*dr + dq'dq) * s
    end
    return J
end

function MOI.eval_objective_gradient(prob::ProblemMOI, grad_f, x)
    obj(x) = MOI.eval_objective(prob, x)
    ForwardDiff.gradient!(grad_f, obj, x)  # TODO: use a cache
    return
end

function MOI.eval_constraint(prob::ProblemMOI, c, x)
    h = prob.params.h
    J = prob.body.J
    mass = prob.body.mass
    # r1 = x[1:3]; q1 = x[4:7]
    # F1 = x[8:10]; T1 = x[11:13]
    # r2 = x[14:16]; q2 = x[17:20]
    # F2 = x[21:23]; T2 = x[24:26]
    # r3 = x[27:29]; q3 = x[30:33]
    # F3 = x[34:36]; T3 = x[37:39]
    # r4 = x[40:42]; q4 = x[43:46]
    # F4 = x[47:49]; T4 = x[50:52]
    # r5 = x[53:55]; q5 = x[56:59]

    # G2 = L(q2)'Hmat
    # G3 = L(q3)'Hmat
    # G4 = L(q4)'Hmat

    # c[1:3]   = mass / h * (r2-r1) - mass/h * (r3-r2) + h * (F1 + F2) / 2
    # c[4:6]   = (2/h) * G2'L(q1)*Hmat * J * Hmat'L(q1)'q2 + (2/h) * G2'Tmat*R(q3)'Hmat * J * Hmat'L(q2)'q3
    # c[7:9]   = mass / h * (r3-r2) - mass/h * (r4-r3) + h * (F2 + F3) / 2
    # c[10:12] = (2/h) * G3'L(q2)*Hmat * J * Hmat'L(q2)'q3 + (2/h) * G3'Tmat*R(q4)'Hmat * J * Hmat'L(q3)'q4
    # c[13:15] = mass / h * (r4-r3) - mass/h * (r5-r4) + h * (F3 + F4) / 2
    # c[16:18] = (2/h) * G4'L(q3)*Hmat * J * Hmat'L(q3)'q4 + (2/h) * G4'Tmat*R(q5)'Hmat * J * Hmat'L(q4)'q5

    rinds, qinds = prob.rinds, prob.qinds
    xinds, uinds = prob.xinds, prob.uinds
    cinds = prob.cinds
    h = prob.params.h
    inds1 = 1:3
    inds2 = 4:6
    for (i,k) in enumerate(2:prob.N-1)
        r1 = x[rinds[k-1]]
        q1 = x[qinds[k-1]]
        u1 = x[uinds[k-1]]
        r2 = x[rinds[k]]
        q2 = x[qinds[k]]
        u2 = x[uinds[k]]
        r3 = x[rinds[k+1]]
        q3 = x[qinds[k+1]]
        F1,T1 = u1[SA[1,2,3]], u1[SA[4,5,6]]
        F2,T2 = u2[SA[1,2,3]], u2[SA[4,5,6]]

        G2 = L(q2)'Hmat

        # Enforce the Discrete Euler-Lagrange equation to satisfy the dynamics
        # c[inds1] = mass / h * (r2-r1) - mass/h * (r3-r2) + h * (F1 + F2) / 2
        # c[inds2] = (2/h) * G2'L(q1)*Hmat * J * Hmat'L(q1)'q2 + (2/h) * G2'Tmat*R(q3)'Hmat * J * Hmat'L(q2)'q3 + h * (T1 + T2) / 2
        c[inds1] = DEL_trans(prob.body, r1, r2, r3, F1, F2, h)
        c[inds2] = DEL_rot(prob.body, q1, q2, q3, T1, T2, h)
        # x1 = [r1; q1]
        # x2 = [r2; q2]
        # x3 = [r3; q3]
        # c[cinds[i]] = DEL(prob.body, x1, x2, x3, u1, u2, h)

        inds1 = inds1 .+ 6
        inds2 = inds2 .+ 6
    end
end

function MOI.eval_constraint_jacobian(prob::ProblemMOI, jac, x)
    J0 = reshape(jac, prob.m_nlp, prob.n_nlp)
    h = prob.params.h
    # c = zeros(eltype(x), prob.m_nlp)
    # ForwardDiff.jacobian!(J0, (c,z)->MOI.eval_constraint(prob, c, z), c, x)
    # return

    # J = zero(J0)
    jac .= 0
    J = J0
    rinds, qinds = prob.rinds, prob.qinds
    xinds, uinds = prob.xinds, prob.uinds
    cinds = prob.cinds
    for (i,k) in enumerate(2:prob.N-1)
        r1 = x[rinds[k-1]]
        q1 = x[qinds[k-1]]
        u1 = x[uinds[k-1]]
        r2 = x[rinds[k]]
        q2 = x[qinds[k]]
        u2 = x[uinds[k]]
        r3 = x[rinds[k+1]]
        q3 = x[qinds[k+1]]
        F1,T1 = u1[SA[1,2,3]], u1[SA[4,5,6]]
        F2,T2 = u2[SA[1,2,3]], u2[SA[4,5,6]]
        F1i = uinds[k-1][SA[1,2,3]]
        T1i = uinds[k-1][SA[4,5,6]]
        F2i = uinds[k][SA[1,2,3]]
        T2i = uinds[k][SA[4,5,6]]

        c1 = cinds[i][SA[1,2,3]]
        c2 = cinds[i][SA[4,5,6]]
        J[c1, rinds[k-1]] .= ForwardDiff.jacobian(r->DEL_trans(prob.body, r,r2,r3,F1,F2,h), r1)
        J[c1, rinds[k]]   .= ForwardDiff.jacobian(r->DEL_trans(prob.body, r1,r,r3,F1,F2,h), r2)
        J[c1, rinds[k+1]] .= ForwardDiff.jacobian(r->DEL_trans(prob.body, r1,r2,r,F1,F2,h), r3)
        J[c1, F1i] .= ForwardDiff.jacobian(F->DEL_trans(prob.body, r1,r2,r3,F,F2,h), F1)
        J[c1, F2i] .= ForwardDiff.jacobian(F->DEL_trans(prob.body, r1,r2,r3,F1,F,h), F2)

        J[c2, qinds[k-1]] .= ForwardDiff.jacobian(q->DEL_rot(prob.body, q,q2,q3,T1,T2,h), q1)
        J[c2, qinds[k]]   .= ForwardDiff.jacobian(q->DEL_rot(prob.body, q1,q,q3,T1,T2,h), q2)
        J[c2, qinds[k+1]] .= ForwardDiff.jacobian(q->DEL_rot(prob.body, q1,q2,q,T1,T2,h), q3)
        J[c2, T1i] .= ForwardDiff.jacobian(T->DEL_rot(prob.body, q1,q2,q3,T,T2,h), T1)
        J[c2, T2i] .= ForwardDiff.jacobian(T->DEL_rot(prob.body, q1,q2,q3,T1,T,h), T2)
    end

    return 

    xinds, uinds = prob.xinds, prob.uinds
    h = prob.params.h
    body = prob.body
    for (i,k) in enumerate(2:prob.N-1)
        x1 = x[xinds[k-1]]
        u1 = x[uinds[k-1]]
        x2 = x[xinds[k]]
        u2 = x[uinds[k]]
        x3 = x[xinds[k+1]]

        ∇x1 = ForwardDiff.jacobian(x->DEL(body, x,x2,x3, u1,u2, h), x1)
        ∇x2 = ForwardDiff.jacobian(x->DEL(body, x1,x,x3, u1,u2, h), x2)
        ∇x3 = ForwardDiff.jacobian(x->DEL(body, x1,x2,x, u1,u2, h), x3)
        ∇u1 = ForwardDiff.jacobian(u->DEL(body, x1,x2,x3, u,u2, h), u1)
        ∇u2 = ForwardDiff.jacobian(u->DEL(body, x1,x2,x3, u1,u, h), u2)

        copyblock!(jac, ∇x1, prob.blocks[i][:x1])
        copyblock!(jac, ∇u1, prob.blocks[i][:u1])
        copyblock!(jac, ∇x2, prob.blocks[i][:x2])
        copyblock!(jac, ∇u2, prob.blocks[i][:u2])
        copyblock!(jac, ∇x3, prob.blocks[i][:x3])
        
    end
end

MOI.features_available(prob::ProblemMOI) = [:Grad, :Jac]

MOI.initialize(prob::ProblemMOI, features) = nothing

function MOI.jacobian_structure(prob::ProblemMOI) 
    return vec(Tuple.(CartesianIndices((prob.m_nlp,prob.n_nlp))))

    inds = Tuple{Int,Int}[]
    for dict in prob.blocks
        for key in (:x1,:u1,:x2,:u2,:x3)
            append!(inds, Tuple.(CartesianIndices((dict[key].i1, dict[key].i2))))
        end
    end
    return inds
end

function ipopt_solve(prob::ProblemMOI, x0; tol=1e-6, c_tol=1e-6, max_iter=1_000)
    n_nlp = prob.n_nlp 
    m_nlp = prob.m_nlp 
    # x_l, x_u = prob.x_lb, prob.x_ub
    # c_l, c_u = prob.c_lb, prob.c_ub
    
    x_l = fill(-Inf,n_nlp)
    x_u = fill(+Inf,n_nlp)
    x_l[1:3] = prob.r0
    x_u[1:3] = prob.r0
    x_l[4:7] = prob.q0
    x_u[4:7] = prob.q0
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