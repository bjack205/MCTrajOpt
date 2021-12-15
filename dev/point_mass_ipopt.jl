
struct PointMassIpopt <: MOI.AbstractNLPEvaluator
    # Problem description
    body::RigidBody
    params::SimParams
    Q::Diagonal{Float64, SVector{3, Float64}}
    R::Diagonal{Float64, SVector{3, Float64}}
    x0::SVector{3,Float64}
    xf::SVector{3,Float64}

    n_nlp::Int
    m_nlp::Int
    N::Int
    xinds::Vector{SVector{3,Int}}
    uinds::Vector{SVector{3,Int}}
    cinds::Vector{SVector{3,Int}}
end

function PointMassIpopt(body::RigidBody, params::SimParams, Q, R, x0, xf)
    N = params.N 
    n,m = 3,3
    xinds = SVector{3}.([(1:3) .+ (k-1)*(n+m) for k = 1:N])
    uinds = SVector{3}.([(1:3) .+ (k-1)*(n+m) .+ n for k = 1:N-1])
    cinds = SVector{3}.([(1:3) .+ (k-1)*3 for k = 1:N-2])
    n_nlp = sum(length.(xinds)) + sum(length.(uinds))
    m_nlp = sum(length.(cinds))
    PointMassIpopt(body, params, Q, R, x0, xf, n_nlp, m_nlp, N, xinds, uinds, cinds)
end

function MOI.eval_objective(prob::PointMassIpopt, x)
    xf = prob.xf
    x1 = x[1:3] - xf
    u1 = x[4:6]
    x2 = x[7:9] - xf
    u2 = x[10:12]
    x3 = x[13:15] - xf
    u3 = x[16:18]
    x4 = x[19:21] - xf
    u4 = x[22:24]
    x5 = x[25:27] - xf
    Q,R = prob.Q, prob.R
    Qf = 100 * Q
    J = 0.0
    J += 0.5*(x1'Q*x1 + u1'R*u1)
    J += 0.5*(x2'Q*x2 + u2'R*u2)
    J += 0.5*(x3'Q*x3 + u3'R*u3)
    J += 0.5*(x4'Q*x4 + u4'R*u4)
    J += 0.5*(x5'Qf*x5)
    return J
end

function MOI.eval_objective_gradient(prob::PointMassIpopt, grad_f, x)
    obj(x) = MOI.eval_objective(prob, x)
    ForwardDiff.gradient!(grad_f, obj, x)  # TODO: use a cache
end

function MOI.eval_constraint(prob::PointMassIpopt, c, x)
    h = prob.params.h
    mass = prob.body.mass
    x1 = x[1:3]
    u1 = x[4:6]
    x2 = x[7:9]
    u2 = x[10:12]
    x3 = x[13:15]
    u3 = x[16:18]
    x4 = x[19:21]
    u4 = x[22:24]
    x5 = x[25:27]
    c[1:3] = mass / h * (x2-x1) - mass/h * (x3-x2) + h * (u1 + u2) / 2
    c[4:6] = mass / h * (x3-x2) - mass/h * (x4-x3) + h * (u2 + u3) / 2
    c[7:9] = mass / h * (x4-x3) - mass/h * (x5-x4) + h * (u3 + u4) / 2
    return
end

function MOI.eval_constraint_jacobian(prob::PointMassIpopt, jac, x)
    J = reshape(jac, prob.m_nlp, prob.n_nlp)
    c = zeros(eltype(x), prob.m_nlp)
    ForwardDiff.jacobian!(J, (c,z)->MOI.eval_constraint(prob, c, z), c, x)
    return
end

MOI.features_available(prob::PointMassIpopt) = [:Grad, :Jac]
MOI.initialize(prob::PointMassIpopt, features) = nothing

function MOI.jacobian_structure(prob::PointMassIpopt)
    vec(Tuple.(CartesianIndices((prob.m_nlp,prob.n_nlp))))
end

function ipopt_solve(prob::PointMassIpopt, x0; tol=1e-6, c_tol=1e-6, max_iter=1_000)
    n_nlp = prob.n_nlp 
    m_nlp = prob.m_nlp 
    
    x_l = fill(-Inf,n_nlp)
    x_u = fill(+Inf,n_nlp)
    x_l[1:3] = prob.x0
    x_u[1:3] = prob.x0
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