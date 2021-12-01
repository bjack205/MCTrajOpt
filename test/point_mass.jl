import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using MCTrajOpt
using ForwardDiff
using StaticArrays
using LinearAlgebra
using Test
using Random
using BenchmarkTools
using MathOptInterface
using Quaternions
const MOI = MathOptInterface 

##
body = RigidBody(1.0, Diagonal([0.1, 1, 1]))
sim = SimParams(1.0, 0.25)

Qcost = Diagonal(SA_F64[1,1,1])
Rcost = Diagonal(@SVector fill(0.1, 3))

x0 = SA[0,0,0.]
xf = SA[1,2,3.]

# Create the NLP problem
prob = MCTrajOpt.PointMassIpopt(body, sim, Qcost, Rcost, x0, xf)
z0 = [repeat([x0; zeros(3)], sim.N-1); x0]
prob.n_nlp
prob.m_nlp
prob.N

grad_f = zeros(27)
J = zeros(9*27)
c = zeros(9)
MOI.eval_objective(prob, z0)
MOI.eval_objective_gradient(prob, grad_f, z0)
MOI.eval_constraint(prob, c, z0)
MOI.eval_constraint_jacobian(prob, J, z0)
zsol, = MCTrajOpt.ipopt_solve(prob, z0)
zsol[25:27]