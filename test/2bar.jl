import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using MCTrajOpt
using ForwardDiff
using StaticArrays
using LinearAlgebra
using Test
using Random
using BenchmarkTools
using FiniteDiff
include("visualization.jl")

using MathOptInterface
const MOI = MathOptInterface 
const MC = MCTrajOpt
using LinearAlgebra
using ForwardDiff
using Plots

## Parameters
m1 = 1.0
m2 = 1.0
J1 = Diagonal([0.1; 1.0; 1.0])
J2 = Diagonal([0.1; 1.0; 1.0])
ℓ1 = 1.0
ℓ2 = 1.0
body1 = RigidBody(m1, J1)
body2 = RigidBody(m2, J2)
model = DoublePendulum(body1, body2, gravity=true)
const GRAVITY = 9.81

M̄ = [m1*I(3) zeros(3,9);
     zeros(3,3) J1 zeros(3,6);
     zeros(3,6) m2*I(3) zeros(3,3);
     zeros(3,9) J2]
mass_matrix(model) ≈ M̄

#Simulation Parameters
h = 0.05 #20 Hz
Tf = 5.0 #final time (sec)
Thist = Array(0:h:Tf)
N = length(Thist)

sim = SimParams(Tf, h)
sim.thist ≈ Thist

function hat(ω)
    return [0 -ω[3] ω[2];
            ω[3] 0 -ω[1];
            -ω[2] ω[1] 0]
end

function L(Q)
    [Q[1] -Q[2:4]'; Q[2:4] Q[1]*I + hat(Q[2:4])]
end

function R(Q)
    [Q[1] -Q[2:4]'; Q[2:4] Q[1]*I - hat(Q[2:4])]
end

H = [zeros(1,3); I];

T = Diagonal([1.0; -1; -1; -1])

function G(Q)
    return L(Q)*H
end

function Ḡ(q)
    Q1 = q[4:7]
    Q2 = q[11:14]
    
    return [I zeros(3,9); zeros(4,3) G(Q1) zeros(4,6); zeros(3,6) I zeros(3,3); zeros(4,9) G(Q2)]
end

function DEL(q_1,q_2,q_3,λ,F1,F2)
    g = GRAVITY
    
    r1_1 = q_1[1:3]
    Q1_1 = q_1[4:7]
    r2_1 = q_1[8:10]
    Q2_1 = q_1[11:14]
    
    r1_2 = q_2[1:3]
    Q1_2 = q_2[4:7]
    r2_2 = q_2[8:10]
    Q2_2 = q_2[11:14]
    
    r1_3 = q_3[1:3]
    Q1_3 = q_3[4:7]
    r2_3 = q_3[8:10]
    Q2_3 = q_3[11:14]
    
    [(1/h)*m1*(r1_2-r1_1) - (1/h)*m1*(r1_3-r1_2) - m1*g*h*[1;0;0];
    (2.0/h)*G(Q1_2)'*L(Q1_1)*H*J1*H'*L(Q1_1)'*Q1_2 + (2.0/h)*G(Q1_2)'*T*R(Q1_3)'*H*J1*H'*L(Q1_2)'*Q1_3;
    (1/h)*m2*(r2_2-r2_1) - (1/h)*m2*(r2_3-r2_2) - m2*g*h*[1;0;0];
    (2.0/h)*G(Q2_2)'*L(Q2_1)*H*J2*H'*L(Q2_1)'*Q2_2 + (2.0/h)*G(Q2_2)'*T*R(Q2_3)'*H*J2*H'*L(Q2_2)'*Q2_3] + (h/2.0)*F1 + (h/2.0)*F2 + h*Dc(q_2)'*λ
end

function Dq3DEL(q_1,q_2,q_3,λ,F1,F2)
    ForwardDiff.jacobian(dq->DEL(q_1,q_2,dq,λ,F1,F2), q_3)*Ḡ(q_3)
end

function c(q)
    r1 = q[1:3]
    Q1 = q[4:7]
    r2 = q[8:10]
    Q2 = q[11:14]

    r0 = [0.0,0,0]
    Q0 = [1,0,0,0.]
    
    # @show r1 + H'*R(Q1)'*L(Q1)*H*[-0.5*ℓ1; 0; 0];
    [
        r0 + H'*R(Q0)'*L(Q0)*H*[-0.5; 0; 0] - r1  - H'*R(Q1)'*L(Q1)*H*[-0.5*ℓ1; 0; 0];
        [0 1 0 0; 0 0 1 0]*L(Q1)'*Q0
        r1 + H'*R(Q1)'*L(Q1)*H*[0.5*ℓ1; 0; 0] - r2 - H'*R(Q2)'*L(Q2)*H*[-0.5*ℓ2; 0; 0];
        [0 1 0 0; 0 0 1 0]*L(Q1)'*Q2
    ]
end

function Dc(q)
    ForwardDiff.jacobian(dq->c(dq),q)*Ḡ(q)
end

# Test functions
x1 = MC.randstate(model)
x2 = MC.randstate(model)
x3 = MC.randstate(model)
λ = @SVector randn(10)
F1 = @SVector randn(12)
F2 = @SVector randn(12)

@test Ḡ(x1) ≈ MC.errstate_jacobian(model, x1)
@test DEL(x1, x2, x3, λ, F1, F2) ≈ MC.DEL(model, x1, x2, x3, λ, F1, F2, h)
@test c(x2) ≈ MC.joint_constraints(model, x2)
@test Dc(x2) ≈ MC.∇joint_constraints(model, x2) * MC.errstate_jacobian(model, x2)
@test MC.∇DEL3(model, x1, x2, x3, λ, F1, F2, h) ≈ Dq3DEL(x1, x2, x3, λ, F1, F2)

#initial conditions
r1_0 = zeros(3)
r2_0 = [1.0; 0; 0]
Q1_0 = [1.0; 0; 0; 0]
Q2_0 = [1.0; 0; 0; 0]

q_0 = [r1_0; Q1_0; r2_0; Q2_0]

#Torque input at joint
uhist = 0.5*[0; 0; ones(19); -ones(40); zeros(10); ones(40); -ones(10)];

#Corresponding F
Fhist = zeros(12,N)
for k = 1:N
    Fhist[:,k] = [0;0;0; 0;0;-uhist[k]; 0;0;0; 0;0;uhist[k]]
end
F = [SVector{12}(F) for F in eachcol(Fhist)] 

## Simulate

function dp_simulate(model::DoublePendulum, params::SimParams, Fhist, q_0; newton_iters=20, tol=1e-12)
    N = params.N
    h = params.h

    #Simulate
    qhist = zeros(14,N)
    qhist[:,1] .= q_0
    qhist[:,2] .= q_0
    p = 10

    for k = 2:(N-1)
        
        #Initial guess
        qhist[:,k+1] .= qhist[:,k]
        λ = zeros(p)
        
        e = [DEL(qhist[:,k-1],qhist[:,k],qhist[:,k+1],λ,Fhist[:,k-1],Fhist[:,k]); c(qhist[:,k+1])]
        
        while maximum(abs.(e)) > tol 
            D = Dq3DEL(qhist[:,k-1],qhist[:,k],qhist[:,k+1],λ,Fhist[:,k-1],Fhist[:,k])
            C2 = Dc(qhist[:,k])
            C3 = Dc(qhist[:,k+1])
            
            Δ = -[D h*C2'; C3 zeros(p,p)]\e
            
            qhist[1:3,k+1] .= qhist[1:3,k+1] + Δ[1:3]
            qhist[4:7,k+1] .= L(qhist[4:7,k+1])*[sqrt(1-Δ[4:6]'*Δ[4:6]); Δ[4:6]]
            qhist[8:10,k+1] .= qhist[8:10,k+1] + Δ[7:9]
            qhist[11:14,k+1] .= L(qhist[11:14,k+1])*[sqrt(1-Δ[10:12]'*Δ[10:12]); Δ[10:12]]
            
            λ .= λ + Δ[12 .+ (1:p)]
            
            e = [DEL(qhist[:,k-1],qhist[:,k],qhist[:,k+1],λ,Fhist[:,k-1],Fhist[:,k]); c(qhist[:,k+1])]
        end
        
    end
    return qhist
end

qhist = dp_simulate(model, sim, Fhist, q_0)
Xsim2 = MC.simulate(model, sim, F, q_0)

# Visualization
# Xsim0 = [SVector{14}(x) for x in eachcol(qhist)]
Xsim = [SVector{14}(x) for x in eachcol(qhist)]
x0 = q_0
# vis = launchvis(model, x0)
visualize!(vis, model, Xsim, sim)
println(norm(Xsim - Xsim2,Inf))
# Xsim - Xsim0