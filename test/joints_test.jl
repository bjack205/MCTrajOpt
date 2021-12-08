
# Jacobian of Jacobian-vector transpose product 
let joint = model.joint0
    r_1 = @SVector randn(3)
    r_2 = @SVector randn(3)
    q_1 = normalize(@SVector randn(4))
    q_2 = normalize(@SVector randn(4))
    λ = @SVector randn(5)
    ir1 = 1:3
    iq1 = 4:7
    ir2 = ir1 .+ 7
    iq2 = iq1 .+ 7
    x = [r_1; q_1; r_2; q_2]
    jac = zeros(5, 14)
    jtvp(x) = MC.joint_constraint(joint, x[ir1], x[iq1], x[ir2], x[iq2])'λ
    ForwardDiff.gradient(jtvp, x) ≈ vcat(MC.jtvp_joint_constraint(joint, r_1, q_1, r_2, q_2, λ)...)
    hess0 = ForwardDiff.hessian(jtvp, x)
    hess = zeros(14, 14)
    # MC.∇²joint_constraint(joint, x[ir1], x[iq1], x[ir2], x[iq2], λ)
    MC.∇²joint_constraint!(joint, hess, x[ir1], x[iq1], x[ir2], x[iq2], λ)
    hess ≈ hess0
end

let joint = model.joint0
    r_1 = @SVector randn(3)
    r_2 = @SVector randn(3)
    q_1 = normalize(@SVector randn(4))
    q_2 = normalize(@SVector randn(4))
    
    dr_1, dq_1, dr_2, dq_2 = MC.∇joint_constraint(joint, r_1, q_1, r_2, q_2)
    @test ForwardDiff.jacobian(x->MC.joint_constraint(joint, x, q_1, r_2, q_2), r_1) ≈ dr_1
    @test ForwardDiff.jacobian(x->MC.joint_constraint(joint, r_1, x, r_2, q_2), q_1) ≈ dq_1
    @test ForwardDiff.jacobian(x->MC.joint_constraint(joint, r_1, q_1, x, q_2), r_2) ≈ dr_2
    @test ForwardDiff.jacobian(x->MC.joint_constraint(joint, r_1, q_1, r_2, x), q_2) ≈ dq_2
end


let
    p1 = @SVector randn(3)
    p2 = @SVector randn(3)
    ax = normalize(@SVector randn(3))
    joint = RevoluteJoint(p1, p2, ax)

    r_1 = @SVector randn(3)
    r_2 = @SVector randn(3)
    q_1 = normalize(@SVector randn(4))
    q_2 = normalize(@SVector randn(4))
    u = randn()
    args = (r_1, q_1, r_2, q_2, u)
    
    @test MC.∇force11(joint, args...)[1] ≈ ForwardDiff.jacobian(x->MC.force1(joint, x, q_1, r_2, q_2, u), r_1)
    @test MC.∇force11(joint, args...)[2] ≈ ForwardDiff.jacobian(x->MC.force1(joint, r_1, x, r_2, q_2, u), q_1)
    @test MC.∇force12(joint, args...)[1] ≈ ForwardDiff.jacobian(x->MC.force1(joint, r_1, q_1, x, q_2, u), r_2)
    @test MC.∇force12(joint, args...)[2] ≈ ForwardDiff.jacobian(x->MC.force1(joint, r_1, q_1, r_2, x, u), q_2)

    @test MC.∇force21(joint, args...)[1] ≈ ForwardDiff.jacobian(x->MC.force2(joint, x, q_1, r_2, q_2, u), r_1)
    @test MC.∇force21(joint, args...)[2] ≈ ForwardDiff.jacobian(x->MC.force2(joint, r_1, x, r_2, q_2, u), q_1)
    @test MC.∇force22(joint, args...)[1] ≈ ForwardDiff.jacobian(x->MC.force2(joint, r_1, q_1, x, q_2, u), r_2)
    @test MC.∇force22(joint, args...)[2] ≈ ForwardDiff.jacobian(x->MC.force2(joint, r_1, q_1, r_2, x, u), q_2)

    @test MC.∇torque11(joint, args...)[1] ≈ ForwardDiff.jacobian(x->MC.torque1(joint, x, q_1, r_2, q_2, u), r_1)
    @test MC.∇torque11(joint, args...)[2] ≈ ForwardDiff.jacobian(x->MC.torque1(joint, r_1, x, r_2, q_2, u), q_1)
    @test MC.∇torque12(joint, args...)[1] ≈ ForwardDiff.jacobian(x->MC.torque1(joint, r_1, q_1, x, q_2, u), r_2)
    @test MC.∇torque12(joint, args...)[2] ≈ ForwardDiff.jacobian(x->MC.torque1(joint, r_1, q_1, r_2, x, u), q_2)

    @test MC.∇torque21(joint, args...)[1] ≈ ForwardDiff.jacobian(x->MC.torque2(joint, x, q_1, r_2, q_2, u), r_1)
    @test MC.∇torque21(joint, args...)[2] ≈ ForwardDiff.jacobian(x->MC.torque2(joint, r_1, x, r_2, q_2, u), q_1)
    @test MC.∇torque22(joint, args...)[1] ≈ ForwardDiff.jacobian(x->MC.torque2(joint, r_1, q_1, x, q_2, u), r_2)
    @test MC.∇torque22(joint, args...)[2] ≈ ForwardDiff.jacobian(x->MC.torque2(joint, r_1, q_1, r_2, x, u), q_2)
end