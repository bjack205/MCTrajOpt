using Quaternions
using Rotations
using ForwardDiff

function qslerp(q1, q2, t, map=ExponentialMap()) 
    qpow(q2*inv(q1), t, map) * q1
end

function qpow(q, t, map=ExponentialMap())
    ϕ = inv(map)(q)
    map(t*ϕ)
end

function ∇qpow(q, t, map=ExponentialMap())
    ϕ = inv(map)(q)
    dϕ = Rotations.jacobian(inv(map), q)
    Rotations.jacobian(map, t*ϕ) * dϕ * t
end

function ∇qslerp1(q1, q2, t, map=ExponentialMap())
    q21 = q2/q1
    qt = qpow(q21, t, map)
    Rotations.lmult(qt) + Rotations.rmult(q1)*∇qpow(q21, t, map)*Rotations.lmult(q2)*Rotations.tmat()
end

function ∇qslerp2(q1, q2, t, map=ExponentialMap())
    q21 = q2/q1
    qt = qpow(q21, t, map)
    Rotations.rmult(q1)*∇qpow(q21, t, map)*Rotations.rmult(q1)'
end

ExponentialMap()

Quaternion(q::UnitQuaternion) = Quaternion(Rotations.params(q)...)
UnitQuaternion(q::Quaternion) = UnitQuaternion(q.s, q.v1, q.v2, q.v3)

# Sample input (rotate away from identity)
q1 = one(UnitQuaternion) 
θ = deg2rad(45)
q2 = q1 * expm(SA[0,0,1]*θ)

# Check that the slerp function is correct
q3 = UnitQuaternion(slerp(Quaternion(q1), Quaternion(q2), 0.5))
q4 = qslerp(q1, q2, 0.5, ExponentialMap())
AngleAxis(q3\q4).theta < 1e-10
AngleAxis(q4).theta ≈ θ / 2

# Pull out vectors for ForwardDiff
q1_ = Rotations.params(q1)
q2_ = Rotations.params(q2)
q = q1\q2
q_ = Rotations.params(q)
emap = ExponentialMap()

# Check qpow Jacobian
∇qpow(q, 0.5, emap) ≈ 
    ForwardDiff.jacobian(q->Rotations.params(qpow(UnitQuaternion(q, false), 0.5, emap)), q_)

# Check Slerp Jacobian
∇qslerp1(q1, q2, 0.5, emap) ≈ 
    ForwardDiff.jacobian(q->Rotations.params(qslerp(UnitQuaternion(q, false), q2, 0.5, emap)), q1_)

∇qslerp2(q1, q2, 0.5, emap) ≈ 
    ForwardDiff.jacobian(q->Rotations.params(qslerp(q1, UnitQuaternion(q, false), 0.5, emap)), q2_)
