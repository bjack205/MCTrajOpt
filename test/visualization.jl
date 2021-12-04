using MeshCat, GeometryBasics, CoordinateTransformations, Rotations, Colors

function launchvis(body::TwoBody, x0)
    vis = Visualizer()
    geom = Cylinder(Point(-0.5,0,0), Point(0.5,0,0), 0.2)
    setobject!(vis["body1"], geom, MeshPhongMaterial(color=colorant"green"))
    setobject!(vis["body2"], geom, MeshPhongMaterial(color=colorant"green"))
    setobject!(vis["body1"]["com"], Triad())
    setobject!(vis["body2"]["com"], Triad())
    let x = x0
        x1,x2 = MCTrajOpt.splitstate(body, x)
        ri,qi = SA[1,2,3], SA[4,5,6,7]
        r1,r2 = x1[ri], x2[ri]
        q1,q2 = x1[qi], x2[qi]
        settransform!(vis["body1"], compose(Translation(r1), LinearMap(UnitQuaternion(q1))))
        settransform!(vis["body2"], compose(Translation(r2), LinearMap(UnitQuaternion(q2))))
    end
    open(vis)
    return vis
end

function visualize!(vis, model::TwoBody, x::AbstractVector{<:Real})
    x1,x2 = MCTrajOpt.splitstate(model, x)
    ri,qi = SA[1,2,3], SA[4,5,6,7]
    r1,r2 = x1[ri], x2[ri]
    q1,q2 = x1[qi], x2[qi]
    settransform!(vis["body1"], compose(Translation(r1), LinearMap(UnitQuaternion(q1))))
    settransform!(vis["body2"], compose(Translation(r2), LinearMap(UnitQuaternion(q2))))
end

function visualize!(vis, model, X::Vector{<:AbstractVector}, params::SimParams)
    anim = MeshCat.Animation(floor(Int,1/params.h))
    for k = 1:params.N
        atframe(anim, k) do
            visualize!(vis, model, X[k])
        end
    end
    setanimation!(vis, anim)
end

