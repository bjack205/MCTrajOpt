using GeometryBasics

struct CylindricalBody
    density::Float64
    radius::Float64
    length::Float64
end

mass(body::CylindricalBody) = π*body.radius^2 * body.length * body.density

function inertiatensor(body::CylindricalBody)
    r,h = body.radius, body.length
    m = mass(body)
    Ixx = Iyy = m*(3&r^2 + h^2) / 12
    Izz = m*r^2/2
    SA[
        Ixx 0 0
        0 Iyy 0
        0 0 Izz
    ]
end

function geometry(body::CylindricalBody)
    h = body.length
    Cylinder(Point3(0,0,-h/2), Point3(0,0,h/2), body.radius)
end