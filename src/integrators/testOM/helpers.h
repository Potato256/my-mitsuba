#pragma once
#include <mitsuba/render/scene.h>

MTS_NAMESPACE_BEGIN

inline void direct2uv(const Vector3& v, Point2& uv) {
    Float r = sqrt(1 - v.z);
    Float phi = atan2(v.y, v.x);
    if (r == 0)
    {
        uv.x = 0;
        uv.y = 0;
        return;
        // return Point2(0.5, 0.5);
    }
    Float a, b;
    if (phi < -M_PI / 4)
        phi += 2 * M_PI;
    if (phi < M_PI / 4)
    {
        a = r;
        b = phi * a / (M_PI / 4);
    }
    else if (phi < M_PI * 3 / 4)
    {
        b = r;
        a = -(phi - M_PI / 2) * b / (M_PI / 4);
    }
    else if (phi < M_PI * 5 / 4)
    {
        a = -r;
        b = (phi - M_PI) * a / (M_PI / 4);
    }
    else
    {
        b = -r;
        a = -(phi - M_PI * 3 / 2) * b / (M_PI / 4);
    }
    uv.x = (a + 1) / 2;
    uv.y = (b + 1) / 2;
    //return Point2((a + 1) / 2, (b + 1) / 2);
}

inline bool pointInAABB(const Point3& p, const AABB& aabb) {
    return (p.x >= aabb.min.x && p.x <= aabb.max.x &&
        p.y >= aabb.min.y && p.y <= aabb.max.y &&
        p.z >= aabb.min.z && p.z <= aabb.max.z);
}

MTS_NAMESPACE_END