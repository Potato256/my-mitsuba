#pragma once
#include <mitsuba/render/scene.h>
#include "helpers.h"

MTS_NAMESPACE_BEGIN

#define OMSIZE 128
#define OMDEPTH OMSIZE / 32
#define OMNUMSQRT 16
#define OMNUM (OMNUMSQRT * OMNUMSQRT)

#define MASK_h27b 0xffffffe0
#define MASK_l27b 0x07ffffff
#define MASK_l5b 0x0000001f

template <int omSize, int omDepth>
class OccupancyMap
{
private:
    int bom[omSize][omSize][omDepth];
    AABB m_AABB;
    Float m_size;
    Float m_gridSize;
    Float m_gridSizeRecp;
    Quaternion m_q;
    Transform m_rotate;
    Vector3 m_center;

public:
    OccupancyMap() {}

    inline void clear()
    {
        memset(bom, 0, sizeof(bom));
    }

    inline void setAABB(const AABB &aabb)
    {
        m_AABB = aabb;
        m_center = Vector3(m_AABB.min + (m_AABB.max - m_AABB.min) / 2);
        Float d = m_AABB.max.x - m_AABB.min.x;
        m_size = d;
        m_gridSize = m_size / omSize;
        m_gridSizeRecp = 1 / m_gridSize;
    }

    inline void setSize(const Float d)
    {
        m_size = d;
        m_gridSize = m_size / omSize;
        m_gridSizeRecp = 1 / m_gridSize;
    }

    inline void set(int x, int y, int z)
    {
        bom[x][y][z >> 5] |= 1 << (z & MASK_l5b);
    }

    inline void set(Point3i p)
    {
        set(p.x, p.y, p.z);
    }

    inline void setArray(int *array, int x, int y, int z)
    {
        array[(z >> 5) & MASK_l27b] |= 1 << (z & MASK_l5b);
    }

    inline bool check(int x, int y, int z) const
    {
        return x >= 0 && x < omSize && y >= 0 && y < omSize && z >= 0 && z < omSize;
    }

    inline bool check(Point3i p) const
    {
        return check(p.x, p.y, p.z);
    }

    inline bool get(int x, int y, int z) const
    {
        return bom[x][y][z >> 5] & (1 << (z & MASK_l5b));
    }

    inline bool get(Point3i p) const
    {
        return get(p.x, p.y, p.z);
    }

    inline bool anyHit(int x, int y) const
    {
        for (int i = 0; i < omDepth; i++)
            if (bom[x][y][i])
                return true;
        return false;
    }

    inline bool closestHit(int U, Float z, Float &id)
    {
        if (U > 0)
        {
            if (z > 0)
            {
                id = 32 - floor(log2(U)) - 0.5;
            }
            else
            {
                id = 32 - log2(U & (-U));
            }
            return true;
        }
        return false;
    }

    inline void setScene(const Scene *scene)
    {
        auto meshes = scene->getMeshes();
        for (auto m : meshes)
            setMesh(m);
    }

    inline void setMesh(const TriMesh *mesh)
    {
        const Triangle *tri = mesh->getTriangles();
        const Point3 *pos = mesh->getVertexPositions();
        int triCnt = (int)mesh->getTriangleCount();
        for (int i = 0; i < triCnt; ++i)
        {
            Point3 p1, p2, p3;
            getGridIndexf(pos[tri[i].idx[0]], p1);
            getGridIndexf(pos[tri[i].idx[1]], p2);
            getGridIndexf(pos[tri[i].idx[2]], p3);
            setTriangle(p1, p2, p3);
        }
        return;
    }

    void setTriangle(Point3 &p0, Point3 &p1, Point3 &p2, int depth = 0)
    {
        Point3i p0i, p1i, p2i;
        getGridIndexf2i(p0, p0i);
        getGridIndexf2i(p1, p1i);
        getGridIndexf2i(p2, p2i);
        set(p0i);
        set(p1i);
        set(p2i);
        if (closeEnough(p0i, p1i, p2i))
            return;
        Point3i p01i, p12i, p20i;
        Point3 p01 = p0 + (p1 - p0) / 2;
        Point3 p12 = p1 + (p2 - p1) / 2;
        Point3 p20 = p2 + (p0 - p2) / 2;

        setTriangle(p0, p01, p20);
        setTriangle(p1, p12, p01);
        setTriangle(p2, p20, p12);
        setTriangle(p01, p12, p20);
        return;
    }

    inline bool closeEnough(Point3i &p1, Point3i &p2, Point3i &p3) const
    {
        Vector3i v12 = p2 - p1;
        Vector3i v23 = p3 - p2;
        Vector3i v31 = p1 - p3;
        return length(v12) + length(v23) + length(v31) <= 4;
    }

    inline int length(Vector3i &v) const
    {
        return abs(v.x) + abs(v.y) + abs(v.z);
    }

    inline void getGridIndexf(const Point3 &p, Point3 &p1) const
    {
        p1.x = (p.x - m_AABB.min.x) * m_gridSizeRecp;
        p1.y = (p.y - m_AABB.min.y) * m_gridSizeRecp;
        p1.z = (p.z - m_AABB.min.z) * m_gridSizeRecp;
    }

    inline void getGridDirf(const Vector3 &d, Vector3 &d1) const
    {
        d1.x = d.x * m_gridSizeRecp;
        d1.y = d.y * m_gridSizeRecp;
        d1.z = d.z * m_gridSizeRecp;
    }

    inline void getGridIndexf2i(const Point3 &p, Point3i &pi) const
    {
        /* Assumes x,y,z > 0 */
        pi.x = (int)(p.x);
        pi.y = (int)(p.y);
        pi.z = (int)(p.z);
    }

    inline void getGridIndexi(const Point &p, Point3i &pi) const
    {
        pi.x = (int)floor((p.x - m_AABB.min.x) * m_gridSizeRecp);
        pi.y = (int)floor((p.y - m_AABB.min.y) * m_gridSizeRecp);
        pi.z = (int)floor((p.z - m_AABB.min.z) * m_gridSizeRecp);
    }

    bool rayInAABB(const Ray &ray, Float &nearT) const
    {
        if (ray.o.x > m_AABB.min.x && ray.o.x < m_AABB.max.x &&
            ray.o.y > m_AABB.min.y && ray.o.y < m_AABB.max.y &&
            ray.o.z > m_AABB.min.z && ray.o.z < m_AABB.max.z)
        {
            nearT = 0;
            return true;
        }
        return false;
    }

    bool rayIntersect(const Ray &ray, Float &nearT) const
    {
        Float farT;
        if (rayInAABB(ray, nearT) || m_AABB.rayIntersect(ray, nearT, farT))
        {
            Point p = ray.o + ray.d * (nearT + Epsilon);
            int x = (int)floor((p.x - m_AABB.min.x) * m_gridSizeRecp);
            int y = (int)floor((p.y - m_AABB.min.y) * m_gridSizeRecp);
            int z = (int)floor((p.z - m_AABB.min.z) * m_gridSizeRecp);
            while (check(x, y, z))
            {
                if (get(x, y, z))
                    return true;
                int sx = ray.d.x > 0 ? 1 : (ray.d.x < 0 ? -1 : 0);
                int sy = ray.d.y > 0 ? 1 : (ray.d.y < 0 ? -1 : 0);
                int sz = ray.d.z > 0 ? 1 : (ray.d.z < 0 ? -1 : 0);
                int nx = x + sx;
                int ny = y + sy;
                int nz = z + sz;

                Float dx = nx * m_gridSize + m_AABB.min.x - p.x;
                Float dy = ny * m_gridSize + m_AABB.min.y - p.y;
                Float dz = nz * m_gridSize + m_AABB.min.z - p.z;

                Float tx = sx == 0 ? 1e30f : dx * ray.dRcp.x;
                Float ty = sy == 0 ? 1e30f : dy * ray.dRcp.y;
                Float tz = sz == 0 ? 1e30f : dz * ray.dRcp.z;
                if (tx < ty)
                {
                    if (tx < tz)
                    {
                        x = nx;
                        nearT += tx;
                    }
                    else
                    {
                        z = nz;
                        nearT += tz;
                    }
                }
                else
                {
                    if (ty < tz)
                    {
                        y = ny;
                        nearT += ty;
                    }
                    else
                    {
                        z = nz;
                        nearT += tz;
                    }
                }
                p = ray.o + ray.d * nearT;
                // return false;
            }
        }
        return false;
    }

    inline Float marchOneCube(const Point3 &p, const Vector3 &d) const
    {
        Point3 p1;
        Point3i pi;
        Vector3 d1;
        getGridIndexf(p, p1);
        getGridIndexf2i(p1, pi);
        getGridDirf(d, d1);
        Float t = 1e30f;
        if (d1.x != 0)
            t = std::min(t, d1.x > 0 ? (pi.x + 1 - p1.x) / d1.x : (pi.x - p1.x) / d1.x);
        if (d1.y != 0)
            t = std::min(t, d1.y > 0 ? (pi.y + 1 - p1.y) / d1.y : (pi.y - p1.y) / d1.y);
        if (d1.z != 0)
            t = std::min(t, d1.z > 0 ? (pi.z + 1 - p1.z) / d1.z : (pi.z - p1.z) / d1.z);
        if (t < 0)
            SLog(EError, "negative step");
        return t;
    }

    bool visibilityBOM(const Point3 &o1, const Point3 &o2) const
    {
        // Vector3 o1_aligned = (Quaternion(-m_q.v, m_q.w) * Quaternion(Vector(o1 - m_center), 0) * m_q).v + m_center;
        // Vector3 o2_aligned = (Quaternion(-m_q.v, m_q.w) * Quaternion(Vector(o2 - m_center), 0) * m_q).v + m_center;
        Point3 o1_aligned = m_rotate(o1 - m_center) + m_center;
        Point3 o2_aligned = m_rotate(o2 - m_center) + m_center;
        Point3 p1 = Point3(o1_aligned);
        Point3 p2 = Point3(o2_aligned);
        Vector3 d = p2 - p1;
        Float dLength = d.length();
        d = d / dLength;

        // for (int i = 0; i < 1; i++)
        // {
        //     if (pointInAABB(p1, m_AABB))
        //     {
        //         Float t = marchOneCube(p1, d);
        //         p1 = p1 + d * (t + Epsilon);
        //         dLength -= t;
        //     }
        //     if (pointInAABB(p2, m_AABB))
        //     {
        //         Float t = marchOneCube(p2, -d);
        //         p2 = p2 - d * (t + Epsilon);
        //         dLength -= t;
        //     }
        //     if (dLength < Epsilon)
        //         return true;
        // }
        while (dLength > Epsilon)
        {
            Point3i p1i;
            getGridIndexi(p1, p1i);
            if (check(p1i))
            {
                if (get(p1i))
                    return false;
            }
            else
            {
                return true;
            }
            Float t = marchOneCube(p1, d);
            p1 = p1 + d * (t + Epsilon);
            dLength -= t;
        }
        return true;
        // Float nearT;
        // Ray r;
        // r.o = p2;
        // r.d = -d;
        // if (rayIntersect(r, nearT))
        //     return nearT > dLength - Epsilon;
        // else
        //     return true;
    }

    bool Trace(const Ray &ray, Float &nearT) const
    {
        // Vector3 o_aligned = (Quaternion(-m_q.v, m_q.w) * Quaternion(Vector(ray.o - m_center), 0) * m_q).v + m_center;
        // Vector3 d_aligned = (Quaternion(-m_q.v, m_q.w) * Quaternion(Vector(ray.d), 0) * m_q).v;
        Point3 o1_aligned = m_rotate(o1 - m_center) + m_center;
        Point3 o2_aligned = m_rotate(o2 - m_center) + m_center;
        int x = (int)floor((o_aligned.x - m_AABB.min.x) * m_gridSizeRecp + Epsilon);
        int y = (int)floor((o_aligned.y - m_AABB.min.y) * m_gridSizeRecp + Epsilon);
        int z = (int)floor((o_aligned.z - m_AABB.min.z) * m_gridSizeRecp + Epsilon);
        // SLog(EDebug, "x %d y %d z %d", x, y, z);

        if (!check(x, y, 0))
            return false;

        // any hit
        return anyHit(x, y);
    }

    bool Visible(const Point3 &o1, const Point3 &o2) const
    {
        // Vector3 o1_aligned = (Quaternion(-m_q.v, m_q.w) * Quaternion(Vector(o1 - m_center), 0) * m_q).v + m_center;
        // Vector3 o2_aligned = (Quaternion(-m_q.v, m_q.w) * Quaternion(Vector(o2 - m_center), 0) * m_q).v + m_center;
        Point3 o1_aligned = m_rotate(o1 - m_center) + m_center;
        Point3 o2_aligned = m_rotate(o2 - m_center) + m_center;
        Float x1 = (o1_aligned.x - m_AABB.min.x) * m_gridSizeRecp + Epsilon;
        Float y1 = (o1_aligned.y - m_AABB.min.y) * m_gridSizeRecp + Epsilon;
        int x = (int)floor(x1);
        int y = (int)floor(y1);

        if (!check(x, y, 0))
            return true;
        // SLog(EError, "test\n");
        int z1 = (int)floor((o1_aligned.z - m_AABB.min.z) * m_gridSizeRecp + Epsilon);
        int z2 = (int)floor((o2_aligned.z - m_AABB.min.z) * m_gridSizeRecp + Epsilon);
        if (z1 > z2)
        {
            int t = z1;
            z1 = z2;
            z2 = t;
        }
        if (z2 - z1 < 2)
        {
            return true;
        }
        z1 += 1;
        z2 -= 1;
        if (z1 < 0)
            z1 = 0;
        else if (z1 >= omSize)
            z1 = omSize - 1;
        if (z2 < 0)
            z2 = 0;
        else if (z2 >= omSize)
            z2 = omSize - 1;
        int p1 = z1 >> 5;
        int p2 = z2 >> 5;
        int r1 = z1 & MASK_l5b;
        int r2 = 31 - z2 & MASK_l5b;
        if (p1 == p2)
        {
            return ((bom[x][y][p1] >> r1) << (r1 + r2)) == 0;
        }
        else
        {
            if (bom[x][y][p1] >> r1 != 0)
                return false;
            for (int i = p1 + 1; i < p2; i++)
            {
                if (bom[x][y][i] != 0)
                    return false;
            }
            if (bom[x][y][p2] << r2 != 0)
                return false;
        }
        return true;
    }

    Quaternion concentricMap(const Point2 &uv)
    {
        Float x = uv.x * 2 - 1;
        Float y = uv.y * 2 - 1;
        Float phi, r;
        if (x > -y)
            if (x > y)
            {
                r = x;
                phi = (M_PI / 4) * (y / x);
            }
            else
            {
                r = y;
                phi = (M_PI / 4) * (2 - x / y);
            }
        else if (x < y)
        {
            r = -x;
            phi = (M_PI / 4) * (4 + y / x);
        }
        else
        {
            r = -y;
            if (y != 0)
                phi = (M_PI / 4) * (6 - x / y);
            else
                phi = 0;
        }
        Float z = 1 - r * r;
        return normalize(Quaternion::fromDirectionPair(Vector3(0, 0, 1), Vector3(cos(phi) * sqrt(1 - z * z) / r, sin(phi) * sqrt(1 - z * z) / r, z)));
    }

    Quaternion generateROMA(OccupancyMap *omarray, Point2 uv)
    {
        /* base direction ---> ray direction */
        Quaternion q = concentricMap(uv);
        omarray->m_q = Quaternion(q);
        omarray->m_rotate = q.toTransform().inverse(); // inversed ratation matrix
        // SLog(EDebug, "q %f %f %f %f", q.v.x, q.v.y, q.v.z, q.w);

        for (int x = 0; x < omSize; x++)
            for (int y = 0; y < omSize; y++)
            {
                Float radiu = Float(omSize) / 2.0f;
                Vector3 x_start(Float(x - radiu), Float(y - radiu), 0.5f - radiu);
                Vector3 x_end(Float(x - radiu), Float(y - radiu), radiu - 0.5f);
                // rotate
                Vector3 x_start_rot = (q * Quaternion(x_start, 0) * Quaternion(-q.v, q.w)).v + Vector3(radiu);
                Vector3 x_end_rot = (q * Quaternion(x_end, 0) * Quaternion(-q.v, q.w)).v + Vector3(radiu);
                Vector3 v_step = (x_end_rot - x_start_rot) / Float(omSize - 1);
                for (int i = 0; i < omSize; i++)
                {
                    int base_x = (int)floor(x_start_rot.x + Epsilon);
                    int base_y = (int)floor(x_start_rot.y + Epsilon);
                    int base_z = (int)floor(x_start_rot.z + Epsilon);
                    // SLog(EDebug, "base %d %d %d", base_x, base_y, base_z);
                    if (check(base_x, base_y, base_z) && get(base_x, base_y, base_z))
                    {
                        setArray(omarray->bom[x][y], x, y, i);
                    }
                    x_start_rot += v_step;
                }
            }
        // SLog(EDebug, "omarray");
        // SLog(EDebug, omarray->toString().c_str());
        return q;
    }

    void testSetAll()
    {
        for (int i = 0; i < omSize; ++i)
            for (int j = 0; j < omSize; ++j)
                for (int k = 0; k < omSize; ++k)
                    set(i, j, k);
    }

    void testSetBoxPattern()
    {
        for (int i = 0; i < omSize; ++i)
            for (int j = 0; j < omSize; ++j)
                for (int k = 0; k < omSize; ++k)
                {
                    if ((i + j + k) & 1)
                        set(i, j, k);
                }
    }

    void testSetBallPattern()
    {
        int c = omSize / 2;
        for (int i = 0; i < omSize; ++i)
            for (int j = 0; j < omSize; ++j)
                for (int k = 0; k < omSize; ++k)
                {
                    int r2 = (i - c) * (i - c) + (j - c) * (j - c) + (k - c) * (k - c);
                    if (r2 < omSize * omSize / 16)
                        set(i, j, k);
                }
    }

    static inline int nearestOMindex(Vector3 d)
    {
        if (d.z < 0)
            d = -d;
        Point2 uv = direct2uv(d);
        if (uv.x > 0.999999)
            uv.x = 0.999999;
        if (uv.y > 0.999999)
            uv.y = 0.999999;
        return int(floor(uv.x * OMNUMSQRT)) * OMNUMSQRT + int(floor(uv.y * OMNUMSQRT));
    }

    std::string toString() const
    {
        std::ostringstream oss;
        oss << "BOM[" << endl
            << "  AABB: " << m_AABB.toString() << "," << endl
            << "  size = " << m_size << "," << endl
            << "  gridSize = " << m_gridSize << endl
            << "  bomSize = " << omSize * omSize * omSize / 8 / 1024 << " KB" << endl
            << "  OM:\n";

        // for (int i = 0; i < omSize; ++i)
        //     for (int j = 0; j < omSize; ++j)
        //     {
        //         oss << "    ";
        //         for (int k = 0; k < omSize; ++k)
        //             oss << get(i, j, k);
        //         oss << endl;
        //  }
        oss << "  ]";
        return oss.str();
    }
};

typedef OccupancyMap<OMSIZE, OMDEPTH> OM;

MTS_NAMESPACE_END