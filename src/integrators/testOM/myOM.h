#include <mitsuba/render/scene.h>

MTS_NAMESPACE_BEGIN

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

public:
    Quaternion m_q;
    Vector3 m_center;
    OccupancyMap() {}

    inline void clear()
    {
        memset(bom, 0, sizeof(bom));
    }

    inline void setAABB(const AABB &aabb)
    {
        m_AABB = aabb;
        m_center = Vector(m_AABB.min + (m_AABB.max - m_AABB.min) / 2);
    }

    inline void setSize(const Float d)
    {
        m_size = d;
        m_gridSize = m_size / omSize;
        m_gridSizeRecp = 1 / m_gridSize;
    }

    inline void set(int x, int y, int z)
    {
        bom[x][y][(z & MASK_h27b) >> 5] |= 1 << (z & MASK_l5b);
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
        return bom[x][y][(z & MASK_h27b) >> 5] & (1 << (z & MASK_l5b));
    }

    inline void get(Point3i p) const
    {
        get(p.x, p.y, p.z);
    }

    inline bool anyhit(int x, int y) const
    {
        for (int i = 0; i < omDepth; i++)
            if (bom[x][y][i])
                return true;
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

    bool rayIntersect(const Ray &ray, Float &nearT) const
    {
        /* TBD: When camera is in AABB? */
        Float farT;
        if (m_AABB.rayIntersect(ray, nearT, farT))
        {
            Point p = ray.o + ray.d * nearT;
            int x = (int)floor((p.x - m_AABB.min.x) * m_gridSizeRecp + Epsilon);
            int y = (int)floor((p.y - m_AABB.min.y) * m_gridSizeRecp + Epsilon);
            int z = (int)floor((p.z - m_AABB.min.z) * m_gridSizeRecp + Epsilon);
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

    bool Trace(const Ray &ray, Float &nearT) const
    {
        // SLog(EDebug, "m_q %f %f %f %f", m_q.v.x, m_q.v.y, m_q.v.z, m_q.w);
        // SLog(EDebug, "o %f %f %f", ray.o.x, ray.o.y, ray.o.z);
        // SLog(EDebug, "d %f %f %f", ray.d.x, ray.d.y, ray.d.z);
        // SLog(EDebug, "center %f %f %f", m_center.x, m_center.y, m_center.z);
        Vector3 o_aligned = (Quaternion(-m_q.v, m_q.w) * Quaternion(Vector(ray.o - m_center), 0) * m_q).v + m_center;
        Vector3 d_aligned = (Quaternion(-m_q.v, m_q.w) * Quaternion(Vector(ray.d), 0) * m_q).v;
        // Point o_aligned = ray.o;

        // if (d_aligned.z < 0.95)
        // {
        // SLog(EDebug, "o_aligned %f %f %f", o_aligned.x, o_aligned.y, o_aligned.z);
        // SLog(EDebug, "d_aligned %f %f %f", d_aligned.x, d_aligned.y, d_aligned.z);
        // }
        // Ray ray_aligned(ray);
        // ray_aligned.d = d_aligned;
        // ray_aligned.o = Point(o_aligned.x, o_aligned.y, o_aligned.z);
        // // SLog(EDebug, "rayintersect %d", rayIntersect(ray, nearT));
        // if (!rayIntersect(ray_aligned, nearT))
        //     return false;
        // return true;

        int x = (int)floor((o_aligned.x - m_AABB.min.x) * m_gridSizeRecp + Epsilon);
        int y = (int)floor((o_aligned.y - m_AABB.min.y) * m_gridSizeRecp + Epsilon);
        int z = (int)floor((o_aligned.z - m_AABB.min.z) * m_gridSizeRecp + Epsilon);
        // SLog(EDebug, "x %d y %d z %d", x, y, z);

        if (!check(x, y, 0))
            return false;

        // any hit
        return anyhit(x, y);
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
        return normalize(Quaternion::fromDirectionPair(Vector3(0, 0, 1), Vector3(cos(phi) * r, sin(phi) * r, sqrt(1 - r * r))));
    }

    Quaternion generateROMA(OccupancyMap *omarray, Point2 uv)
    {
        /* base direction ---> ray direction */
        Quaternion q = concentricMap(uv);
        omarray->m_q = Quaternion(q);
        // SLog(EDebug, "q %f %f %f %f", q.v.x, q.v.y, q.v.z, q.w);

        for (int x = 0; x < omSize; x++)
            for (int y = 0; y < omSize; y++)
            {
                Float radiu = Float(omSize) / 2.0;
                Vector3 x_start(Float(x - radiu), Float(y - radiu), 0.5 - radiu);
                Vector3 x_end(Float(x - radiu), Float(y - radiu), radiu - 0.5);
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
        int c = omSize / 2;
        for (int i = 0; i < omSize; ++i)
            for (int j = 0; j < omSize; ++j)
                for (int k = 0; k < omSize; ++k)
                {
                    // if ((i + j + k) & 1)
                    if (check(2 * i - c, 2 * j - c, 2 * k - c))
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

    std::string toString() const
    {
        std::ostringstream oss;
        oss << "BOM[" << endl
            << "  AABB: " << m_AABB.toString() << "," << endl
            << "  size = " << m_size << "," << endl
            << "  gridSize = " << m_gridSize << endl
            << "  OM:\n";

        for (int i = 0; i < omSize; ++i)
            for (int j = 0; j < omSize; ++j)
            {
                oss << "    ";
                for (int k = 0; k < omSize; ++k)
                    oss << get(i, j, k);
                oss << endl;
            }
        oss << "  ]";
        return oss.str();
    }
};

MTS_NAMESPACE_END