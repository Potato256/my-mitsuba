#include <mitsuba/render/scene.h>

MTS_NAMESPACE_BEGIN

#define MASK_h27b 0xffffffe0
#define MASK_l5b 0x0000001f
#define MASK_l27b 0x07ffffff
#define PI 3.1415926535

template <int omSize, int omDepth>
class OccupancyMap
{
private:
    int bom[omSize][omSize][omDepth];
    AABB m_AABB;
    Float m_size;
    Float m_gridSize;

public:
    OccupancyMap() {}

    inline void clear()
    {
        memset(bom, 0, sizeof(bom));
    }

    inline void setAABB(const AABB &aabb)
    {
        m_AABB = aabb;
    }

    inline void setSize(const Float d)
    {
        m_size = d;
        m_gridSize = m_size / omSize;
    }

    inline void set(int x, int y, int z)
    {
        bom[x][y][(z >> 5) & MASK_l27b] |= 1 << (z & MASK_l5b);
    }

    inline void setArray(int *array, int x, int y, int z)
    {
        array[(z >> 5) & MASK_l27b] |= 1 << (z & MASK_l5b);
    }

    inline bool check(int x, int y, int z) const
    {
        return x >= 0 && x < omSize && y >= 0 && y < omSize && z >= 0 && z < omSize;
    }

    inline bool get(int x, int y, int z) const
    {
        return bom[x][y][(z >> 5) & MASK_l27b] & (1 << (z & MASK_l5b));
    }

    bool rayIntersect(const Ray &ray, Float &nearT) const
    {
        Float farT;
        if (m_AABB.rayIntersect(ray, nearT, farT))
        {
            Point p = ray.o + ray.d * nearT;
            int x = (int)floor((p.x - m_AABB.min.x) / m_gridSize + Epsilon);
            int y = (int)floor((p.y - m_AABB.min.y) / m_gridSize + Epsilon);
            int z = (int)floor((p.z - m_AABB.min.z) / m_gridSize + Epsilon);
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

    static Quaternion concentricMap(const Point2 &uv)
    {
        Float x = uv.x * 2 - 1;
        Float y = uv.y * 2 - 1;
        Float phi, r;
        if (x > -y)
            if (x > y)
            {
                r = x;
                phi = (PI / 4) * (y / x);
            }
            else
            {
                r = y;
                phi = (PI / 4) * (2 - x / y);
            }
        else if (x < y)
        {
            r = -x;
            phi = (PI / 4) * (4 + y / x);
        }
        else
        {
            r = -y;
            if (y != 0)
                phi = (PI / 4) * (6 - x / y);
            else
                phi = 0;
        }
        return normalize(Quaternion::fromDirectionPair(Vector3(0, 0, 1), Vector3(cos(phi) * r, sin(phi) * r, sqrt(1 - r * r))));
    }

    void generateROMA(OccupancyMap *omarray, Point2 uv)
    {
        Quaternion q = concentricMap(uv);
        SLog(EDebug, "q %f %f %f %f", q.v.x, q.v.y, q.v.z, q.w);
        for (int x = 0; x < omSize; x++)
            for (int y = 0; y < omSize; y++)
            {
                Vector3 x_start(Float(x - omSize/2), Float(y - omSize/2), 0.5 - omSize/2);
                Vector3 x_end(Float(x - omSize/2), Float(y - omSize/2), omSize/2 - 1.4);
                // rotate
                Vector3 x_start_rot = (q * Quaternion(x_start, 0) * Quaternion(-q.v, q.w)).v + Vector3(omSize/2);
                Vector3 x_end_rot = (q * Quaternion(x_end, 0) * Quaternion(-q.v, q.w)).v + Vector3(omSize/2);
                Vector3 v_step = (x_end_rot - x_start_rot) / Float(omSize - 1);
                for (int i = 0; i < omSize; i++)
                {
                    int base_x = (int)floor(x_start_rot.x);
                    int base_y = (int)floor(x_start_rot.y);
                    int base_z = (int)floor(x_start_rot.z);
                    // SLog(EDebug, "base %d %d %d", base_x, base_y, base_z);
                    if (check(base_x, base_y, base_z) && get(base_x, base_y, base_z))
                    {
                        setArray(omarray->bom[x][y], x, y, i);
                        SLog(EDebug, "set %d %d %d", x, y, i);
                    }
                    x_start_rot += v_step;
                }
            }
        // SLog(EDebug, "omarray");
        // SLog(EDebug, omarray->toString().c_str());
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
                    if ((i + j + k) & 1)
                        set(i, j, k);
    }

    void testSetBallPattern()
    {
        int c = omSize / 2;
        for (int i = 0; i < omSize; ++i)
            for (int j = 0; j < omSize; ++j)
                for (int k = 0; k < omSize; ++k)
                {
                    int r2 = (i - c) * (i - c) + (j - c) * (j - c) + (k - c) * (k - c);
                    if (r2 < omSize * omSize / 4)
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