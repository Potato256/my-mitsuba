#include <mitsuba/render/scene.h>

MTS_NAMESPACE_BEGIN

#define MASK_h27b 0xffffffe0
#define MASK_l5b 0x0000001f


template<int omSize, int omDepth>
class OccupancyMap{
private:
    int bom[omSize][omSize][omDepth];
    AABB m_AABB;
    Float m_size;
    Float m_gridSize;

public:
    OccupancyMap() {}

    inline void clear(){
        memset(bom, 0, sizeof(bom));
    }

    inline void setAABB(const AABB &aabb){
        m_AABB = aabb;
    }
    
    inline void setSize(const Float d){
        m_size = d;
        m_gridSize = m_size / omSize;
    }

    inline void set(int x, int y, int z){
        bom[x][y][z & MASK_h27b] |= 1 << (z & MASK_l5b);
    }

    inline bool check(int x, int y, int z) const {
        return x >= 0 && x < omSize && y >= 0 && y < omSize && z >= 0 && z < omSize;
    }

    inline bool get(int x, int y, int z) const {
        return bom[x][y][z & MASK_h27b] & (1 << (z & MASK_l5b));
    }

    bool rayIntersect(const Ray &ray, Float &nearT) const {
        Float farT;
        if (m_AABB.rayIntersect(ray, nearT, farT)){
            Point p = ray.o + ray.d * nearT;
            int x = (int) floor((p.x - m_AABB.min.x) / m_gridSize + Epsilon);
            int y = (int) floor((p.y - m_AABB.min.y) / m_gridSize + Epsilon);
            int z = (int) floor((p.z - m_AABB.min.z) / m_gridSize + Epsilon);
            while (check(x, y, z)){
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

                if (tx < ty){
                    if (tx < tz){
                        x = nx;
                        nearT += tx;
                    }
                    else{
                        z = nz;
                        nearT += tz;
                    }
                }
                else{
                    if (ty < tz){
                        y = ny;
                        nearT += ty;
                    }
                    else{
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

    void testSetAll(){
        for (int i=0; i<omSize; ++i)
            for (int j=0; j<omSize; ++j)
                for (int k=0; k<omSize; ++k)
                    set(i, j, k);
    }

    void testSetBoxPattern(){
        for (int i=0; i<omSize; ++i)
            for (int j=0; j<omSize; ++j)
                for (int k=0; k<omSize; ++k)
                    if ( (i+j+k)&1 )
                        set(i, j, k);
    }

    void testSetBallPattern(){
        int c = omSize/2;
        for (int i=0; i<omSize; ++i)
            for (int j=0; j<omSize; ++j)
                for (int k=0; k<omSize; ++k){
                    int r2 = (i-c)*(i-c)+(j-c)*(j-c)+(k-c)*(k-c);
                    if ( r2 < omSize*omSize/4 )
                        set(i, j, k);
                }
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "BOM[" << endl
            << "  AABB: " << m_AABB.toString() << "," << endl
            << "  size = " << m_size << "," << endl
            << "  gridSize = " << m_gridSize << endl
            << "  OM:\n";

        for (int i=0; i<omSize; ++i)
            for (int j=0; j<omSize; ++j){
                oss << "    ";
                for (int k=0; k<omSize; ++k)
                    oss << get(i, j, k);
                oss << endl;
            }
        oss << "  ]";
        return oss.str();
    }

};

MTS_NAMESPACE_END