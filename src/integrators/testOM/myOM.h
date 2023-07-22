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
            int x = (int) floor((p.x - m_AABB.min.x) / m_gridSize);
            int y = (int) floor((p.y - m_AABB.min.y) / m_gridSize);
            int z = (int) floor((p.z - m_AABB.min.z) / m_gridSize);
            if (check(x, y, z) && get(x, y, z)){
                return true;
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

    void testSetPattern(){
        for (int i=0; i<omSize; ++i)
            for (int j=0; j<omSize; ++j)
                for (int k=0; k<omSize; ++k)
                    if ( (i+j+k)&1 )
                        set(i, j, k);
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