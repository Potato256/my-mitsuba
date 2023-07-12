#include <mitsuba/render/scene.h>

MTS_NAMESPACE_BEGIN

struct BDPTVertex{
    Point3 pos;
    Normal n;
    Vector wi;
    Spectrum value;
    Point2 uv;
    const BSDF *bsdf;
    const Emitter *e;
    Float pdf;
    std::string toString() const {
        std::ostringstream oss;
        oss << "pos: " +  pos.toString() + "\n" 
            << "n: " +    n.toString() + "\n"
            << "wi: " +   wi.toString() + "\n"
            << "uv: " <<  uv.toString() + "\n"
            << "pdf: "<<  pdf << "\n"
            << "value: "+ value.toString() + "\n";
        return oss.str();
    }    
};

MTS_NAMESPACE_END