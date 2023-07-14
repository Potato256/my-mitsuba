#include <mitsuba/render/scene.h>

MTS_NAMESPACE_BEGIN

struct BDPTVertex{
    Point3 pos;
    Normal n;
    Vector3 wi;
    Spectrum value;
    Point2 uv;
    const BSDF *bsdf;
    const Emitter *e;
    /* The local pdf of sampling this vertex */
    Float pdf;
    /* The local pdf of sampling last vertex inversely */
    Float pdfInverse;
    /* The pdf of sampling this vertex's position on light */
    Float pdfLight;

    std::string toString() const {
        std::ostringstream oss;
        oss << "pos:        "<< pos.toString() + "\n" 
            << "n:          "<< n.toString() + "\n"
            << "wi:         "<< wi.toString() + "\n"
            << "value:      "<< value.toString() + "\n"
            << "uv:         "<< uv.toString() + "\n"
            << "pdf:        "<<  pdf << "\n"
            << "pdfInverse: "<<  pdfInverse << "\n"
            << "pdfLight:   "<<  pdfLight << "\n";
        return oss.str();
    }    
};

MTS_NAMESPACE_END