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
    /* Trace septh */
    int depth;

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

/// wi point outwards
Float computePdfForward(Vector3& wi, BDPTVertex* mid, BDPTVertex* next) {
    Intersection its;
    its.p = mid->pos;
    its.shFrame = Frame(mid->n);
    its.uv = mid->uv;
    its.hasUVPartials = 0;
    Vector3 d = next->pos - mid->pos;
    Float distSquared = d.lengthSquared();
    d /= sqrt(distSquared);
    BSDFSamplingRecord bRec(its, its.toLocal(wi), its.toLocal(d), ERadiance);
    Float pdfBsdf = mid->bsdf->pdf(bRec);
    return pdfBsdf * absDot(next->n, d) / distSquared;
}

Float computePdfLightDir(BDPTVertex* l, BDPTVertex* e) {
    PositionSamplingRecord pRec(0.0f);
    pRec.n = l->n;
    Vector3 d = e->pos - l->pos;
    Float distSquared = d.lengthSquared();
    d /= sqrt(distSquared);
    DirectionSamplingRecord dRec(d);
    Float ans = l->e->pdfDirection(dRec, pRec) * absDot(e->n, d) / distSquared;
    return ans;        
}

MTS_NAMESPACE_END